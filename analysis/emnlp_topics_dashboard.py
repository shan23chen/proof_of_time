"""Interactive EMNLP topic explorer with rich filtering and summary exports."""

from __future__ import annotations

import argparse
import textwrap
import zipfile
import math
from dataclasses import dataclass
from functools import lru_cache
import io
from pathlib import Path
from typing import Iterable, Sequence

from collections.abc import Iterable as IterableABC

import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, State, dash_table, dcc, html
from dash.exceptions import PreventUpdate

try:
    from datasets import load_from_disk
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise SystemExit(
        "The 'datasets' package is required. Install it with `pip install datasets dash`."
    ) from exc


DATASET_DIR = Path("content/emnlp_with_topics_benchmarks")
DATASET_ZIP = Path("emnlp_with_topics_benchmarks.zip")
EXPORT_DIR = Path("reports")
SUMMARY_CSV = EXPORT_DIR / "emnlp_topics_summary.csv"
FINE_TOPIC_CSV = EXPORT_DIR / "emnlp_finetopic_summary.csv"
BENCHMARK_CSV = EXPORT_DIR / "emnlp_benchmark_summary.csv"
BENCHMARK_YEARLY_CSV = EXPORT_DIR / "emnlp_benchmark_yearly.csv"
DOMAIN_CSV = EXPORT_DIR / "emnlp_domain_summary.csv"
AUTHOR_CSV = EXPORT_DIR / "emnlp_author_summary.csv"

PAGE_STYLE = {"padding": "24px", "fontFamily": "'Helvetica Neue', Arial, sans-serif"}
FILTERS_GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))",
    "gap": "16px",
    "margin": "24px 0",
}
FILTER_CARD_STYLE = {
    "background": "#f7f9fc",
    "border": "1px solid #dde3ea",
    "borderRadius": "8px",
    "padding": "12px",
}
GRAPH_GRID_STYLE = {"display": "grid", "gap": "24px", "margin": "24px 0"}
SUMMARY_GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
    "gap": "16px",
    "marginBottom": "24px",
}
SUMMARY_CARD_STYLE = {
    "background": "#0d1b2a",
    "color": "white",
    "borderRadius": "8px",
    "padding": "16px",
    "textAlign": "center",
}

INSIGHT_CARD_STYLE = {
    "background": "#f0f4f8",
    "border": "1px solid #dde3ea",
    "borderRadius": "8px",
    "padding": "16px",
}

TABLE_CARD_STYLE = {
    "background": "white",
    "border": "1px solid #dde3ea",
    "borderRadius": "8px",
    "padding": "16px",
}

FINE_TOPIC_MIN = 5
FINE_TOPIC_MAX = 60
FINE_TOPIC_DEFAULT = 30


@dataclass(frozen=True)
class TrendSummary:
    topic: str
    share_change: float
    peak_year: int
    peak_share: float
    trough_year: int
    trough_share: float


def _ensure_dataset() -> Path:
    """Ensure the Hugging Face dataset directory exists by extracting the zip."""

    if DATASET_DIR.exists():
        return DATASET_DIR
    if not DATASET_ZIP.exists():
        raise FileNotFoundError(
            "Could not find the EMNLP dataset. Expected 'emnlp_with_topics_benchmarks.zip'."
        )
    with zipfile.ZipFile(DATASET_ZIP) as archive:
        archive.extractall()
    return DATASET_DIR


@lru_cache(maxsize=1)
def _load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load all EMNLP splits into a single pandas DataFrame."""

    dataset_path = _ensure_dataset() if path is None else path
    dataset = load_from_disk(str(dataset_path))

    frames: list[pd.DataFrame] = []
    for split_name, split_ds in dataset.items():
        year = int(split_name.split("_")[-1])
        venue = "Findings" if "findings" in split_name else "Main"
        frame = split_ds.to_pandas()
        frame["year"] = year
        frame["venue"] = venue
        # normalise list fields
        frame["fine_topics"] = frame["fine_topics"].apply(_normalise_sequence)
        frame["topics"] = frame["topics"].apply(_normalise_sequence)
        frame["domain_tags"] = frame["domain_tags"].apply(_normalise_sequence)
        frame["benchmark_tasks"] = frame["benchmark_tasks"].apply(_normalise_sequence)
        if "tags" in frame:
            frame["tags"] = frame["tags"].apply(_normalise_sequence)
        else:
            frame["tags"] = [[] for _ in range(len(frame))]
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined["primary_topic"] = combined["primary_topic"].fillna("Unknown")
    combined["authors"] = combined["authors"].fillna("")
    combined["abstract"] = combined["abstract"].fillna("")
    combined["title"] = combined["title"].fillna("Untitled")
    return combined


def _year_order(years: Iterable[int]) -> list[int]:
    return sorted({int(y) for y in years})


def _normalise_sequence(value) -> list[str]:
    """Coerce nested sequences into sorted, de-duplicated string lists."""

    if value is None:
        return []
    if isinstance(value, float) and math.isnan(value):
        return []
    if isinstance(value, str):
        # Split simple delimiter-separated strings while avoiding single-token noise.
        tokens = [token.strip() for token in value.replace(";", ",").split(",") if token.strip()]
        return sorted(set(tokens)) if len(tokens) > 1 else ([value] if value else [])
    if isinstance(value, IterableABC):
        cleaned: list[str] = []
        for item in value:
            if item is None:
                continue
            if isinstance(item, float) and math.isnan(item):
                continue
            text = str(item).strip()
            if not text:
                continue
            cleaned.append(text)
        return sorted(set(cleaned))
    text = str(value).strip()
    return [text] if text else []


def _deduplicate_preserve_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def _split_authors(text: str | float | None) -> list[str]:
    if text is None:
        return []
    if isinstance(text, float) and math.isnan(text):
        return []
    cleaned = str(text).replace("\n", " ").strip()
    if not cleaned:
        return []
    if ";" in cleaned:
        tokens = [token.strip() for token in cleaned.split(";")]
    else:
        normalised = cleaned.replace(" and ", ",")
        tokens = [token.strip() for token in normalised.split(",")]
    tokens = [token for token in tokens if token]
    return _deduplicate_preserve_order(tokens)


def _top_list_counts(
    df: pd.DataFrame,
    column: str,
    limit: int = 15,
) -> pd.DataFrame:
    exploded = df.explode(column)
    exploded = exploded[exploded[column].notna() & (exploded[column] != "")]
    counts = (
        exploded.groupby(column).size().rename("paper_count").sort_values(ascending=False).reset_index()
    )
    if limit:
        counts = counts.head(limit)
    counts["share"] = counts["paper_count"] / len(df) * 100 if len(df) else 0.0
    return counts


def build_domain_distribution(df: pd.DataFrame) -> pd.DataFrame:
    counts = _top_list_counts(df, "domain_tags", limit=None)
    return counts


def build_top_authors(df: pd.DataFrame, limit: int | None = 20) -> pd.DataFrame:
    author_lists = df["authors"].apply(_split_authors)
    exploded = df.assign(author_list=author_lists).explode("author_list")
    exploded = exploded[exploded["author_list"].notna() & (exploded["author_list"] != "")]
    counts = (
        exploded.groupby("author_list").size().rename("paper_count").sort_values(ascending=False)
    )
    counts = counts.reset_index().rename(columns={"author_list": "author"})
    if limit:
        counts = counts.head(limit)
    counts["share"] = counts["paper_count"] / len(df) * 100 if len(df) else 0.0
    return counts

def _explode_counts(df: pd.DataFrame, column: str, top_n: int | None = None) -> pd.DataFrame:
    exploded = df.explode(column)
    exploded = exploded[exploded[column].notna() & (exploded[column] != "")]
    counts = exploded.groupby([column, "year"]).size().rename("paper_count").reset_index()
    if top_n:
        top_values = (
            counts.groupby(column)["paper_count"].sum().sort_values(ascending=False).head(top_n).index
        )
        counts = counts[counts[column].isin(top_values)]
    return counts


def build_primary_topic_trends(df: pd.DataFrame) -> pd.DataFrame:
    yearly_totals = df.groupby("year").size().rename("year_total").reset_index()
    primary_counts = (
        df.groupby(["year", "primary_topic"]).size().rename("paper_count").reset_index()
        .merge(yearly_totals, on="year", how="left")
    )
    primary_counts["share"] = primary_counts["paper_count"] / primary_counts["year_total"] * 100
    return primary_counts


def build_fine_topic_heatmap(df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    return _explode_counts(df, "fine_topics", top_n=top_n)


def build_benchmark_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["year", "venue"])  # aggregated counts
        .agg(papers=("title", "count"), benchmark_papers=("is_benchmark", "sum"))
        .reset_index()
    )
    summary["benchmark_share"] = summary["benchmark_papers"] / summary["papers"] * 100
    return summary


def summarise_trends(primary_trends: pd.DataFrame) -> list[TrendSummary]:
    summaries: list[TrendSummary] = []
    for topic, topic_df in primary_trends.groupby("primary_topic"):
        sorted_df = topic_df.sort_values("year")
        start = sorted_df.iloc[0]
        end = sorted_df.iloc[-1]
        peak_row = sorted_df.loc[sorted_df["share"].idxmax()]
        trough_row = sorted_df.loc[sorted_df["share"].idxmin()]
        summaries.append(
            TrendSummary(
                topic=topic,
                share_change=float(end["share"] - start["share"]),
                peak_year=int(peak_row["year"]),
                peak_share=float(peak_row["share"]),
                trough_year=int(trough_row["year"]),
                trough_share=float(trough_row["share"]),
            )
        )
    summaries.sort(key=lambda item: item.share_change, reverse=True)
    return summaries


def _contains_any(row_values: Sequence[str], filter_values: Sequence[str]) -> bool:
    return any(value in row_values for value in filter_values)


def filter_papers(
    df: pd.DataFrame,
    years: Sequence[int] | None = None,
    venues: Sequence[str] | None = None,
    primary_topics: Sequence[str] | None = None,
    fine_topics: Sequence[str] | None = None,
    topic_clusters: Sequence[str] | None = None,
    domain_tags: Sequence[str] | None = None,
    benchmark_tasks: Sequence[str] | None = None,
    benchmark_mode: str | None = None,
    text_query: str | None = None,
    author_query: str | None = None,
) -> pd.DataFrame:
    filtered = df
    if years:
        filtered = filtered[filtered["year"].isin(years)]
    if venues:
        filtered = filtered[filtered["venue"].isin(venues)]
    if primary_topics:
        filtered = filtered[filtered["primary_topic"].isin(primary_topics)]
    if fine_topics:
        filtered = filtered[filtered["fine_topics"].apply(lambda x: _contains_any(x, fine_topics))]
    if topic_clusters:
        filtered = filtered[filtered["topics"].apply(lambda x: _contains_any(x, topic_clusters))]
    if domain_tags:
        filtered = filtered[filtered["domain_tags"].apply(lambda x: _contains_any(x, domain_tags))]
    if benchmark_tasks:
        filtered = filtered[
            filtered["benchmark_tasks"].apply(lambda x: _contains_any(x, benchmark_tasks))
        ]
    if benchmark_mode == "bench_only":
        filtered = filtered[filtered["is_benchmark"]]
    elif benchmark_mode == "non_bench":
        filtered = filtered[~filtered["is_benchmark"]]
    if text_query:
        pattern = text_query.strip().lower()
        if pattern:
            filtered = filtered[
                filtered["title"].str.lower().str.contains(pattern, regex=False, na=False)
                | filtered["abstract"].str.lower().str.contains(pattern, regex=False, na=False)
                | filtered["topics"].apply(lambda values: any(pattern in v.lower() for v in values))
                | filtered["fine_topics"].apply(lambda values: any(pattern in v.lower() for v in values))
            ]
    if author_query:
        pattern = author_query.strip().lower()
        if pattern:
            filtered = filtered[
                filtered["authors"].str.lower().str.contains(pattern, regex=False, na=False)
            ]
    return filtered


def _empty_figure(title: str, subtitle: str) -> dict:
    fig = px.scatter(x=[], y=[])
    fig.update_layout(
        title=title,
        annotations=[
            dict(
                text=subtitle,
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14),
            )
        ],
    )
    return fig


def create_dashboard(df: pd.DataFrame) -> Dash:
    years = _year_order(df["year"].unique())
    venues = sorted(df["venue"].unique())
    primary_topics = sorted(df["primary_topic"].unique())
    fine_topics = sorted({fine for values in df["fine_topics"] for fine in values})
    topic_clusters = sorted({topic for values in df["topics"] for topic in values})
    domains = sorted({domain for values in df["domain_tags"] for domain in values})
    bench_tasks = sorted({task for values in df["benchmark_tasks"] for task in values})

    app = Dash(__name__)
    app.title = "EMNLP Topic Explorer"

    app.layout = html.Div(
        [
            html.H1("EMNLP Topic Explorer (2021-2024)"),
            html.P(
                "Interactively explore topic trends, benchmark activity, and individual papers. "
                "Use the filters below to focus on specific years, venues, or research themes."
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Year"),
                            dcc.Dropdown(
                                id="year-filter",
                                options=[{"label": str(year), "value": year} for year in years],
                                value=years,
                                multi=True,
                                placeholder="Select years",
                                persistence=True,
                                persistence_type="local",
                            ),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Label("Track"),
                            dcc.Dropdown(
                                id="venue-filter",
                                options=[{"label": venue, "value": venue} for venue in venues],
                                value=venues,
                                multi=True,
                                placeholder="Select track",
                                persistence=True,
                                persistence_type="local",
                            ),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Label("Primary topic"),
                            dcc.Dropdown(
                                id="primary-topic-filter",
                                options=[{"label": topic, "value": topic} for topic in primary_topics],
                                multi=True,
                                placeholder="Filter by primary topic",
                            ),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Label("Fine-grained topic"),
                            dcc.Dropdown(
                                id="fine-topic-filter",
                                options=[{"label": topic, "value": topic} for topic in fine_topics],
                                multi=True,
                                placeholder="Filter by fine-grained topic",
                            ),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Label("Topic cluster"),
                            dcc.Dropdown(
                                id="cluster-filter",
                                options=[{"label": topic, "value": topic} for topic in topic_clusters],
                                multi=True,
                                placeholder="Filter by topic cluster",
                            ),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Label("Domain"),
                            dcc.Dropdown(
                                id="domain-filter",
                                options=[{"label": domain, "value": domain} for domain in domains],
                                multi=True,
                                placeholder="Filter by domain tag",
                            ),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Label("Benchmark tasks"),
                            dcc.Dropdown(
                                id="benchmark-task-filter",
                                options=[{"label": task, "value": task} for task in bench_tasks],
                                multi=True,
                                placeholder="Filter by benchmark task",
                            ),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Label("Benchmark mode"),
                            dcc.RadioItems(
                                id="benchmark-mode",
                                options=[
                                    {"label": "All papers", "value": "all"},
                                    {"label": "Benchmarks only", "value": "bench_only"},
                                    {"label": "Non-benchmarks", "value": "non_bench"},
                                ],
                                value="all",
                                inline=True,
                            ),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Label("Primary trend metric"),
                            dcc.RadioItems(
                                id="trend-metric",
                                options=[
                                    {"label": "Share (%)", "value": "share"},
                                    {"label": "Paper count", "value": "paper_count"},
                                ],
                                value="share",
                                inline=True,
                            ),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Label("Fine topic slots"),
                            dcc.Slider(
                                id="fine-topic-limit",
                                min=FINE_TOPIC_MIN,
                                max=FINE_TOPIC_MAX,
                                step=5,
                                value=FINE_TOPIC_DEFAULT,
                                marks={
                                    FINE_TOPIC_MIN: str(FINE_TOPIC_MIN),
                                    FINE_TOPIC_DEFAULT: str(FINE_TOPIC_DEFAULT),
                                    FINE_TOPIC_MAX: str(FINE_TOPIC_MAX),
                                },
                                tooltip={"placement": "bottom", "always_visible": False},
                            ),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Label("Search titles / abstracts / topics"),
                            dcc.Input(
                                id="text-search",
                                type="text",
                                placeholder="Type a keyword (e.g. retrieval, hallucination)",
                                debounce=True,
                                style={"width": "100%"},
                            ),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Label("Search authors"),
                            dcc.Input(
                                id="author-search",
                                type="text",
                                placeholder="e.g. Smith, Brown",
                                debounce=True,
                                style={"width": "100%"},
                            ),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Label("Download filtered set"),
                            html.Button(
                                "Download CSV",
                                id="download-button",
                                n_clicks=0,
                                style={
                                    "width": "100%",
                                    "padding": "8px",
                                    "backgroundColor": "#1b263b",
                                    "color": "white",
                                    "border": "none",
                                    "borderRadius": "6px",
                                    "cursor": "pointer",
                                },
                            ),
                            dcc.Download(id="download-data"),
                        ],
                        style=FILTER_CARD_STYLE,
                    ),
                ],
                style=FILTERS_GRID_STYLE,
            ),
            html.Div(
                [
                    html.Div(id="summary-cards"),
                ]
            ),
            html.Div(
                [
                    dcc.Loading(dcc.Graph(id="primary-trends"), type="cube"),
                    dcc.Loading(dcc.Graph(id="fine-topic-heatmap"), type="cube"),
                    dcc.Loading(dcc.Graph(id="benchmark-view"), type="cube"),
                    dcc.Loading(dcc.Graph(id="domain-distribution"), type="cube"),
                ],
                style=GRAPH_GRID_STYLE,
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.H2("Quick insights"),
                            html.Div(id="insights-panel"),
                        ],
                        style=INSIGHT_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.H3("Top authors in view"),
                            dash_table.DataTable(
                                id="top-authors-table",
                                columns=[
                                    {"name": "Author", "id": "author"},
                                    {"name": "Papers", "id": "paper_count"},
                                    {"name": "Share (%)", "id": "share"},
                                ],
                                data=[],
                                sort_action="native",
                                style_table={"maxHeight": "320px", "overflowY": "auto"},
                                style_cell={"textAlign": "left", "padding": "6px"},
                            ),
                        ],
                        style=TABLE_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.H3("Top benchmark tasks"),
                            dash_table.DataTable(
                                id="top-benchmark-table",
                                columns=[
                                    {"name": "Benchmark task", "id": "task"},
                                    {"name": "Papers", "id": "paper_count"},
                                    {"name": "Share (%)", "id": "share"},
                                ],
                                data=[],
                                sort_action="native",
                                style_table={"maxHeight": "320px", "overflowY": "auto"},
                                style_cell={"textAlign": "left", "padding": "6px"},
                            ),
                        ],
                        style=TABLE_CARD_STYLE,
                    ),
                ],
                style={"display": "grid", "gap": "24px", "margin": "24px 0"},
            ),
            html.H2("Filtered papers"),
            dash_table.DataTable(
                id="papers-table",
                columns=[
                    {"name": "Title", "id": "title"},
                    {"name": "Authors", "id": "authors"},
                    {"name": "Year", "id": "year"},
                    {"name": "Track", "id": "venue"},
                    {"name": "Primary topic", "id": "primary_topic"},
                    {"name": "Fine topics", "id": "fine_topics"},
                    {"name": "Benchmark?", "id": "is_benchmark"},
                    {"name": "Benchmark name", "id": "benchmark_name"},
                    {"name": "Benchmark tasks", "id": "benchmark_tasks"},
                    {"name": "URL", "id": "url", "presentation": "markdown"},
                ],
                data=[],
                page_current=0,
                page_size=20,
                sort_action="native",
                filter_action="native",
                style_table={"overflowX": "auto", "maxHeight": "540px", "overflowY": "auto"},
                style_cell={"textAlign": "left", "minWidth": "120px", "whiteSpace": "normal", "height": "auto"},
                style_header={"backgroundColor": "#e0e6ef", "fontWeight": "600"},
                markdown_options={"link_target": "_blank"},
            ),
            html.Footer(
                [
                    html.P(
                        "This dashboard runs locally with Plotly Dash. "
                        "Export aggregated CSVs with `python analysis/emnlp_topics_dashboard.py --export`."
                    ),
                ],
                style={"marginTop": "40px"},
            ),
        ],
        style=PAGE_STYLE,
    )

    @app.callback(
        Output("primary-trends", "figure"),
        Output("fine-topic-heatmap", "figure"),
        Output("benchmark-view", "figure"),
        Output("domain-distribution", "figure"),
        Output("papers-table", "data"),
        Output("summary-cards", "children"),
        Output("insights-panel", "children"),
        Output("top-authors-table", "data"),
        Output("top-benchmark-table", "data"),
        Input("year-filter", "value"),
        Input("venue-filter", "value"),
        Input("primary-topic-filter", "value"),
        Input("fine-topic-filter", "value"),
        Input("cluster-filter", "value"),
        Input("domain-filter", "value"),
        Input("benchmark-task-filter", "value"),
        Input("benchmark-mode", "value"),
        Input("trend-metric", "value"),
        Input("fine-topic-limit", "value"),
        Input("text-search", "value"),
        Input("author-search", "value"),
    )
    def update_dashboard(
        selected_years,
        selected_venues,
        selected_primary,
        selected_fine,
        selected_clusters,
        selected_domains,
        selected_tasks,
        benchmark_mode,
        trend_metric,
        fine_topic_limit,
        text_query,
        author_query,
    ):
        years_value = selected_years if selected_years else years
        venues_value = selected_venues if selected_venues else venues
        fine_topic_limit = int(fine_topic_limit or FINE_TOPIC_DEFAULT)
        filtered = filter_papers(
            df,
            years=years_value,
            venues=venues_value,
            primary_topics=selected_primary,
            fine_topics=selected_fine,
            topic_clusters=selected_clusters,
            domain_tags=selected_domains,
            benchmark_tasks=selected_tasks,
            benchmark_mode=None if benchmark_mode == "all" else benchmark_mode,
            text_query=text_query,
            author_query=author_query,
        )

        if filtered.empty:
            empty_fig = _empty_figure("No data", "No papers match the selected filters.")
            return (
                empty_fig,
                empty_fig,
                empty_fig,
                empty_fig,
                [],
                _render_summary_cards(0, 0, 0, 0.0),
                html.P("Adjust the filters to see insights."),
                [],
                [],
            )

        primary_trends = build_primary_topic_trends(filtered)
        y_axis = "share" if trend_metric == "share" else "paper_count"
        y_label = "Share of papers (%)" if y_axis == "share" else "Paper count"
        trend_fig = px.line(
            primary_trends,
            x="year",
            y=y_axis,
            color="primary_topic",
            markers=True,
            category_orders={"year": _year_order(primary_trends["year"])},
            labels={
                "share": "Share of papers (%)",
                "paper_count": "Paper count",
                "year": "Year",
                "primary_topic": "Primary topic",
            },
            title="Primary topic share across selected filters",
            hover_data={"paper_count": True, "share": ":.2f"},
        )
        trend_fig.update_layout(yaxis_title=y_label)

        fine_trends = build_fine_topic_heatmap(filtered, top_n=fine_topic_limit)
        if fine_trends.empty:
            fine_fig = _empty_figure("Fine topics", "No fine-grained topics in the filtered set.")
        else:
            heatmap = (
                fine_trends.pivot(index="fine_topics", columns="year", values="paper_count").fillna(0)
            )
            fine_fig = px.imshow(
                heatmap,
                labels=dict(x="Year", y="Fine-grained topic", color="# Papers"),
                title="Fine-grained topic intensity",
                aspect="auto",
            )
            fine_fig.update_yaxes(categoryorder="array", categoryarray=list(heatmap.index))

        benchmark_summary = build_benchmark_summary(filtered)
        bench_fig = px.bar(
            benchmark_summary,
            x="year",
            y="benchmark_papers",
            color="venue",
            barmode="group",
            category_orders={"year": _year_order(benchmark_summary["year"])},
            labels={"benchmark_papers": "Benchmark papers", "year": "Year", "venue": "Track"},
            title="Benchmark papers for the filtered set",
            text="benchmark_share",
        )
        bench_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        bench_fig.update_layout(uniformtext_minsize=8, uniformtext_mode="hide")

        domain_counts = build_domain_distribution(filtered)
        if domain_counts.empty:
            domain_fig = _empty_figure("Domain coverage", "No domain tags annotated.")
        else:
            domain_subset = domain_counts.head(25)
            domain_fig = px.bar(
                domain_subset,
                x="paper_count",
                y="domain_tags",
                orientation="h",
                labels={"paper_count": "Papers", "domain_tags": "Domain"},
                title="Domains represented in the filtered set",
                text="share",
            )
            domain_fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            domain_fig.update_layout(yaxis=dict(autorange="reversed"))

        table_data = (
            filtered.sort_values(["year", "primary_topic", "title"], ascending=[False, True, True])
            .assign(
                fine_topics=lambda df_: df_["fine_topics"].apply(lambda values: ", ".join(values)),
                benchmark_tasks=lambda df_: df_["benchmark_tasks"].apply(lambda values: ", ".join(values)),
                url=lambda df_: df_["url"].fillna("").apply(lambda x: f"[Link]({x})" if x else ""),
                is_benchmark=lambda df_: df_["is_benchmark"].map({True: "Yes", False: "No"}),
            )
            [[
                "title",
                "authors",
                "year",
                "venue",
                "primary_topic",
                "fine_topics",
                "is_benchmark",
                "benchmark_name",
                "benchmark_tasks",
                "url",
            ]]
            .to_dict("records")
        )

        bench_count = int(filtered["is_benchmark"].sum())
        total = len(filtered)
        coverage = bench_count / total * 100 if total else 0.0
        topic_count = filtered["primary_topic"].nunique()

        summary_cards = _render_summary_cards(total, topic_count, bench_count, coverage)
        summaries = summarise_trends(primary_trends)
        risers = [item for item in summaries if item.share_change > 0][:3]
        decliners = [item for item in sorted(summaries, key=lambda s: s.share_change) if item.share_change < 0][
            :3
        ]
        insight_children: list = []
        if risers:
            insight_children.append(html.H4("Fastest-growing primary topics"))
            insight_children.append(
                html.Ul(
                    [
                        html.Li(
                            f"{item.topic}: {item.share_change:+.2f}pp since {primary_trends['year'].min()} (peak {item.peak_share:.2f}% in {item.peak_year})"
                        )
                        for item in risers
                    ]
                )
            )
        if decliners:
            insight_children.append(html.H4("Topics losing share"))
            insight_children.append(
                html.Ul(
                    [
                        html.Li(
                            f"{item.topic}: {item.share_change:+.2f}pp since {primary_trends['year'].min()} (trough {item.trough_share:.2f}% in {item.trough_year})"
                        )
                        for item in decliners
                    ]
                )
            )
        if not insight_children:
            insight_children = [html.P("No share shifts detected for the selected filters.")]

        top_authors = build_top_authors(filtered).assign(
            share=lambda df_: df_["share"].map(lambda value: f"{value:.2f}%")
        )
        top_benchmarks = (
            _top_list_counts(filtered, "benchmark_tasks")
            .rename(columns={"benchmark_tasks": "task"})
            .assign(share=lambda df_: df_["share"].map(lambda value: f"{value:.2f}%"))
        )

        return (
            trend_fig,
            fine_fig,
            bench_fig,
            domain_fig,
            table_data,
            summary_cards,
            insight_children,
            top_authors.to_dict("records"),
            top_benchmarks.to_dict("records"),
        )

    @app.callback(
        Output("download-data", "data"),
        Input("download-button", "n_clicks"),
        State("year-filter", "value"),
        State("venue-filter", "value"),
        State("primary-topic-filter", "value"),
        State("fine-topic-filter", "value"),
        State("cluster-filter", "value"),
        State("domain-filter", "value"),
        State("benchmark-task-filter", "value"),
        State("benchmark-mode", "value"),
        State("text-search", "value"),
        State("author-search", "value"),
        prevent_initial_call=True,
    )
    def download_filtered(
        n_clicks,
        selected_years,
        selected_venues,
        selected_primary,
        selected_fine,
        selected_clusters,
        selected_domains,
        selected_tasks,
        benchmark_mode,
        text_query,
        author_query,
    ):
        if not n_clicks:
            raise PreventUpdate

        years_value = selected_years if selected_years else years
        venues_value = selected_venues if selected_venues else venues
        filtered = filter_papers(
            df,
            years=years_value,
            venues=venues_value,
            primary_topics=selected_primary,
            fine_topics=selected_fine,
            topic_clusters=selected_clusters,
            domain_tags=selected_domains,
            benchmark_tasks=selected_tasks,
            benchmark_mode=None if benchmark_mode == "all" else benchmark_mode,
            text_query=text_query,
            author_query=author_query,
        )

        if filtered.empty:
            raise PreventUpdate

        export_df = filtered.copy()
        for column in ["topics", "fine_topics", "domain_tags", "benchmark_tasks", "tags"]:
            if column in export_df:
                export_df[column] = export_df[column].apply(lambda values: "; ".join(values))
        export_df["is_benchmark"] = (
            export_df["is_benchmark"].map({True: "Yes", False: "No"}).fillna("Unknown")
        )

        csv_buffer = io.StringIO()
        export_df[
            [
                "title",
                "authors",
                "year",
                "venue",
                "primary_topic",
                "topics",
                "fine_topics",
                "domain_tags",
                "is_benchmark",
                "benchmark_name",
                "benchmark_tasks",
                "url",
            ]
        ].to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return dict(content=csv_buffer.getvalue(), filename="emnlp_filtered_papers.csv")

    return app


def _render_summary_cards(total: int, topic_count: int, benchmark_count: int, benchmark_share: float):
    text_style = {"margin": "4px 0 0", "fontSize": "14px", "opacity": 0.85}
    value_style = {"margin": "0", "fontSize": "28px"}
    return html.Div(
        [
            html.Div(
                [
                    html.H3(f"{total:,}", style=value_style),
                    html.P("Papers in view", style=text_style),
                ],
                style=SUMMARY_CARD_STYLE,
            ),
            html.Div(
                [
                    html.H3(str(topic_count), style=value_style),
                    html.P("Primary topics represented", style=text_style),
                ],
                style=SUMMARY_CARD_STYLE,
            ),
            html.Div(
                [
                    html.H3(f"{benchmark_count:,}", style=value_style),
                    html.P("Benchmark papers", style=text_style),
                ],
                style=SUMMARY_CARD_STYLE,
            ),
            html.Div(
                [
                    html.H3(f"{benchmark_share:.1f}%", style=value_style),
                    html.P("Benchmark share", style=text_style),
                ],
                style=SUMMARY_CARD_STYLE,
            ),
        ],
        style=SUMMARY_GRID_STYLE,
    )


def export_reports(df: pd.DataFrame) -> None:
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    primary_trends = build_primary_topic_trends(df)
    fine_trends = build_fine_topic_heatmap(df, top_n=None)
    benchmark_summary = build_benchmark_summary(df)
    domain_summary = build_domain_distribution(df)
    author_summary = build_top_authors(df, limit=None)

    primary_trends.to_csv(SUMMARY_CSV, index=False)
    fine_trends.to_csv(FINE_TOPIC_CSV, index=False)
    benchmark_summary.to_csv(BENCHMARK_CSV, index=False)
    domain_summary.to_csv(DOMAIN_CSV, index=False)
    author_summary.to_csv(AUTHOR_CSV, index=False)

    yearly = (
        benchmark_summary.groupby("year")
        .agg(papers=("papers", "sum"), benchmark_papers=("benchmark_papers", "sum"))
        .reset_index()
    )
    yearly["benchmark_share"] = yearly["benchmark_papers"] / yearly["papers"] * 100
    yearly.to_csv(BENCHMARK_YEARLY_CSV, index=False)

    summaries = summarise_trends(primary_trends)
    console_lines = [
        "Exported summary tables:",
        f" - Primary topic trends → {SUMMARY_CSV}",
        f" - Fine-grained topic counts → {FINE_TOPIC_CSV}",
        f" - Benchmark summary → {BENCHMARK_CSV}",
        f" - Benchmark yearly summary → {BENCHMARK_YEARLY_CSV}",
        f" - Domain summary → {DOMAIN_CSV}",
        f" - Author frequency summary → {AUTHOR_CSV}",
        "Top share gains:",
    ]
    for summary in summaries[:10]:
        console_lines.append(
            f"   • {summary.topic}: {summary.share_change:+.2f}pp (peak {summary.peak_share:.2f}%/{summary.peak_year})"
        )
    print("\n".join(console_lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the EMNLP topic explorer dashboard or export summary tables.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
              Launch the dashboard (default on http://127.0.0.1:8050)
                python analysis/emnlp_topics_dashboard.py

              Export CSV summaries only
                python analysis/emnlp_topics_dashboard.py --export
            """
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface for the Dash app (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8050,
        help="Port for the Dash app (default: 8050)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Dash debug mode for development.",
    )
    parser.add_argument(
        "--export",
        action="store_true",
        help="Generate CSV summaries and exit without starting the app.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = _load_dataset()

    if args.export:
        export_reports(df)
        return

    app = create_dashboard(df)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
