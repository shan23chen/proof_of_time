# proof_of_time

project plan: https://docs.google.com/document/d/13FkKrJ3X19Sr3rgxFWRGWlSjj1Fm_eJZYZMO4hXrz7I/edit?usp=sharing

data: https://huggingface.co/AIM-Harvard

## EMNLP Topic Explorer Dashboard

The interactive EMNLP explorer lives in `analysis/emnlp_topics_dashboard.py`. It supports
filtering papers by year, track (Main vs. Findings), primary and fine-grained topics,
topic clusters, domain tags, benchmark tasks, keyword/author search, and CSV download of
the currently filtered set. Toggle the primary-topic trend metric (share vs. raw counts),
choose how many fine topics appear in the heatmap, explore domain distributions, and scan
auto-generated insight callouts plus top-author/benchmark tables for quick orientation.

Run it locally after installing the dependencies (`pip install datasets dash plotly pandas`):

```bash
python analysis/emnlp_topics_dashboard.py
```

By default the Dash app will be available at http://127.0.0.1:8050. Use the `--host` and
`--port` flags to customise the binding. Pass `--export` to generate refreshed CSV
summaries in `reports/` without launching the web UI.

The export mode now writes:

* `emnlp_topics_summary.csv` — primary-topic counts, shares, and totals per year.
* `emnlp_finetopic_summary.csv` — exploded fine-grained topic counts per year.
* `emnlp_benchmark_summary.csv` — benchmark vs. non-benchmark paper totals by track.
* `emnlp_benchmark_yearly.csv` — annual benchmark totals and shares across tracks.
* `emnlp_domain_summary.csv` — aggregated domain-tag activity across the corpus.
* `emnlp_author_summary.csv` — frequency table of author appearances (deduplicated).
