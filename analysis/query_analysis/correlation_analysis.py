#!/usr/bin/env python3
"""
Analyze correlations between agent queries, token usage, and cost.

This script reads the agent_query_analysis.csv file and computes
correlation statistics.

Usage:
    python scripts/correlation_analysis.py --input analysis/agent_query_analysis.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def compute_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for key metrics."""
    metrics = [
        'total_tool_calls',
        'avg_tool_calls_per_sample',
        'total_messages',
        'avg_messages_per_sample',
        'input_tokens',
        'output_tokens',
        'total_tokens',
        'estimated_cost_usd',
        'duration_seconds'
    ]

    # Filter to only successful runs
    df_success = df[df['status'] == 'success']

    return df_success[metrics].corr()


def print_summary_stats(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    # Filter successful runs
    df_success = df[df['status'] == 'success']

    print(f"\nTotal eval runs: {len(df)}")
    print(f"  Successful: {len(df_success)}")
    print(f"  Failed/Error: {len(df[df['status'].isin(['error', 'cancelled'])])}")

    print(f"\nAgent Query Statistics:")
    print(f"  Total tool calls: {df_success['total_tool_calls'].sum():,}")
    print(f"  Avg tool calls per run: {df_success['total_tool_calls'].mean():.2f}")
    print(f"  Avg tool calls per sample: {df_success['avg_tool_calls_per_sample'].mean():.2f}")
    print(f"  Max tool calls in a run: {df_success['total_tool_calls'].max()}")

    print(f"\nToken Usage:")
    print(f"  Total tokens: {df_success['total_tokens'].sum():,}")
    print(f"  Avg tokens per run: {df_success['total_tokens'].mean():,.0f}")
    print(f"  Avg input tokens: {df_success['input_tokens'].mean():,.0f}")
    print(f"  Avg output tokens: {df_success['output_tokens'].mean():,.0f}")

    print(f"\nCost:")
    print(f"  Total estimated cost: ${df_success['estimated_cost_usd'].sum():.4f}")
    print(f"  Avg cost per run: ${df_success['estimated_cost_usd'].mean():.4f}")
    print(f"  Min cost: ${df_success['estimated_cost_usd'].min():.4f}")
    print(f"  Max cost: ${df_success['estimated_cost_usd'].max():.4f}")


def print_correlation_analysis(df: pd.DataFrame):
    """Print correlation analysis."""
    print("\n" + "=" * 80)
    print("CORRELATION ANALYSIS (Successful runs only)")
    print("=" * 80)

    corr_matrix = compute_correlations(df)

    print("\n📊 Correlation: Tool Calls vs Other Metrics")
    print("─" * 80)
    tool_call_corr = corr_matrix['total_tool_calls'].sort_values(ascending=False)
    for metric, corr in tool_call_corr.items():
        if metric != 'total_tool_calls':
            strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
            print(f"  {metric:35s}: {corr:6.3f} ({strength})")

    print("\n💰 Correlation: Cost vs Other Metrics")
    print("─" * 80)
    cost_corr = corr_matrix['estimated_cost_usd'].sort_values(ascending=False)
    for metric, corr in cost_corr.items():
        if metric != 'estimated_cost_usd':
            strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
            print(f"  {metric:35s}: {corr:6.3f} ({strength})")


def analyze_by_task(df: pd.DataFrame):
    """Analyze metrics by task."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY TASK")
    print("=" * 80)

    df_success = df[df['status'] == 'success']

    by_task = df_success.groupby('task').agg({
        'total_tool_calls': ['count', 'mean', 'sum'],
        'avg_tool_calls_per_sample': 'mean',
        'total_tokens': ['mean', 'sum'],
        'estimated_cost_usd': ['mean', 'sum'],
        'duration_seconds': 'mean'
    }).round(2)

    print("\nTop 10 tasks by total tool calls:")
    top_tasks = by_task.sort_values(('total_tool_calls', 'sum'), ascending=False).head(10)
    print(top_tasks.to_string())


def analyze_by_model(df: pd.DataFrame):
    """Analyze metrics by model."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY MODEL")
    print("=" * 80)

    df_success = df[df['status'] == 'success']

    by_model = df_success.groupby('model').agg({
        'total_tool_calls': ['count', 'mean', 'sum'],
        'avg_tool_calls_per_sample': 'mean',
        'total_tokens': ['mean', 'sum'],
        'estimated_cost_usd': ['mean', 'sum'],
        'duration_seconds': 'mean'
    }).round(2)

    print("\nAll models:")
    print(by_model.sort_values(('estimated_cost_usd', 'sum'), ascending=False).to_string())


def analyze_by_message_limit(df: pd.DataFrame):
    """Analyze metrics by message limit."""
    print("\n" + "=" * 80)
    print("ANALYSIS BY MESSAGE LIMIT")
    print("=" * 80)

    df_success = df[df['status'] == 'success']

    # Filter out None message limits
    df_with_limits = df_success[df_success['message_limit'].notna()]

    if len(df_with_limits) == 0:
        print("\nNo runs with message_limit information found.")
        print("Message limits should be extracted from log directory paths.")
        return

    by_limit = df_with_limits.groupby('message_limit').agg({
        'total_tool_calls': ['count', 'mean', 'sum'],
        'avg_tool_calls_per_sample': 'mean',
        'total_tokens': ['mean', 'sum'],
        'estimated_cost_usd': ['mean', 'sum'],
        'duration_seconds': 'mean'
    }).round(2)

    print("\nBy message limit:")
    print(by_limit.to_string())


def main():
    parser = argparse.ArgumentParser(description='Analyze correlations from agent query data')
    parser.add_argument('--input', default='analysis/query_analysis/outputs/agent_query_analysis_msg_limits.csv',
                       help='Input CSV file path')

    args = parser.parse_args()

    # Read CSV
    df = pd.read_csv(args.input)

    print(f"Loaded {len(df)} eval runs from {args.input}")

    # Print analyses
    print_summary_stats(df)
    print_correlation_analysis(df)
    analyze_by_task(df)
    analyze_by_model(df)
    analyze_by_message_limit(df)

    # Save correlation matrix
    corr_matrix = compute_correlations(df)
    corr_output = Path(args.input).parent / 'correlation_matrix.csv'
    corr_matrix.to_csv(corr_output)
    print(f"\n✓ Correlation matrix saved to: {corr_output}")


if __name__ == "__main__":
    main()
