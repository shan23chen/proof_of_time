#!/usr/bin/env python3
"""
Analyze agent queries, token usage, and cost from Inspect AI evaluation logs.

This script reads from summary CSV files (logs_msg15_summary.csv, logs_msg30_summary.csv,
logs_msg50_summary.csv) to get eval file paths and their corresponding message limits.

It analyzes the correlation between:
- Number of agent tool calls (queries)
- Token usage (input/output)
- Estimated cost

Usage:
    python scripts/analyze_agent_queries.py --summary-files logs_msg15_summary.csv logs_msg30_summary.csv logs_msg50_summary.csv
    python scripts/analyze_agent_queries.py --output results.csv
"""

import argparse
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import pandas as pd

from inspect_ai.log import read_eval_log


@dataclass
class QueryStats:
    """Statistics for a single evaluation run."""
    log_file: str
    task: str
    model: str
    message_limit: Optional[int]
    status: str
    num_samples: int
    total_messages: int
    total_tool_calls: int
    avg_messages_per_sample: float
    avg_tool_calls_per_sample: float
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    estimated_cost_usd: float
    duration_seconds: float
    accuracy: float


# Model pricing (per million tokens) - approximate rates
MODEL_PRICING = {
    # OpenAI GPT-5 series (estimated)
    'openai/gpt-5.1': {'input': 2.50, 'output': 10.00, 'cache_read': 0.625, 'cache_write': 5.00},
    'openai/gpt-5-mini': {'input': 0.30, 'output': 1.20, 'cache_read': 0.075, 'cache_write': 0.60},
    'openai/gpt-5-nano': {'input': 0.10, 'output': 0.40, 'cache_read': 0.025, 'cache_write': 0.20},

    # Groq models (free tier, but showing typical costs)
    'groq/llama': {'input': 0.05, 'output': 0.08, 'cache_read': 0.0, 'cache_write': 0.0},
    'groq/kimi': {'input': 0.10, 'output': 0.15, 'cache_read': 0.0, 'cache_write': 0.0},
    'groq/qwen': {'input': 0.05, 'output': 0.08, 'cache_read': 0.0, 'cache_write': 0.0},

    # Google Gemini (estimated)
    'google/gemini-3': {'input': 0.125, 'output': 0.50, 'cache_read': 0.03125, 'cache_write': 0.25},
    'google/gemini-2.5': {'input': 0.625, 'output': 2.50, 'cache_read': 0.15625, 'cache_write': 1.25},

    # Anthropic Claude via Vertex AI (estimated)
    'anthropic/claude-haiku-4-5': {'input': 0.80, 'output': 4.00, 'cache_read': 0.20, 'cache_write': 1.60},
    'anthropic/claude-sonnet-4-5': {'input': 3.00, 'output': 15.00, 'cache_read': 0.75, 'cache_write': 6.00},
    'anthropic/claude-opus-4-5': {'input': 15.00, 'output': 75.00, 'cache_read': 3.75, 'cache_write': 30.00},
}


def get_model_pricing(model: str) -> Dict[str, float]:
    """Get pricing for a model (approximate match)."""
    for key, pricing in MODEL_PRICING.items():
        if key in model.lower():
            return pricing
    # Default pricing if not found
    return {'input': 1.0, 'output': 3.0, 'cache_read': 0.25, 'cache_write': 2.0}


def calculate_cost(model: str, input_tokens: int, output_tokens: int,
                   cache_read: int = 0, cache_write: int = 0) -> float:
    """Calculate estimated cost in USD."""
    pricing = get_model_pricing(model)

    cost = (
        (input_tokens / 1_000_000) * pricing['input'] +
        (output_tokens / 1_000_000) * pricing['output'] +
        (cache_read / 1_000_000) * pricing['cache_read'] +
        (cache_write / 1_000_000) * pricing['cache_write']
    )

    return cost


def count_tool_calls(messages: List[Any]) -> int:
    """Count tool calls in a message list."""
    tool_calls = 0
    for msg in messages:
        # Check if message has tool calls
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            tool_calls += len(msg.tool_calls)
        # Alternative: check content for tool usage patterns
        elif hasattr(msg, 'content'):
            content_str = str(msg.content)
            # Count tool invocations (bash, python, etc.)
            if 'tool_call' in content_str.lower():
                tool_calls += 1

    return tool_calls


def extract_message_limit(log_path: str) -> Optional[int]:
    """Extract message limit from log directory path."""
    path = str(log_path).lower()
    if 'msg15' in path or 'message_limit_15' in path:
        return 15
    elif 'msg30' in path or 'message_limit_30' in path:
        return 30
    elif 'msg50' in path or 'message_limit_50' in path:
        return 50
    return None


def analyze_eval_log(log_path: Path, message_limit: Optional[int] = None, accuracy: float = 0.0) -> Optional[QueryStats]:
    """
    Analyze a single .eval log file.

    Args:
        log_path: Path to the .eval file
        message_limit: Message limit for this eval (from summary CSV)
        accuracy: Accuracy score (from summary CSV)
    """
    try:
        log = read_eval_log(str(log_path))

        # Extract basic info
        task = log.eval.task
        model = log.eval.model
        status = log.status
        num_samples = len(log.samples)

        # Count messages and tool calls
        total_messages = 0
        total_tool_calls = 0

        for sample in log.samples:
            total_messages += len(sample.messages)
            total_tool_calls += count_tool_calls(sample.messages)

        avg_messages = total_messages / num_samples if num_samples > 0 else 0
        avg_tool_calls = total_tool_calls / num_samples if num_samples > 0 else 0

        # Extract token usage
        model_usage = log.stats.model_usage.get(model) if log.stats.model_usage else None
        input_tokens = model_usage.input_tokens if model_usage else 0
        output_tokens = model_usage.output_tokens if model_usage else 0
        total_tokens = model_usage.total_tokens if model_usage else 0
        cache_read = model_usage.input_tokens_cache_read if model_usage and model_usage.input_tokens_cache_read else 0
        cache_write = model_usage.input_tokens_cache_write if model_usage and model_usage.input_tokens_cache_write else 0

        # Calculate cost
        estimated_cost = calculate_cost(model, input_tokens, output_tokens, cache_read, cache_write)

        # Calculate duration
        if log.stats.started_at and log.stats.completed_at:
            from datetime import datetime
            start = datetime.fromisoformat(log.stats.started_at.replace('Z', '+00:00'))
            end = datetime.fromisoformat(log.stats.completed_at.replace('Z', '+00:00'))
            duration = (end - start).total_seconds()
        else:
            duration = 0.0

        # Use accuracy passed from summary CSV (already extracted)

        return QueryStats(
            log_file=log_path.name,
            task=task,
            model=model,
            message_limit=message_limit,
            status=status,
            num_samples=num_samples,
            total_messages=total_messages,
            total_tool_calls=total_tool_calls,
            avg_messages_per_sample=avg_messages,
            avg_tool_calls_per_sample=avg_tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cache_read_tokens=cache_read,
            cache_write_tokens=cache_write,
            estimated_cost_usd=estimated_cost,
            duration_seconds=duration,
            accuracy=accuracy
        )

    except Exception as e:
        print(f"Error processing {log_path}: {e}")
        return None


def load_summary_csvs(summary_files: List[str]) -> Dict[str, tuple]:
    """
    Load summary CSV files and extract eval file paths with message limits and accuracy.

    Args:
        summary_files: List of paths to summary CSV files (e.g., logs_msg15_summary.csv)

    Returns:
        Dictionary mapping log_path to (task, model, message_limit, accuracy) tuple
    """
    eval_mapping = {}

    for summary_file in summary_files:
        # Extract message limit from filename
        filename = Path(summary_file).stem  # e.g., "logs_msg15_summary"
        message_limit = None
        if 'msg15' in filename:
            message_limit = 15
        elif 'msg30' in filename:
            message_limit = 30
        elif 'msg50' in filename:
            message_limit = 50

        # Read CSV
        try:
            df = pd.read_csv(summary_file)
            print(f"Loaded {len(df)} entries from {summary_file} (message_limit={message_limit})")

            for _, row in df.iterrows():
                log_path = row['log_path']
                task = row['task_name']
                model = row['model']
                accuracy = row.get('accuracy', 0.0)  # Extract accuracy from CSV

                # Store in mapping
                eval_mapping[log_path] = (task, model, message_limit, accuracy)

        except Exception as e:
            print(f"Error loading {summary_file}: {e}")

    return eval_mapping


def find_eval_files_from_mapping(eval_mapping: Dict[str, tuple]) -> List[tuple]:
    """
    Find eval files from the mapping.

    Args:
        eval_mapping: Dictionary from load_summary_csvs

    Returns:
        List of (Path, task, model, message_limit, accuracy) tuples for existing files
    """
    eval_files = []

    for log_path, (task, model, message_limit, accuracy) in eval_mapping.items():
        path = Path(log_path)
        if path.exists():
            eval_files.append((path, task, model, message_limit, accuracy))
        else:
            print(f"Warning: File not found: {log_path}")

    return eval_files


def write_results_csv(stats: List[QueryStats], output_path: str):
    """Write results to CSV file."""
    if not stats:
        print("No statistics to write")
        return

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(stats[0]).keys()))
        writer.writeheader()

        for stat in stats:
            writer.writerow(asdict(stat))

    print(f"\n✓ Results written to: {output_path}")


def print_summary(stats: List[QueryStats]):
    """Print summary statistics."""
    if not stats:
        print("No statistics to summarize")
        return

    print("\n" + "=" * 80)
    print("AGENT QUERY ANALYSIS SUMMARY")
    print("=" * 80)

    # Group by message limit
    by_msg_limit = defaultdict(list)
    for stat in stats:
        by_msg_limit[stat.message_limit].append(stat)

    for msg_limit in sorted(by_msg_limit.keys(), key=lambda x: x if x is not None else 0):
        limit_stats = by_msg_limit[msg_limit]

        print(f"\n{'─' * 80}")
        print(f"MESSAGE LIMIT: {msg_limit if msg_limit else 'Unknown'}")
        print(f"{'─' * 80}")

        total_cost = sum(s.estimated_cost_usd for s in limit_stats)
        total_tokens = sum(s.total_tokens for s in limit_stats)
        total_tool_calls = sum(s.total_tool_calls for s in limit_stats)
        avg_tool_calls = sum(s.avg_tool_calls_per_sample for s in limit_stats) / len(limit_stats)

        print(f"  Eval runs: {len(limit_stats)}")
        print(f"  Total tool calls: {total_tool_calls:,}")
        print(f"  Avg tool calls per sample: {avg_tool_calls:.2f}")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Estimated total cost: ${total_cost:.4f}")

        # Show top 5 most expensive runs
        print(f"\n  Top 5 most expensive runs:")
        top_5 = sorted(limit_stats, key=lambda x: x.estimated_cost_usd, reverse=True)[:5]
        for i, stat in enumerate(top_5, 1):
            print(f"    {i}. {stat.model:40s} | {stat.task:30s} | ${stat.estimated_cost_usd:.4f} | {stat.total_tool_calls} tool calls")


def main():
    parser = argparse.ArgumentParser(description='Analyze agent queries and costs from eval logs')
    parser.add_argument('--summary-files', nargs='+',
                       default=['logs_msg15_summary.csv', 'logs_msg30_summary.csv', 'logs_msg50_summary.csv'],
                       help='Summary CSV files containing eval paths and message limits')
    parser.add_argument('--output', default='analysis/query_analysis/outputs/agent_query_analysis_msg_limits.csv',
                       help='Output CSV file path')

    args = parser.parse_args()

    # Ensure output directory exists
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing output CSV to get accuracy values
    accuracy_lookup = {}
    if output_path.exists():
        print(f"Loading existing accuracy values from {args.output}...")
        try:
            existing_df = pd.read_csv(args.output)
            for _, row in existing_df.iterrows():
                log_file = row['log_file']
                acc = row.get('accuracy', 0.0)
                accuracy_lookup[log_file] = acc
            print(f"  Loaded {len(accuracy_lookup)} accuracy values")
        except Exception as e:
            print(f"  Warning: Could not load existing CSV: {e}")

    # Load summary CSVs
    print("\nLoading summary CSV files...")
    eval_mapping = load_summary_csvs(args.summary_files)

    if not eval_mapping:
        print("No eval files found in summary CSVs!")
        return

    # Find eval files
    print(f"\nFound {len(eval_mapping)} entries in summary files")
    eval_files = find_eval_files_from_mapping(eval_mapping)
    print(f"Found {len(eval_files)} existing .eval files\n")

    # Analyze each file
    all_stats = []
    for i, (eval_file, task, model, message_limit, accuracy) in enumerate(eval_files, 1):
        if i % 10 == 0:
            print(f"  Processed {i}/{len(eval_files)} files...")

        # Use accuracy from existing CSV if available, otherwise use from summary
        log_filename = eval_file.name
        if log_filename in accuracy_lookup:
            accuracy = accuracy_lookup[log_filename]

        stats = analyze_eval_log(eval_file, message_limit=message_limit, accuracy=accuracy)
        if stats:
            all_stats.append(stats)

    print(f"\n✓ Successfully analyzed {len(all_stats)}/{len(eval_files)} files")

    # Write results
    write_results_csv(all_stats, args.output)

    # Print summary
    print_summary(all_stats)

    print("\n" + "=" * 80)
    print(f"Full results saved to: {args.output}")
    print("=" * 80)


if __name__ == "__main__":
    main()
