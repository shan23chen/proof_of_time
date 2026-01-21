#!/usr/bin/env python3
"""
Parse eval files from logs_all directory based on log IDs found in ablation directories.

This script:
1. Scans log files in logs_exp_msg15/ and logs_exp_msg50/ to extract eval IDs
2. Finds corresponding .eval files in logs_all/
3. Parses them and generates two separate CSV files (one for each message limit)
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set

# Import from existing parse script
sys.path.insert(0, str(Path(__file__).parent))
from parse_logs_to_csv import LogData, parse_eval_file


def extract_eval_id_from_log(log_path: Path) -> Optional[str]:
    """Extract eval ID from a .log file by finding the 'Log: logs/...' line."""
    try:
        with log_path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Look for pattern: Log: logs/YYYY-MM-DD...TASK_ID.eval
                match = re.search(r'Log:\s+logs/[^/]+_([A-Za-z0-9]+)\.eval', line)
                if match:
                    return match.group(1)
    except Exception as e:
        print(f"Error reading {log_path}: {e}", file=sys.stderr)
    return None


def collect_eval_ids_from_logs(log_dir: Path) -> Set[str]:
    """Scan all .log files in directory tree and extract eval IDs."""
    eval_ids: Set[str] = set()

    for log_file in log_dir.rglob("*.log"):
        eval_id = extract_eval_id_from_log(log_file)
        if eval_id:
            eval_ids.add(eval_id)

    return eval_ids


def find_eval_files_by_ids(eval_dir: Path, eval_ids: Set[str]) -> List[Path]:
    """Find .eval files in eval_dir that match the given eval IDs."""
    matched_files: List[Path] = []

    for eval_file in eval_dir.glob("*.eval"):
        # Extract ID from filename: YYYY-MM-DD...TASK_ID.eval
        match = re.search(r'_([A-Za-z0-9]+)\.eval$', eval_file.name)
        if match:
            file_id = match.group(1)
            if file_id in eval_ids:
                matched_files.append(eval_file)

    return matched_files


def write_csv(logs: List[LogData], output_path: Path, message_limit: int) -> None:
    """Write parsed log data to CSV file."""
    # Sort by task name, then model
    logs.sort(key=lambda x: (x.task_name, x.model))

    with output_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)

        # Write header
        writer.writerow([
            'task_name',
            'model',
            'accuracy',
            'mean_score',
            'total_samples',
            'samples_hit_limit',
            'run_time',
            'log_path',
        ])

        # Write data rows
        for log in logs:
            writer.writerow([
                log.task_name,
                log.model,
                f"{log.accuracy:.4f}",
                f"{log.mean_score:.4f}",
                log.total_samples,
                log.samples_hit_limit,
                log.run_time.isoformat(),
                str(log.log_path),
            ])

    print(f"Wrote {len(logs)} entries to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse eval files from logs_all based on ablation log IDs"
    )
    parser.add_argument(
        '--log-dir-15',
        type=Path,
        default=Path('ablations/logs_msg15'),
        help='Directory containing message_limit=15 log files (default: logs_exp_msg15)',
    )
    parser.add_argument(
        '--log-dir-30',
        type=Path,
        default=Path('ablations/logs_msg30'),
        help='Directory containing message_limit=15 log files (default: logs_exp_msg30)',
    )
    parser.add_argument(
        '--log-dir-50',
        type=Path,
        default=Path('ablations/logs_msg50'),
        help='Directory containing message_limit=50 log files (default: logs_exp_msg50)',
    )
    parser.add_argument(
        '--eval-dir',
        type=Path,
        default=Path('logs_all'),
        help='Directory containing all .eval files (default: logs_all)',
    )
    parser.add_argument(
        '--output-15',
        type=Path,
        default=Path('logs_msg15_summary.csv'),
        help='Output CSV file for message_limit=15 (default: logs_msg15_summary.csv)',
    )
    parser.add_argument(
        '--output-30',
        type=Path,
        default=Path('logs_msg30_summary.csv'),
        help='Output CSV file for message_limit=15 (default: logs_msg30_summary.csv)',
    )
    parser.add_argument(
        '--output-50',
        type=Path,
        default=Path('logs_msg50_summary.csv'),
        help='Output CSV file for message_limit=50 (default: logs_msg50_summary.csv)',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Check directories exist
    if not args.eval_dir.exists():
        print(f"Error: Eval directory not found: {args.eval_dir}", file=sys.stderr)
        return 1

    # Process message_limit=15
    if args.log_dir_15.exists():
        print(f"\nScanning {args.log_dir_15} for eval IDs (message_limit=15)...")
        eval_ids_15 = collect_eval_ids_from_logs(args.log_dir_15)
        print(f"Found {len(eval_ids_15)} unique eval IDs")

        print(f"Finding corresponding .eval files in {args.eval_dir}...")
        eval_files_15 = find_eval_files_by_ids(args.eval_dir, eval_ids_15)
        print(f"Found {len(eval_files_15)} matching .eval files")

        print("Parsing .eval files...")
        logs_15: List[LogData] = []
        for eval_file in eval_files_15:
            log_data = parse_eval_file(eval_file)
            if log_data:
                logs_15.append(log_data)

        if logs_15:
            write_csv(logs_15, args.output_15, message_limit=15)
        else:
            print("Warning: No valid logs found for message_limit=15", file=sys.stderr)
    else:
        print(f"Warning: Directory not found: {args.log_dir_15}", file=sys.stderr)

    
    # Process message_limit=30
    if args.log_dir_30.exists():
        print(f"\nScanning {args.log_dir_30} for eval IDs (message_limit=30)...")
        eval_ids_30 = collect_eval_ids_from_logs(args.log_dir_30)
        print(f"Found {len(eval_ids_30)} unique eval IDs")

        print(f"Finding corresponding .eval files in {args.eval_dir}...")
        eval_files_30 = find_eval_files_by_ids(args.eval_dir, eval_ids_30)
        print(f"Found {len(eval_files_30)} matching .eval files")

        print("Parsing .eval files...")
        logs_30: List[LogData] = []
        for eval_file in eval_files_30:
            log_data = parse_eval_file(eval_file)
            if log_data:
                logs_30.append(log_data)

        if logs_30:
            write_csv(logs_30, args.output_30, message_limit=30)
        else:
            print("Warning: No valid logs found for message_limit=30", file=sys.stderr)
    else:
        print(f"Warning: Directory not found: {args.log_dir_30}", file=sys.stderr)

    # Process message_limit=50
    if args.log_dir_50.exists():
        print(f"\nScanning {args.log_dir_50} for eval IDs (message_limit=50)...")
        eval_ids_50 = collect_eval_ids_from_logs(args.log_dir_50)
        print(f"Found {len(eval_ids_50)} unique eval IDs")

        print(f"Finding corresponding .eval files in {args.eval_dir}...")
        eval_files_50 = find_eval_files_by_ids(args.eval_dir, eval_ids_50)
        print(f"Found {len(eval_files_50)} matching .eval files")

        print("Parsing .eval files...")
        logs_50: List[LogData] = []
        for eval_file in eval_files_50:
            log_data = parse_eval_file(eval_file)
            if log_data:
                logs_50.append(log_data)

        if logs_50:
            write_csv(logs_50, args.output_50, message_limit=50)
        else:
            print("Warning: No valid logs found for message_limit=50", file=sys.stderr)
    else:
        print(f"Warning: Directory not found: {args.log_dir_50}", file=sys.stderr)

    return 0


if __name__ == '__main__':
    sys.exit(main())
