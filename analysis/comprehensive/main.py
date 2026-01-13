#!/usr/bin/env python3
"""Main entry point for Proof-of-Time comprehensive analysis.

This script:
1. Loads experimental data from all message limit CSV files
2. Generates publication-ready plots
3. Produces a comprehensive markdown report

Usage:
    python main.py                    # Run full analysis
    python main.py --plots-only       # Only generate plots
    python main.py --report-only      # Only generate report

Output:
    analysis/comprehensive/plots/     # All generated figures (PNG + PDF)
    analysis/comprehensive/REPORT.md  # Comprehensive analysis report
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_all_data, get_unique_models, get_unique_tasks
from plots import generate_all_plots
from report_generator import generate_markdown_report


OUTPUT_DIR = Path(__file__).resolve().parent


def print_data_summary(df) -> None:
    """Print a summary of the loaded data."""
    print("\n" + "=" * 60)
    print("DATA SUMMARY")
    print("=" * 60)
    
    print(f"\nTotal rows: {len(df):,}")
    print(f"Message limits: {sorted(df['message_limit'].unique())}")
    print(f"Models: {get_unique_models(df)}")
    print(f"Tasks: {len(get_unique_tasks(df))} unique tasks")
    
    print("\nTask families:")
    for family in sorted(df["task_family"].unique()):
        count = len(df[df["task_family"] == family])
        print(f"  - {family}: {count} rows")
    
    print("\nVariants:")
    for variant in sorted(df["variant"].unique()):
        count = len(df[df["variant"] == variant])
        print(f"  - {variant}: {count} rows")
    
    print("\nModel families:")
    for family in sorted(df["model_family"].unique()):
        count = len(df[df["model_family"] == family])
        models = df[df["model_family"] == family]["short_model"].unique()
        print(f"  - {family}: {count} rows ({', '.join(models)})")
    
    print()


def main():
    """Main entry point for the analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="Proof-of-Time comprehensive analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--plots-only",
        action="store_true",
        help="Only generate plots, skip report generation"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Only generate report, skip plot generation"
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Skip data summary printout"
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("PROOF OF TIME: COMPREHENSIVE ANALYSIS")
    print("=" * 60)
    
    # Step 1: Load data
    print("\n📁 Step 1: Loading Data...")
    try:
        df = load_all_data()
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return 1
    
    if not args.no_summary:
        print_data_summary(df)
    
    # Step 2: Generate plots
    if not args.report_only:
        print("\n📊 Step 2: Generating Plots...")
        try:
            generate_all_plots(df, OUTPUT_DIR)
        except Exception as e:
            print(f"❌ Failed to generate plots: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\n⏭️  Step 2: Skipping plots (--report-only)")
    
    # Step 3: Generate report
    if not args.plots_only:
        print("\n📝 Step 3: Generating Report...")
        try:
            generate_markdown_report(df, OUTPUT_DIR)
        except Exception as e:
            print(f"❌ Failed to generate report: {e}")
            import traceback
            traceback.print_exc()
            return 1
    else:
        print("\n⏭️  Step 3: Skipping report (--plots-only)")
    
    # Done
    print("\n" + "=" * 60)
    print(f"✅ DONE! Results saved to: {OUTPUT_DIR}")
    print("=" * 60 + "\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

