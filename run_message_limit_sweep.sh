#!/bin/bash
# Run run_inspect_ablations.py with different message limits
# Each run uses its own log directory: logs/ablations_msg15, logs/ablations_msg30, logs/ablations_msg50

set -e  # Exit on error

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "Running Inspect AI ablations with different message limits"
echo "================================================================================"
echo ""

# Run with message limit 15
echo "=========================================="
echo "Run 1/3: Message limit = 15"
echo "=========================================="
uv run scripts/run_inspect_ablations.py \
    --message-limit 15 \
    --log-dir logs/ablations_msg15

echo ""
echo "✓ Completed run with message limit 15"
echo ""

# Run with message limit 30
echo "=========================================="
echo "Run 2/3: Message limit = 30"
echo "=========================================="
uv run scripts/run_inspect_ablations.py \
    --message-limit 30 \
    --log-dir logs/ablations_msg30

echo ""
echo "✓ Completed run with message limit 30"
echo ""

# Run with message limit 50
echo "=========================================="
echo "Run 3/3: Message limit = 50"
echo "=========================================="
uv run scripts/run_inspect_ablations.py \
    --message-limit 50 \
    --log-dir logs/ablations_msg50

echo ""
echo "✓ Completed run with message limit 50"
echo ""

echo "================================================================================"
echo "All runs completed successfully!"
echo "================================================================================"
echo ""
echo "Results:"
echo "  Message limit 15: logs/ablations_msg15/"
echo "  Message limit 30: logs/ablations_msg30/"
echo "  Message limit 50: logs/ablations_msg50/"
echo ""
