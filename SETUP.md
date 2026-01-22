# Setup Guide

This guide will help you install and configure the Proof of Time benchmark suite.

## Prerequisites

- **Python 3.10 or higher**
- **Docker** (for sandbox environments)
- **uv** (Python package manager)
- **API keys** for at least one LLM provider (see below)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shan23chen/proof_of_time.git
cd proof_of_time
```

### 1.1 download the files from huggingface and patch them
```bash
export HF_TOKEN="your_hf_token"
huggingface-cli download AIM-Harvard/proof-of-time \
  --repo-type dataset \
  --include "benchmarks/*" \
  --local-dir ./temp_patch

rsync -av ./temp_patch/benchmarks/ ./benchmarks/
rm -rf temp_patch
```

### 2. Install uv (if not already installed)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### 3. Install Dependencies

```bash
# Install all dependencies
uv sync

source .venv/bin/activate
```

### 4. Verify Docker Installation

The benchmarks use Docker containers as sandboxed environments for agents to analyze paper data.

```bash
# Check Docker is running
docker --version
docker ps

# Test sandbox configuration (optional)
cd benchmarks/citation_react
docker compose -f sandbox/compose.yaml up -d
docker compose -f sandbox/compose.yaml down
```

## API Key Configuration

The benchmark suite supports multiple LLM providers. You need API keys for the models you want to test.

### Supported Models

The repository has been tested with the following commercial API models:

**OpenAI Models:**
- `openai/gpt-5.2-2025-12-11`
- `openai/gpt-5.1-2025-11-13`
- `openai/gpt-5-mini-2025-08-07`
- `openai/gpt-5-nano-2025-08-07`

**Google Gemini Models:**
- `google/gemini-3-pro-preview`
- `google/gemini-3-flash-preview`
- `google/vertex/gemini-2.5-pro`
- `google/vertex/gemini-2.5-flash`

**Anthropic Claude Models (via Vertex AI):**
- `anthropic/vertex/claude-haiku-4-5@20251001`
- `anthropic/vertex/claude-opus-4-5@20251101`
- `anthropic/vertex/claude-sonnet-4-5@20250929`

### Setting Up API Keys

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` and add your API keys:
```bash
# Required: Add keys for models you want to test
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
GOOGLE_API_KEY=your_google_key_here

# Optional: For HuggingFace dataset download
HF_TOKEN=your_huggingface_token_here
```

3. Load environment variables:
```bash
# In your shell (or add to ~/.bashrc or ~/.zshrc)
export $(cat .env | xargs)
```

### Getting API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/settings/keys
- **Google AI**: https://makersuite.google.com/app/apikey
- **Google Vertex AI**: https://cloud.google.com/vertex-ai/docs/authentication
- **HuggingFace**: https://huggingface.co/settings/tokens

## Running Your First Benchmark

### Quick Test (5 samples)

```bash
# Run a small test with GPT-5-mini
inspect eval benchmarks/award_react/benchmark.py@pre_cutoff_simple_task \
    --model openai/gpt-5-mini-2025-08-07 \
    --limit 5
```

### Running Full Benchmarks

```bash
# Run all award prediction tasks for one model
inspect eval benchmarks/award_react/benchmark.py \
    --model openai/gpt-5-mini-2025-08-07

# Run citation forecasting (all variants)
inspect eval benchmarks/citation_react/benchmark.py \
    --model openai/gpt-5-mini-2025-08-07

# Run faculty future work prediction
inspect eval benchmarks/future_work_react/benchmark.py \
    --model openai/gpt-5-mini-2025-08-07

# Run SOTA benchmark forecasting
inspect eval benchmarks/sota_forecast/benchmark.py \
    --model openai/gpt-5-mini-2025-08-07
```

### Running Ablation Studies

The repository includes scripts for systematic ablations:

```bash
# Run all benchmarks across multiple models (with/without offline prompt)
uv run scripts/run_inspect_ablations.py

# Run with specific models
uv run scripts/run_inspect_ablations.py \
    --models openai/gpt-5-mini-2025-08-07 google/gemini-3-flash-preview

# Run with different message limits (15, 30, 50)
bash run_message_limit_sweep.sh
```

Logs are written to `logs/ablations/<model-slug>/<task>.log`.

## Downloading the Dataset

Benchmark datasets and sandbox data are available on HuggingFace:

```bash
# Using datasets library
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('AIM-Harvard/proof-of-time')"

# Or manual clone
git clone https://huggingface.co/datasets/AIM-Harvard/proof-of-time
```

## Verifying Your Setup

Run these commands to verify everything is working:

```bash
# 1. Check Python version
python --version  # Should be 3.10+

# 2. Check Inspect AI is installed
inspect --version

# 3. Check Docker is running
docker ps

# 4. Check environment variables
echo $OPENAI_API_KEY  # Should show your key (partially masked)

# 5. Run a minimal test
inspect eval benchmarks/award_react/benchmark.py@pre_cutoff_simple_task \
    --model openai/gpt-5-mini-2025-08-07 \
    --limit 1
```

## Common Issues

### Issue: Docker not running

**Error**: `Cannot connect to the Docker daemon`

**Solution**: Start Docker Desktop or the Docker daemon:
```bash
# macOS
open -a Docker

# Linux
sudo systemctl start docker
```

### Issue: API authentication failed

**Error**: `AuthenticationError` or `Invalid API key`

**Solution**:
1. Verify your API key is correct in `.env`
2. Export environment variables: `export $(cat .env | xargs)`
3. Check API key hasn't expired on the provider's dashboard

### Issue: Network isolation in sandbox

**Error**: Agent reports "Network unreachable" when trying to access external resources

**Solution**: This is expected behavior. The sandbox uses `network_mode: none` for isolation. Agents should only access the mounted data in `/dataset`.

### Issue: Out of memory during evaluation

**Solution**:
- Reduce batch size with `--limit` flag
- Use smaller models (e.g., `gpt-5-mini` instead of `gpt-5.2`)
- Run tasks sequentially instead of in parallel

### Issue: Rate limits

**Error**: `RateLimitError` from API provider

**Solution**:
- Add delays between runs
- Use `--limit` to run fewer samples at a time
- Check your API tier/quota on the provider's dashboard

## Next Steps

- Read [README.md](README.md) for an overview of the benchmark suite
- See [CITATION.md](CITATION.md) for citation information
- Explore `benchmarks/*/README.md` for detailed task descriptions
- Check `analysis/` for result analysis scripts
- Run `scripts/run_inspect_ablations.py --help` for ablation options

## Getting Help

- **Issues**: https://github.com/shan23chen/proof_of_time/issues
- **Documentation**: See README files in each benchmark directory
- **Inspect AI Docs**: https://inspect.ai-safety-institute.org.uk/

## Development Setup (Optional)

For contributing to the codebase:

```bash
# Install development dependencies
uv sync --extra dev

# Install code formatters
pip install ruff black

# Run code formatting
ruff check benchmarks/
black benchmarks/

# Run tests (if available)
pytest tests/
```
