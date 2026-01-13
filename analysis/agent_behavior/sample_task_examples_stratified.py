#!/usr/bin/env python3

"""
Sample task examples from Inspect AI eval files with STRATIFICATION by completion status.

This script stratifies samples into three categories:
1. Complete + Correct: Agent got the right answer and didn't hit message limit
2. Complete + Wrong: Agent got wrong answer (or no answer extracted) and didn't hit message limit
3. Incomplete: Agent hit the message limit (determined by explicit limit events in log)

NOTE: We do NOT use 'C' or 'I' from scores.match.value to determine completion status because:
      - 'C' = complete + correct (unambiguous)
      - 'I' = AMBIGUOUS - could mean complete + wrong OR incomplete (hit limit)

Instead, we:
- Check for explicit sample_limit events in the log to identify incomplete samples
- Compare target and answer directly to determine correctness
- This gives us clean 3-way stratification without ambiguity

This ensures balanced analysis across different agent behaviors.
"""

import argparse
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional

import pandas as pd


def load_summary_csvs(csv_dir: Path) -> Dict[int, pd.DataFrame]:
    """Load all summary CSV files (logs_msg{N}_summary.csv)."""
    summaries = {}

    for msg_limit in [15, 30, 50]:
        csv_path = csv_dir / f"logs_msg{msg_limit}_summary.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            summaries[msg_limit] = df
            print(f"[info] Loaded {len(df)} rows from {csv_path.name}")
        else:
            print(f"[warn] File not found: {csv_path}")

    return summaries


def filter_agentic_tasks(df: pd.DataFrame, exclude_no_offline_prompt: bool = True) -> pd.DataFrame:
    """
    Filter out non-agentic (simple) tasks and optionally no_offline_prompt tasks.

    Args:
        df: DataFrame with task_name column
        exclude_no_offline_prompt: If True, exclude tasks with 'no_offline_prompt' in name (default: True)

    Returns:
        Filtered DataFrame
    """
    # Always filter out simple tasks
    filtered_df = df[~df['task_name'].str.contains('simple', case=False, na=False)]
    simple_filtered = len(df) - len(filtered_df)
    print(f"  Filtered out {simple_filtered} simple (non-agentic) tasks")

    # Optionally filter out no_offline_prompt tasks
    if exclude_no_offline_prompt:
        filtered_df = filtered_df[~filtered_df['task_name'].str.contains('no_offline_prompt', case=False, na=False)]
        no_offline_filtered = len(df) - simple_filtered - len(filtered_df)
        if no_offline_filtered > 0:
            print(f"  Filtered out {no_offline_filtered} no_offline_prompt tasks")

    return filtered_df


def extract_samples_from_eval(eval_path: Path) -> List[Dict[str, Any]]:
    """Extract all task samples from an Inspect AI .eval ZIP file."""
    samples = []

    try:
        with zipfile.ZipFile(eval_path, 'r') as zf:
            header = json.loads(zf.read('header.json'))
            task_name = header.get('eval', {}).get('task', 'unknown')
            model = header.get('eval', {}).get('model', 'unknown')

            sample_files = [f for f in zf.namelist()
                          if f.startswith('samples/') and f.endswith('.json')]

            for sample_file in sample_files:
                sample_data = json.loads(zf.read(sample_file))

                sample_data['_eval_metadata'] = {
                    'task_name': task_name,
                    'model': model,
                    'eval_path': str(eval_path),
                    'sample_file': sample_file
                }

                samples.append(sample_data)

    except Exception as e:
        print(f"  [error] Failed to read {eval_path}: {e}")
        return []

    return samples


def classify_sample(sample: Dict[str, Any], message_limit: int) -> str:
    """
    Classify a sample using 3-way stratification.

    NOTE: We do NOT use 'C' or 'I' from scores.match.value because 'I' is ambiguous:
          - 'C' means complete + correct
          - 'I' means either complete + wrong OR incomplete (hit limit)

    3-way stratification:
        1. complete_correct: Got correct answer AND no message limit hit
        2. complete_wrong: Got wrong answer OR couldn't extract answer, but no message limit hit
        3. incomplete: Explicitly hit message limit (sample_limit event present)

    Returns:
        - 'complete_correct': Target matches answer, no limit hit
        - 'complete_wrong': Target doesn't match answer or no answer extracted, no limit hit
        - 'incomplete': Hit message limit (has explicit sample_limit event)
    """
    scores = sample.get('scores', {})
    match_score = scores.get('match', {})

    # Check if limit was explicitly exceeded using sample_limit event
    events = sample.get('events', [])
    limit_events = [e for e in events if e.get('event') == 'sample_limit']

    # If there's a sample_limit event, classify as incomplete
    if limit_events:
        return 'incomplete'

    # No limit hit - check correctness by comparing target and answer directly
    target = sample.get('target', '')
    answer = match_score.get('answer', '')

    # Normalize for comparison
    target_normalized = str(target).strip().lower()
    answer_normalized = str(answer).strip().lower()

    is_correct = (target_normalized == answer_normalized) and (target_normalized != '')

    # Classify based on correctness only (ignore C/I status)
    if is_correct:
        return 'complete_correct'
    else:
        # Wrong answer or couldn't extract answer (but didn't hit limit)
        return 'complete_wrong'


def stratified_sample(
    samples: List[Dict[str, Any]],
    message_limit: int,
    samples_per_stratum: int,
    rng: random.Random,
    fallback_sampling: bool = True
) -> List[Dict[str, Any]]:
    """
    Sample examples with 3-way stratification.

    Args:
        samples: All available samples
        message_limit: Message limit for this configuration
        samples_per_stratum: Number of samples to take from each stratum
        rng: Random number generator
        fallback_sampling: If True, when a stratum is empty, sample extra from available strata

    Returns:
        List of sampled examples (up to 3 * samples_per_stratum)
    """
    # Stratify samples
    strata = defaultdict(list)
    for sample in samples:
        stratum = classify_sample(sample, message_limit)
        strata[stratum].append(sample)

    # Print stratum sizes (3 strata now)
    print(f"    Strata: complete_correct={len(strata['complete_correct'])}, "
          f"complete_wrong={len(strata['complete_wrong'])}, "
          f"incomplete={len(strata['incomplete'])}")

    # Check for empty strata
    stratum_names = ['complete_correct', 'complete_wrong', 'incomplete']
    empty_strata = [name for name in stratum_names if len(strata[name]) == 0]

    if empty_strata:
        print(f"    [warn] Empty strata: {', '.join(empty_strata)}")
        if fallback_sampling:
            print(f"    [info] Will use fallback sampling from available strata")

    # Sample from each stratum
    sampled = []
    samples_needed = samples_per_stratum * 3  # Target total samples (3 strata)

    for stratum_name in stratum_names:
        stratum_samples = strata[stratum_name]
        if stratum_samples:
            rng.shuffle(stratum_samples)
            take = min(samples_per_stratum, len(stratum_samples))
            sampled.extend(stratum_samples[:take])
            if len(stratum_samples) < samples_per_stratum:
                print(f"    [warn] {stratum_name}: only {len(stratum_samples)} available (wanted {samples_per_stratum})")

    # Fallback sampling if we don't have enough samples and fallback is enabled
    if fallback_sampling and len(sampled) < samples_needed:
        deficit = samples_needed - len(sampled)
        print(f"    [info] Deficit of {deficit} samples, applying fallback sampling")

        # Collect remaining samples from all non-empty strata
        remaining_samples = []
        for stratum_name in stratum_names:
            stratum_samples = strata[stratum_name]
            if stratum_samples:
                # Add samples not already selected
                already_sampled = [s for s in sampled if classify_sample(s, message_limit) == stratum_name]
                remaining = [s for s in stratum_samples if s not in already_sampled]
                remaining_samples.extend(remaining)

        # Sample additional examples from remaining pool
        if remaining_samples:
            rng.shuffle(remaining_samples)
            additional = min(deficit, len(remaining_samples))
            sampled.extend(remaining_samples[:additional])
            print(f"    [info] Added {additional} additional samples via fallback")

    return sampled


def sample_task_examples(
    summaries: Dict[int, pd.DataFrame],
    samples_per_stratum: int,
    random_seed: int,
    csv_dir: Path,
    fallback_sampling: bool = True,
    exclude_no_offline_prompt: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Sample N task examples per stratum for each (model, task, message_limit) combination.

    Args:
        summaries: Loaded summary CSV data
        samples_per_stratum: Number of samples to take from each stratum
        random_seed: Random seed for reproducibility
        csv_dir: Directory containing eval files
        fallback_sampling: If True, use fallback sampling when strata are empty
        exclude_no_offline_prompt: If True, exclude no_offline_prompt tasks (default: True)

    Returns:
        Dict mapping "model|task|msg_limit" -> list of sampled task examples
    """
    rng = random.Random(random_seed)
    sampled = defaultdict(list)

    for msg_limit, df in summaries.items():
        df = filter_agentic_tasks(df, exclude_no_offline_prompt=exclude_no_offline_prompt)
        grouped = df.groupby(['model', 'task_name'])

        for (model, task_name), group in grouped:
            combo_key = f"{model}|{task_name}|msg{msg_limit}"

            # Collect all task examples from all eval files
            all_examples = []
            for eval_path_str in group['log_path'].tolist():
                eval_path = csv_dir / eval_path_str if not Path(eval_path_str).is_absolute() else Path(eval_path_str)

                if not eval_path.exists():
                    continue

                samples = extract_samples_from_eval(eval_path)
                all_examples.extend(samples)

            if not all_examples:
                continue

            # Stratified sampling
            selected = stratified_sample(all_examples, msg_limit, samples_per_stratum, rng, fallback_sampling)

            sampled[combo_key] = selected

            print(f"  [{combo_key}] Sampled {len(selected)} examples (stratified) from {len(all_examples)} available")

    return sampled


def format_sample_for_output(sample: Dict[str, Any], stratum_name: str = None) -> Dict[str, Any]:
    """Format a task sample for JSONL output."""
    metadata = sample.get('_eval_metadata', {})

    # Add analysis fields
    messages = sample.get('messages', [])
    non_system_msgs = [m for m in messages if m.get('role') != 'system']
    num_messages = len(non_system_msgs)

    scores = sample.get('scores', {})
    match_score = scores.get('match', {})

    # Check completion status
    completion_status = match_score.get('value', '')
    is_complete = (completion_status.upper() == 'C') if isinstance(completion_status, str) else False

    # Check correctness
    target = sample.get('target', '')
    answer = match_score.get('answer', '')
    is_correct = (str(target) == str(answer))

    result = {
        'task_name': metadata.get('task_name', 'unknown'),
        'model': metadata.get('model', 'unknown'),
        'eval_path': metadata.get('eval_path', ''),
        'sample_file': metadata.get('sample_file', ''),
        'input': sample.get('input', ''),
        'target': sample.get('target', ''),
        'messages': sample.get('messages', []),
        'output': sample.get('output', {}),
        'scores': sample.get('scores', {}),
        'metadata': sample.get('metadata', {}),
        '_analysis': {
            'num_messages': num_messages,
            'is_complete': is_complete,
            'is_correct': is_correct,
            'completion_status': completion_status,
            'answer': answer
        }
    }

    # Add stratum_name if provided
    if stratum_name:
        result['stratum_name'] = stratum_name

    return result


def save_sampled_examples(
    sampled: Dict[str, List[Dict[str, Any]]],
    output_dir: Path,
    combined_output: Optional[Path] = None
) -> None:
    """Save sampled task examples to JSONL files.

    Args:
        sampled: Dict mapping combo keys to sample lists
        output_dir: Directory for per-combination JSONL files
        combined_output: Optional path for single combined JSONL file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    stratum_counts = defaultdict(int)
    all_samples = []  # Collect all samples for combined output

    for combo_key, examples in sampled.items():
        model, task, msg_limit_str = combo_key.split('|')
        msg_limit = int(msg_limit_str.replace('msg', ''))

        safe_model = model.replace('/', '_').replace('@', '_')
        filename = f"{safe_model}__{task}__{msg_limit_str}.jsonl"
        output_path = output_dir / filename

        # Write per-combination file
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in examples:
                # Classify sample to get stratum name
                stratum = classify_sample(example, msg_limit)
                stratum_counts[stratum] += 1

                # Format with stratum name
                formatted = format_sample_for_output(example, stratum_name=stratum)

                # Add combination metadata for combined file
                if combined_output:
                    formatted['_combination'] = {
                        'model': model,
                        'task_name': task,
                        'message_limit': msg_limit,
                        'combo_key': combo_key
                    }
                    all_samples.append(formatted)

                f.write(json.dumps(formatted, ensure_ascii=False) + '\n')

        total_saved += len(examples)
        print(f"  [write] {output_path.name}: {len(examples)} examples")

    # Save combined file if requested
    if combined_output:
        print(f"\n[info] Saving combined file to {combined_output}")
        combined_output.parent.mkdir(parents=True, exist_ok=True)

        with open(combined_output, 'w', encoding='utf-8') as f:
            for sample in all_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')

        file_size_mb = combined_output.stat().st_size / (1024 * 1024)
        print(f"  [write] {combined_output.name}: {len(all_samples)} examples ({file_size_mb:.2f} MB)")

    print(f"\n[summary] Saved {total_saved} task examples across {len(sampled)} combinations")
    print(f"\n[strata] Distribution:")
    for stratum, count in sorted(stratum_counts.items()):
        print(f"  {stratum}: {count} examples")


def main():
    parser = argparse.ArgumentParser(
        description="Sample task examples with stratification by completion status"
    )
    parser.add_argument(
        "--csv-dir",
        type=Path,
        default=Path("."),
        help="Directory containing logs_msg{N}_summary.csv files (default: current dir)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis/agent_behavior/outputs/sampled_task_examples_stratified"),
        help="Output directory for JSONL files (default: analysis/agent_behavior/outputs/sampled_task_examples_stratified)"
    )
    parser.add_argument(
        "--samples-per-stratum",
        type=int,
        default=2,
        help="Number of examples to sample per stratum (default: 2, total ~6 per combo with 3 strata)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling (default: 42)"
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable fallback sampling when strata are empty (default: fallback enabled)"
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        help="Optional: Save all samples to a single JSONL file (in addition to per-combo files)"
    )
    parser.add_argument(
        "--include-no-offline-prompt",
        action="store_true",
        help="Include no_offline_prompt tasks (default: exclude them)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("STRATIFIED TASK EXAMPLE SAMPLING FOR AGENT BEHAVIOR ANALYSIS")
    print("=" * 80)
    print(f"CSV Directory: {args.csv_dir}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Samples per stratum: {args.samples_per_stratum} (3 strata = ~{args.samples_per_stratum * 3} total per combo)")
    print(f"Random seed: {args.seed}")
    print(f"Fallback sampling: {'disabled' if args.no_fallback else 'enabled (default)'}")
    print(f"No-offline-prompt tasks: {'included' if args.include_no_offline_prompt else 'excluded (default)'}")
    print()

    summaries = load_summary_csvs(args.csv_dir)

    if not summaries:
        print("[error] No summary CSV files found")
        return

    print("\n[info] Sampling task examples with stratification...")
    sampled = sample_task_examples(
        summaries,
        args.samples_per_stratum,
        args.seed,
        args.csv_dir,
        fallback_sampling=not args.no_fallback,
        exclude_no_offline_prompt=not args.include_no_offline_prompt
    )

    print(f"\n[info] Saving sampled examples to {args.output_dir}/")
    save_sampled_examples(sampled, args.output_dir, args.combined_output)

    print("\n✓ Stratified sampling complete!")
    print(f"\nNext steps:")
    print(f"  1. Review samples by stratum in: {args.output_dir}/")
    if args.combined_output:
        print(f"  2. Use combined file for bulk analysis: {args.combined_output}")
        print(f"  3. Filter by model/task using '_combination' field")
        print(f"  4. Compare agent behavior across completion status")
    else:
        print(f"  2. Compare agent behavior across completion status (3 strata)")
        print(f"  3. Analyze message limit failures in 'incomplete' examples")
    if not args.no_fallback:
        print(f"  {'5' if args.combined_output else '4'}. Check console output for warnings about empty or sparse strata")


if __name__ == "__main__":
    main()
