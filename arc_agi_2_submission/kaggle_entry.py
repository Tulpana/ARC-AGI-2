#!/usr/bin/env python3
"""
Kaggle Competition Entry Point for ARC-AGI-2
Wrapper script that executes the RIL solver with Kaggle-compatible defaults.
"""

import json
import os
import sys
from pathlib import Path

# Import core prediction functions
from predict_competition import (
    validate_environment,
    setup_device,
    load_settings,
    run_ril_prediction,
    write_predictions,
)


KAGGLE_DATA_ROOT = Path("/kaggle/input/arc-prize-2025")
FALLBACK_DATASETS = (
    Path("/kaggle/input/arc-agi-2-public-dataset"),
    Path("/kaggle/input/arc-prize-2024"),
    Path("."),
    Path("data"),
)

def get_task_ids_from_aggregated_file(file_path):
    """Extract task IDs from Kaggle's aggregated test file"""
    try:
        with open(file_path, 'r') as f:
            aggregated_blob = json.load(f)

        task_ids = []
        if isinstance(aggregated_blob, dict):
            task_ids = list(aggregated_blob.keys())
        elif isinstance(aggregated_blob, list):
            seen = set()
            for entry in aggregated_blob:
                if not isinstance(entry, dict):
                    continue
                task_obj = entry.get("task") if "task" in entry else entry
                task_id = entry.get("task_id") or entry.get("id")
                if task_id and isinstance(task_obj, dict) and task_id not in seen:
                    seen.add(task_id)
                    task_ids.append(task_id)

        if not task_ids:
            raise ValueError("No task IDs found in aggregated file")

        print(f"✅ Extracted {len(task_ids)} task IDs from {file_path}")
        return task_ids
    except Exception as e:
        print(f"❌ Failed to extract task IDs: {e}")
        return None

def main():
    """Kaggle entry point - executes solver on competition data"""
    print("🚀 ARC-AGI-2 Rule Induction Layer - Kaggle Submission")
    print("=" * 60)
    
    # Setup environment with safe defaults
    validate_environment()
    
    # Setup device
    device = setup_device()
    
    # Load settings
    settings_path = "SETTINGS.json"
    if not Path(settings_path).exists():
        print(f"❌ SETTINGS.json not found")
        sys.exit(1)
    
    settings = load_settings(settings_path)
    if not settings:
        sys.exit(1)
    
    # Find Kaggle test data file
    kaggle_test_paths = [
        Path("/kaggle/input/arc-agi-2-public-dataset/arc-agi_test_challenges.json"),
        Path("/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json"),
        Path("arc-agi_test_challenges.json"),
        Path("data/arc-agi_test_challenges.json")
    ]
    
    test_file = None
    for path in kaggle_test_paths:
        if path.exists():
            test_file = path
            break
    
    if not test_file:
        print(f"❌ Could not find arc-agi_test_challenges.json in any expected location")
        print(f"   Tried: {[str(p) for p in kaggle_test_paths]}")
        sys.exit(1)
    
    print(f"📁 Test data: {test_file}")
    
    # Extract task IDs from aggregated file
    task_ids = get_task_ids_from_aggregated_file(test_file)
    if not task_ids:
        sys.exit(1)
    
    print(f"📋 Processing {len(task_ids)} tasks")
    
    # Run predictions
    predictions, policy_name, total_cases = run_ril_prediction(
        task_ids, settings, topk=2
    )

    # Write output
    output_path = "submission.json"
    write_predictions(predictions, output_path, format_type="json", topk=2)

    # Summary
    total_entries = sum(len(entries) for entries in predictions.values())
    print(f"\n📊 Execution Summary:")
    print(f"   Tasks processed: {len(predictions)}")
    print(f"   Total test inputs: {total_entries}")
    if policy_name:
        print(f"   Policy: {policy_name}")
    if total_cases is not None:
        print(f"   Cases evaluated: {total_cases}")
    print(f"   Output: {output_path}")
    print(f"\n✅ Kaggle submission ready!")

if __name__ == "__main__":
    main()
