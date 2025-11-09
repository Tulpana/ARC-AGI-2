#!/usr/bin/env python3
"""
Demo script to showcase the ARC-AGI-2 Rule Induction Layer system
"""

import os
import subprocess
import sys
from pathlib import Path


DATASET_ROOT = Path("arc-agi-2-public-dataset").resolve()


def _guard_out(path_like):
    rp = Path(path_like).resolve()
    try:
        rp.relative_to(DATASET_ROOT)
        raise SystemExit(f"[GUARD] refusing to write inside dataset: {rp}")
    except ValueError:
        try:
            rp.relative_to(SPEC_ROOT)
            raise SystemExit(f"[GUARD] refusing to write inside spec root: {rp}")
        except ValueError:
            return rp

def setup_environment():
    """Set up required environment variables"""
    env_vars = {
        "COMP_MODE": "1",
        "EVAL_MODE": "1", 
        "NO_ADAPT": "1",
        "MODEL_FREEZE": "1",
        "DETERMINISTIC": "1",
        "RIL_SEED": "1337",
        "PYTHONHASHSEED": "1337",
        "RIL_DEVICE": "cpu"
    }
    
    for key, value in env_vars.items():
        os.environ[key] = value
        
    print("✅ Environment configured for ARC-AGI-2 Rule Induction Layer")
    return env_vars

def run_demo():
    """Run a demonstration of the system"""
    print("🎯 ARC-AGI-2 Rule Induction Layer System Demo")
    print("=" * 50)
    
    # Setup environment
    setup_environment()
    
    # Create demo output directory
    demo_dir = _guard_out(Path("demo_output"))
    demo_dir.mkdir(exist_ok=True)
    
    # Check if we have the required files
    settings_file = Path("SETTINGS.json")
    manifest_file = _guard_out(Path("test_manifest.txt"))
    
    if not settings_file.exists():
        print(f"❌ Settings file not found: {settings_file}")
        return False
    
    if not manifest_file.exists():
        print(f"❌ Test manifest file not found: {manifest_file}")
        print("Creating a sample test manifest...")
        with manifest_file.open('w') as f:
            f.write("00576224\n007bbfb7\n009d5c81\n")
    
    print(f"✅ Using settings: {settings_file}")
    print(f"✅ Using manifest: {manifest_file}")
    
    # Run the prediction system
    output_file = _guard_out(demo_dir / "demo_predictions.jsonl")
    
    cmd = [
        sys.executable, "predict_competition.py",
        "--settings", str(settings_file),
        "--manifest", str(manifest_file),
        "--topk", "2",
        "--format", "jsonl",
        "--out", str(output_file)
    ]
    
    print(f"\n🔄 Running command: {' '.join(cmd)}")
    print("This may take a moment...")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            print("✅ Demo completed successfully!")
            print(f"📄 Output saved to: {output_file}")
            
            if output_file.exists():
                print(f"📊 Output file size: {output_file.stat().st_size} bytes")
            
            print("\n--- System Output ---")
            print(result.stdout)
            if result.stderr:
                print("\n--- System Messages ---") 
                print(result.stderr)
                
            return True
        else:
            print(f"❌ Demo failed with return code: {result.returncode}")
            print("--- Error Output ---")
            print(result.stderr)
            if result.stdout:
                print("--- Standard Output ---")
                print(result.stdout)
            return False
            
    except subprocess.TimeoutExpired:
        print("⏰ Demo timed out after 2 minutes")
        return False
    except Exception as e:
        print(f"❌ Demo failed with exception: {e}")
        return False

def show_system_info():
    """Show information about the system"""
    print("\n📋 System Information")
    print("-" * 20)
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Check for key files
    key_files = ["SETTINGS.json", "predict_competition.py", "requirements.txt"]
    for filename in key_files:
        if Path(filename).exists():
            print(f"✅ {filename}")
        else:
            print(f"❌ {filename} (missing)")

if __name__ == "__main__":
    print("🚀 Starting ARC-AGI-2 Rule Induction Layer Demo")
    
    show_system_info()
    
    if run_demo():
        print("\n🎉 Demo completed successfully!")
        print("The Rule Induction Layer system is ready for use.")
    else:
        print("\n⚠️  Demo encountered issues.")
        print("Please check the system configuration and try again.")