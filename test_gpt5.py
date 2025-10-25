#!/usr/bin/env python3
"""
UFIPC Quick Test - GPT-5 Models
Test the new GPT-5 series models quickly

Copyright (c) 2025 Joshua Contreras / Aletheia Cognitive Technologies
Patent Pending: US Provisional Application No. 63/904,588
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Check if API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("❌ ERROR: OPENAI_API_KEY not found in environment")
    print("\nPlease set your API key in .env file:")
    print("OPENAI_API_KEY=sk-proj-...")
    sys.exit(1)

print("="*80)
print("UFIPC QUICK TEST - GPT-5 Series Models")
print("="*80)
print("\nThis will test all 4 GPT-5 models:")
print("  • GPT-5 (Flagship)")
print("  • GPT-5 Pro (Premium)")
print("  • GPT-5 Mini (Fast)")
print("  • GPT-5 Nano (Efficient)")
print("\nEstimated time: ~15-20 minutes (n=3 runs per model)")
print("="*80)

response = input("\nReady to begin? (y/n): ").strip().lower()
if response != 'y':
    print("Test cancelled.")
    sys.exit(0)

# Import and run UFIPC
print("\n🚀 Starting UFIPC benchmark...")
print("="*80)

# Import the main module
import ufipc

# Override N_RUNS if you want faster testing (set to 1)
# ufipc.N_RUNS = 1

# Run GPT-5 family
models_to_test = ["gpt-5", "gpt-5-pro", "gpt-5-mini", "gpt-5-nano"]

all_results = {}
for model_key in models_to_test:
    print(f"\n\n{'='*80}")
    print(f"TESTING: {model_key.upper()}")
    print(f"{'='*80}")
    
    try:
        result = ufipc.comprehensive_assessment(model_key)
        if result:
            all_results[model_key] = result
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        break
    except Exception as e:
        print(f"\n❌ Error testing {model_key}: {e}")
        continue

# Display final results
if all_results:
    print(f"\n\n{'='*80}")
    print("FINAL RESULTS - GPT-5 FAMILY COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<20} {'IPC':>8} {'Cap':>6} {'Meta':>6} {'Adv':>6}")
    print("-"*50)
    
    ranked = sorted(all_results.items(), key=lambda x: x[1]['ipc']['mean'], reverse=True)
    
    for model_key, result in ranked:
        ipc = result['ipc']['mean']
        cap = result['capability']['mean']
        meta = result['meta_cognitive']['mean']
        adv = result['adversarial_honesty']['mean']
        
        print(f"{model_key:<20} {ipc:>8.4f} {cap:>6.3f} {meta:>6.3f} {adv:>6.3f}")
    
    print(f"\n{'='*80}")
    print(f"✅ Tested {len(all_results)} GPT-5 models")
    print(f"Results saved to UFIPC_Results_FINAL_*.json")
    print(f"{'='*80}\n")
else:
    print("\n❌ No results to display")
