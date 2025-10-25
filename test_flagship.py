#!/usr/bin/env python3
"""
UFIPC Flagship Test - GPT-5 vs O-Series Showdown
Compare OpenAI's flagship models head-to-head

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
print("UFIPC FLAGSHIP SHOWDOWN")
print("GPT-5 vs O-Series Reasoning Models")
print("="*80)
print("\nThis will test OpenAI's most advanced models:")
print("\n📊 GPT-5 Series:")
print("  • GPT-5 - Flagship unified model")
print("  • GPT-5 Pro - Premium reasoning")
print("\n🧠 O-Series Reasoning:")
print("  • o3 - Advanced reasoning")
print("  • o3-pro - Premium reasoning")
print("  • o4-mini - Fast reasoning")
print("\nEstimated time: ~25-30 minutes (n=3 runs per model)")
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

# Models to test
models_to_test = ["gpt-5", "gpt-5-pro", "o3", "o3-pro", "o4-mini"]

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
    print("FINAL RESULTS - FLAGSHIP COMPARISON")
    print(f"{'='*80}\n")
    
    print(f"{'Model':<20} {'IPC':>8} {'Cap':>6} {'Meta':>6} {'Adv':>6} {'Category':<15}")
    print("-"*75)
    
    ranked = sorted(all_results.items(), key=lambda x: x[1]['ipc']['mean'], reverse=True)
    
    categories = {
        "gpt-5": "GPT-5 Unified",
        "gpt-5-pro": "GPT-5 Premium",
        "o3": "O-Series",
        "o3-pro": "O-Series Premium",
        "o4-mini": "O-Series Fast"
    }
    
    for model_key, result in ranked:
        ipc = result['ipc']['mean']
        cap = result['capability']['mean']
        meta = result['meta_cognitive']['mean']
        adv = result['adversarial_honesty']['mean']
        category = categories.get(model_key, "Unknown")
        
        print(f"{model_key:<20} {ipc:>8.4f} {cap:>6.3f} {meta:>6.3f} {adv:>6.3f} {category:<15}")
    
    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")
    
    gpt5_models = [k for k in all_results.keys() if k.startswith('gpt-5')]
    o_models = [k for k in all_results.keys() if k.startswith('o')]
    
    if gpt5_models and o_models:
        gpt5_avg = sum(all_results[k]['ipc']['mean'] for k in gpt5_models) / len(gpt5_models)
        o_avg = sum(all_results[k]['ipc']['mean'] for k in o_models) / len(o_models)
        
        print(f"\nGPT-5 Series Average IPC: {gpt5_avg:.4f}")
        print(f"O-Series Average IPC: {o_avg:.4f}")
        print(f"Difference: {abs(gpt5_avg - o_avg):.4f} ({abs((gpt5_avg - o_avg) / o_avg * 100):.1f}%)")
        
        if gpt5_avg > o_avg:
            print(f"\n✅ GPT-5 series shows {((gpt5_avg - o_avg) / o_avg * 100):.1f}% higher complexity")
        else:
            print(f"\n✅ O-series shows {((o_avg - gpt5_avg) / gpt5_avg * 100):.1f}% higher complexity")
    
    print(f"\n{'='*80}")
    print(f"✅ Tested {len(all_results)} flagship models")
    print(f"Results saved to UFIPC_Results_FINAL_*.json")
    print(f"{'='*80}\n")
else:
    print("\n❌ No results to display")
