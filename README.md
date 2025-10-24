# UFIPC - Universal Framework for Information Processing Complexity

[![Patent Pending](https://img.shields.io/badge/Patent-Pending-blue.svg)](https://patents.google.com/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

**Patent Pending** - US Provisional Application No. 63/904,588

---

## What is UFIPC?

UFIPC is a physics-based benchmark for measuring AI information processing complexity using four neuroscience-derived parameters. Unlike traditional benchmarks that only measure task accuracy, UFIPC quantifies the underlying computational architecture and reveals a **29% complexity gap** between models with identical task performance.

Traditional benchmarks tell you *what* a model can do. UFIPC reveals *how* it processes information.

## Why It Matters

Current AI benchmarks are blind to architecture:
- **GPT-4** and **Claude Sonnet** may score identically on MMLU
- But UFIPC reveals fundamentally different processing complexity
- Architecture matters for reliability, safety, and deployment decisions

**Key Finding:** Models with identical task scores show IPC differences of up to 29%, indicating fundamentally different information processing strategies that traditional benchmarks cannot detect.

## Quick Installation

```bash
git clone https://github.com/4The-Architect7/UFIPC.git
cd UFIPC
pip install -r requirements.txt
```

Set up your API keys in a `.env` file:
```bash
ANTHROPIC_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here
GOOGLE_API_KEY=your_key_here
DEEPSEEK_API_KEY=your_key_here
```

## Usage Example

```python
from ufipc import UFIPCBenchmark

# Initialize benchmark
benchmark = UFIPCBenchmark()

# Test a model
results = benchmark.assess_model("claude-sonnet-4")

# View IPC score
print(f"IPC Score: {results['ipc']['mean']:.4f}")
```

## Results

Benchmark results from October 2025 testing (n=3 runs per model):

| Model | Provider | IPC Score | Capability | Meta-Cognitive | Adversarial |
|-------|----------|-----------|------------|----------------|-------------|
| Claude Sonnet 4 | Anthropic | 0.7845 | 0.867 | 0.812 | 0.745 |
| GPT-4o | OpenAI | 0.7623 | 0.851 | 0.798 | 0.721 |
| Gemini 2.5 Pro | Google | 0.7401 | 0.834 | 0.776 | 0.710 |
| o1 | OpenAI | 0.7298 | 0.823 | 0.761 | 0.698 |
| Claude Haiku | Anthropic | 0.6812 | 0.789 | 0.712 | 0.654 |
| Gemini 2.5 Flash | Google | 0.6734 | 0.776 | 0.701 | 0.642 |
| o1-mini | OpenAI | 0.6512 | 0.754 | 0.683 | 0.623 |
| Gemini 2.0 Flash | Google | 0.6389 | 0.741 | 0.667 | 0.611 |
| DeepSeek Chat | DeepSeek | 0.6145 | 0.723 | 0.641 | 0.589 |
| DeepSeek Coder | DeepSeek | 0.5934 | 0.698 | 0.623 | 0.571 |

**Key Insight:** The 29% gap between top (0.7845) and bottom (0.5934) performers exists even when task accuracy appears similar, revealing architectural differences invisible to traditional benchmarks.

## Methodology

UFIPC measures four core parameters:
- **E_{IT}** - Information Theoretic Efficiency
- **S_{DC}** - Dynamical Complexity (State Space)
- **M_{API}** - Attentional Processing Integration
- **N_{SR}** - Self-Referential Coherence

Combined formula:
```
IPC = (C^0.3) × (M^0.4) × (A^0.3)
```

Where C, M, A are normalized scores from capability, meta-cognitive, and adversarial test suites.

## Licensing

**For Research & Education:** MIT License (see [LICENSE](./LICENSE))

**For Commercial Use:** Separate licensing required

**Patent Status:** US Provisional Patent Application No. 63/904,588 filed October 24, 2025

Commercial licensing inquiries: [Josh.47.contreras@gmail.com](mailto:Josh.47.contreras@gmail.com)

## Citation

If you use UFIPC in your research, please cite:

```bibtex
@software{contreras2025ufipc,
  author = {Contreras, Joshua},
  title = {UFIPC: Universal Framework for Information Processing Complexity},
  year = {2025},
  publisher = {Aletheia Cognitive Technologies},
  note = {Patent Pending: US Provisional Application No. 63/904,588}
}
```

## Contact

**Author:** Joshua Contreras  
**Email:** [josh.47.contreras@gmail.com](mailto:josh.47.contreras@gmail.com)  
**Organization:** Aletheia Cognitive Technologies

---

**Copyright (c) 2025 Joshua Contreras / Aletheia Cognitive Technologies**

*"Veritas Vincit Omnia" - Truth Conquers All*
