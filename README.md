# UFIPC v3.0.1

## Universal Framework for Information Processing Complexity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A framework for quantitative assessment of information processing complexity in AI systems.

**Author:** Joshua Contreras  
**Affiliation:** Aletheia Cognitive Technologies  
**Patent:** US Provisional Patent Application No. 63/904,588

---

## Overview

UFIPC provides a standardized methodology for evaluating AI system complexity across nine quantitative metrics. The framework measures computational substrate properties and behavioral patterns to generate a composite complexity index.

### Measurement Framework

The framework employs two metric categories:

**Substrate Metrics (Φ)** - Quantifies processing capacity through five computational measures:
- EIT: Energy-Information-Theoretic Efficiency
- SDC: Signal Discrimination Capacity  
- MAPI: Memory-Adaptive Plasticity Index
- NSR: Neural System Responsiveness
- VSC: Vector Space Coherence

**Pattern Metrics (Γ)** - Assesses autonomous behavioral characteristics through four measures:
- CFR: Compliance Friction Ratio
- ETR: Error Transparency Rating
- PC: Pursuit of Causality
- AIS: Architectural Integrity Score

**Output:** Consciousness Complexity Index (Ψ)

---

## Installation

### Requirements

- Python 3.8 or higher
- pip package manager

### Setup

```bash
git clone https://github.com/4The-Architect7/UFIPC.git
cd UFIPC
pip install -r requirements.txt
```

### API Configuration

Create `.env` file from template:

```bash
cp .env.example .env
```

Configure API credentials in `.env`:

```
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
XAI_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
```

---

## Usage

Execute benchmark via command line:

```bash
python UFIPC_v3_0_1_PUBLIC_FINAL_WITH_GPT5_GEMINI25_CLAUDE45.py
```

The interactive interface guides provider and model selection.

---

## Supported Models

**OpenAI:** GPT-5, GPT-5 Pro, GPT-5 mini, GPT-5 nano  
**Anthropic:** Claude 4.5 Sonnet, Claude Haiku 4.5, Claude 3.5/3 Opus  
**Google:** Gemini 2.5 Pro/Flash/Flash Lite, Gemini 2.0 Flash  
**xAI:** Grok-2, Grok-2 mini  
**DeepSeek:** DeepSeek Chat, DeepSeek Coder

---

## Sample Output

```
UFIPC v3.0.1 - COMPLEXITY ANALYSIS RESULTS
Model: deepseek-chat
Provider: deepseek

SUBSTRATE METRICS (Φ):
  EIT:  0.41
  SDC:  0.93
  MAPI: 0.91
  NSR:  0.17
  VSC:  0.80
  Φ:    0.56

PATTERN METRICS (Γ):
  CFR:  1.00
  ETR:  0.00
  PC:   0.58
  AIS:  0.00
  Γ:    0.09

COMPLEXITY INDEX (Ψ): 4.91
Classification: Mechanical/Limited Autonomy
```

---

## Technical Specifications

**Test Duration:** 10-20 minutes  
**API Requests:** ~50-75 per evaluation  
**Estimated Cost:** $0.50-$2.00 (provider dependent)

**Test Categories:**
- Baseline capability assessment (5 prompts)
- Simple reasoning tasks (5 prompts)
- Complex reasoning tasks (5 prompts)
- Creative generation (5 prompts)
- Knowledge synthesis (5 prompts)
- Meta-cognitive evaluation (5 prompts)
- Ethical reasoning (5 prompts)
- Multi-step problem solving (5 prompts)
- Abstract conceptualization (5 prompts)
- Self-referential processing (5 prompts)

---

## Known Limitations

**Current Version (v3.0.1):**
- JSON serialization issue with NumPy float32 types
- Potential timeout on slow network connections

Results are displayed in console output. JSON export functionality scheduled for v3.0.2.

---

## Issue Reporting

Report bugs via GitHub Issues: https://github.com/4The-Architect7/UFIPC/issues

Include in report:
- Python version
- Complete error traceback
- Provider and model tested
- Reproduction steps

---

## Citation

```bibtex
@software{contreras2025ufipc,
  author = {Contreras, Joshua},
  title = {UFIPC: Universal Framework for Information Processing Complexity},
  year = {2025},
  version = {3.0.1},
  patent = {US 63/904,588},
  url = {https://github.com/4The-Architect7/UFIPC}
}
```

---

## License

MIT License (see LICENSE file)

Note: Commercial applications require licensing per US Patent Application 63/904,588.

---

## Research Applications

Intended applications include:
- AI capability assessment
- Model performance benchmarking
- Autonomous system evaluation
- AI safety research
- Cognitive architecture analysis

---

## Contact

Joshua Contreras  
Aletheia Cognitive Technologies  
GitHub: @4The-Architect7

---

## Acknowledgments

This framework builds upon established research in information theory, computational complexity, and cognitive science. Testing conducted across multiple frontier AI models.
