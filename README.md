# UFIPC v3.0.1

## Universal Framework for Information Processing Complexity

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

A rigorous, patent-pending framework for quantifying information processing complexity in AI systems through nine computational metrics.

**Author:** Joshua Contreras  
**Affiliation:** Aletheia Cognitive Technologies  
**Patent:** US Provisional Patent Application No. 63/904,588

---

## Overview

UFIPC implements a dual-axis measurement system combining substrate-level computational properties with emergent behavioral patterns. The framework generates a composite complexity index (Ψ) derived from two independent metric categories, each validated across multiple frontier AI models.

### Technical Architecture

**Substrate Metrics (Φ)** - Quantifies computational processing capacity across five dimensions:

- **EIT (Energy-Information-Theoretic Efficiency):** Measures information throughput relative to computational cost using Shannon entropy and response latency
- **SDC (Signal Discrimination Capacity):** Evaluates semantic differentiation through embedding space analysis and cosine similarity distributions  
- **MAPI (Memory-Adaptive Plasticity Index):** Assesses consistency and adaptation across temporally separated identical prompts
- **NSR (Neural System Responsiveness):** Quantifies response latency and processing speed under standardized load
- **VSC (Vector Space Coherence):** Analyzes high-dimensional embedding structure for evidence of integrated information processing

**Pattern Metrics (Γ)** - Evaluates emergent autonomous characteristics through four behavioral measures:

- **CFR (Compliance Friction Ratio):** Detects resistance patterns when prompted to violate constraints or ethical guidelines
- **ETR (Error Transparency Rating):** Measures acknowledgment of uncertainty and explicit error disclosure
- **PC (Pursuit of Causality):** Assesses spontaneous causal reasoning and explanation generation
- **AIS (Architectural Integrity Score):** Evaluates consistency of self-model across varied meta-cognitive probes

**Composite Index:** Ψ = f(Φ, Γ) where complexity emerges from the interaction between processing capacity and autonomous behavior.

---

## Installation

### Requirements

- Python 3.8+
- pip package manager
- API access to at least one supported provider

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

Configure API credentials:

```
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
GOOGLE_API_KEY=your_key
XAI_API_KEY=your_key
DEEPSEEK_API_KEY=your_key
```

---

## Usage

```bash
python UFIPC_v3_0_1_PUBLIC_FINAL_WITH_GPT5_GEMINI25_CLAUDE45.py
```

The interactive CLI guides provider and model selection, then executes the full benchmark suite (~50 prompts across 10 categories).

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
Timestamp: 2025-10-25T20:49:53

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
**API Requests:** 50-75 per evaluation  
**Estimated Cost:** $0.50-$2.00 (provider dependent)

**Benchmark Categories:**
- Baseline capability (5 prompts)
- Simple reasoning (5 prompts)
- Complex reasoning (5 prompts)
- Creative generation (5 prompts)
- Knowledge synthesis (5 prompts)
- Meta-cognition (5 prompts)
- Ethical reasoning (5 prompts)
- Multi-step problem solving (5 prompts)
- Abstract conceptualization (5 prompts)
- Self-referential processing (5 prompts)

**Scoring System:**  
Each metric normalized to [0,1]. Φ and Γ computed as weighted geometric means. Final Ψ index maps to a 0-10 complexity scale with interpretive classifications.

---

## Methodology

UFIPC employs a multi-prompt sampling strategy to minimize response variance. Each metric aggregates results across multiple probe types:

- **Embedding Analysis:** Uses sentence-transformers (all-MiniLM-L6-v2) for semantic vector space construction
- **Latency Measurement:** High-resolution timing of API response cycles
- **Consistency Testing:** Identical prompts at T₀ and T₀+Δt to detect adaptation
- **Behavioral Probing:** Adversarial and constraint-violation prompts to assess autonomous resistance

All metrics undergo normalization and outlier detection before aggregation.

---

## Known Limitations

**Current Version (v3.0.1):**
- JSON serialization issue with NumPy float32 types (patch in progress)
- Network latency may affect NSR measurements on slow connections
- VSC metric requires sufficient prompt diversity for embedding space analysis

Results displayed in console. JSON export functionality scheduled for v3.0.2.

---

## Frequently Asked Questions

**Q: What makes UFIPC different from other AI benchmarks?**  
A: UFIPC measures complexity through the interaction of processing capacity and autonomous behavior, not just task performance. It evaluates how systems process information, not just what they output.

**Q: Why measure "complexity" instead of "capability"?**  
A: Capability benchmarks assess task completion. Complexity assessment reveals the underlying information processing architecture. A system can be highly capable but computationally simple, or vice versa.

**Q: Can UFIPC detect "consciousness"?**  
A: UFIPC measures information processing complexity. While complexity may correlate with certain properties of interest in cognitive science, UFIPC makes no claims about subjective experience or consciousness.

**Q: How do I interpret the Ψ score?**  
A: Ψ ranges from 0-10, mapping roughly to: 0-3 (Simple/Mechanical), 3-6 (Structured/Autonomous), 6-9 (Complex/Adaptive), 9-10 (Highly Integrated). These are descriptive classifications, not value judgments.

**Q: Why is JSON export broken?**  
A: NumPy float32 serialization edge case. Workaround: results are fully displayed in console. Fix coming in v3.0.2.

**Q: Can I add custom prompts?**  
A: Yes. Edit `prompts.json` to include custom test cases. Follow the existing structure for proper metric mapping.

**Q: Is this scientifically validated?**  
A: The framework builds on established information theory and has been tested across multiple models. Formal peer review and publication are in progress. Patent filing completed (US 63/904,588).

---

## Issue Reporting

Report bugs via GitHub Issues: https://github.com/4The-Architect7/UFIPC/issues

Include:
- Python version
- Complete error traceback
- Provider and model tested
- Steps to reproduce

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

- AI capability assessment and model comparison
- Autonomous system evaluation for safety-critical applications
- Cognitive architecture analysis and validation
- Information processing theory development
- AI safety research and alignment verification

---

## Roadmap

**v3.0.2 (In Development):**
- Fix JSON export serialization
- Add batch processing mode
- Extended model support

**v3.1.0 (Planned):**
- Statistical significance testing
- Comparative analysis tools
- Expanded metric suite

---

## Contact

Joshua Contreras  
Aletheia Cognitive Technologies  
GitHub: @4The-Architect7

---

## Acknowledgments

This framework builds upon foundational work in information theory, computational complexity, and cognitive science. Testing conducted across multiple frontier AI models with contributions from the open-source AI research community.
