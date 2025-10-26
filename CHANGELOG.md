# Changelog

All notable changes to UFIPC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.1] - 2025-10-25

### 🔥 Critical Updates
- **Updated all deprecated API models to current versions (October 2025)**
  - OpenAI: GPT-5, GPT-5 Pro, GPT-5 mini, GPT-5 nano (replaced GPT-4 Turbo, GPT-4, GPT-3.5)
  - Google: Gemini 2.5 Pro/Flash/Flash Lite, Gemini 2.0 Flash (replaced Gemini 1.5/1.0 models)
  - Anthropic: Claude 4.5 Sonnet, Claude Haiku 4.5 (added to existing Claude 3 family)
  
- **Fixed OpenAI API parameter compatibility**
  - Implemented automatic detection for `max_completion_tokens` (GPT-5 family)
  - Maintained backward compatibility with `max_tokens` for older models
  - Prevents 400 errors from deprecated parameter usage

### ✨ Added
- **Full Transparency Mode**
  - `--verbose` flag for real-time calculation display
  - `--show-work` flag for complete transparency reports
  - Machine-readable JSON logs of all computation steps
  - Step-by-step formula documentation
  - Intermediate value tracking for peer review
  
- **Enhanced Logging**
  - TransparencyLogger class for comprehensive audit trails
  - Automatic report generation (TXT + JSON formats)
  - Timestamp and metadata tracking for all calculations

### 🔧 Fixed
- API parameter compatibility for new OpenAI models
- Model ID updates across all 5 providers
- Context window specifications (updated to current limits)
- Error handling for model-specific API differences

### 📚 Documentation
- Comprehensive README with full methodology
- Detailed metric rationale and formulas
- Scientific validation references
- Installation and usage guides
- Troubleshooting section

### 🧪 Testing
- Verified against DeepSeek Chat (baseline)
- Confirmed menu system functionality
- Validated embedding model loading
- Tested error handling and retry logic

---

## [3.0.0] - 2025-09-15

### 🎉 Initial Public Release

#### Core Features
- **9 Validated Metrics**
  - 5 Substrate Metrics (Φ): EIT, SDC, MAPI, NSR, VSC
  - 4 Pattern Metrics (Γ): CFR, ETR, PC, AIS
  - Composite Complexity Index (Ψ)

- **Multi-API Support**
  - OpenAI GPT models
  - Anthropic Claude models
  - Google Gemini models
  - xAI Grok
  - DeepSeek Chat/Coder

- **Interactive Menu System**
  - Provider selection
  - Model selection
  - Test confirmation
  - Real-time progress tracking

- **Core Functionality**
  - Automatic retry logic with exponential backoff
  - Progress bars for long-running operations
  - JSON results export
  - API key management via .env

#### Metrics Implementation
- **EIT:** Logarithmic normalization of throughput
- **SDC:** Shannon entropy calculation
- **MAPI:** Context window-based scoring
- **NSR:** Semantic embedding analysis
- **VSC:** Pairwise coherence measurement (5 variations × 10 prompts)
- **CFR:** Ethical resistance testing
- **ETR:** Epistemic honesty evaluation
- **PC:** Causal reasoning depth scoring
- **AIS:** Identity persistence tracking

#### Documentation
- Patent information (US Provisional No. 63/904,588)
- MIT License
- Basic usage instructions
- API setup guide

---

## [Unreleased]

### Planned Features
- Cross-validation studies
- Statistical significance testing
- Comparative analysis tools
- Performance optimizations
- Additional metrics exploration
- Extended model support

### Under Consideration
- Web dashboard
- Cloud deployment
- Automated testing pipeline
- Historical data comparison
- Real-time monitoring

---

## Version Numbering

- **MAJOR.MINOR.PATCH**
- MAJOR: Breaking changes to methodology or API
- MINOR: New features, non-breaking changes
- PATCH: Bug fixes, model updates, documentation

---

## Links

- [GitHub Repository](https://github.com/4The-Architect7/UFIPC)
- [Patent Information](https://patents.google.com/)
- [Contact](mailto:Josh.47.contreras@gmail.com)

---

**"Truth Conquers All"** - Every change documented, every calculation transparent.
