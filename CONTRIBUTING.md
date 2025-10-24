# Contributing to UFIPC

Thank you for your interest in contributing to the Universal Framework for Information Processing Complexity (UFIPC)!

## Patent Notice

**Important:** This project is patent-pending (US Provisional Application No. 63/904,588). By contributing, you agree that your contributions will be licensed under the same MIT License (research/educational use) and that you have the right to contribute the code.

## Ways to Contribute

### 1. Bug Reports
- Check existing issues first
- Provide clear reproduction steps
- Include error messages and logs
- Specify your environment (Python version, OS, model tested)

### 2. Feature Requests
- Describe the use case clearly
- Explain why it would benefit the project
- Consider if it aligns with the physics-based methodology

### 3. Code Contributions

#### Adding New Models

To add support for a new AI model:

1. **Add model configuration** in `MODELS` dict:
```python
"your-model": {
    "name": "model-api-name",
    "provider": "provider-name",
    "rpm": 60,  # requests per minute
    "delay": 1,  # seconds between requests
    "tier": "CATEGORY",
    "context_window": 100000
}
```

2. **Add API integration** in `call_model()` function
3. **Test thoroughly** with n=3 runs minimum
4. **Update documentation** (README.md, model list)

#### Improving Test Suites

- Propose new tests that align with neuroscience parameters
- Ensure tests are objective and scorable
- Provide validation data if possible

### 4. Documentation

- Fix typos or unclear explanations
- Add examples or tutorials
- Improve code comments

## Development Setup

```bash
# Clone the repository
git clone https://github.com/4The-Architect7/UFIPC.git
cd UFIPC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure .env
cp .env.example .env
# Add your API keys to .env

# Run tests
python ufipc.py
```

## Pull Request Process

1. **Fork** the repository
2. **Create a branch** for your feature: `git checkout -b feature/your-feature-name`
3. **Make your changes** with clear, descriptive commits
4. **Test thoroughly** - ensure existing functionality isn't broken
5. **Update documentation** as needed
6. **Submit PR** with:
   - Clear description of changes
   - Reference to any related issues
   - Test results if applicable

## Code Style

- Follow PEP 8 guidelines
- Use descriptive variable names
- Add docstrings for functions
- Keep functions focused and modular
- Comment complex logic

## Testing Requirements

- All new models must be tested with n≥3 runs
- Report API success rates
- Include IPC scores with confidence intervals
- Compare against baseline models

## Questions?

- **Technical questions:** Open a GitHub Discussion
- **Bug reports:** Open a GitHub Issue  
- **Security issues:** Email josh.47.contreras@gmail.com directly
- **General inquiries:** Email Josh.47.contreras@gmail.com

## License

By contributing, you agree that your contributions will be licensed under the MIT License for research and educational use, consistent with the project's patent-pending status.

---

**Copyright (c) 2025 Joshua Contreras / Aletheia Cognitive Technologies**

Patent Pending - US Provisional Application No. 63/904,588
