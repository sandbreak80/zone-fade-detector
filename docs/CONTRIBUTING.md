# Contributing to Zone Fade Detector

Thank you for your interest in contributing to the Zone Fade Detector project! This document provides guidelines and information for contributors.

## 🚀 Getting Started

### Prerequisites
- Docker and Docker Compose
- Python 3.11+
- Git
- Basic understanding of trading strategies

### Development Setup
1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/your-username/zone-fade-detector.git
   cd zone-fade-detector
   ```
3. Set up environment:
   ```bash
   cp .env.example .env
   # Edit .env with your API credentials
   ```
4. Build development environment:
   ```bash
   docker-compose up --build zone-fade-detector-dev
   ```

## 📋 Development Guidelines

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters and return values
- Write comprehensive docstrings for all public methods
- Keep functions focused and single-purpose

### Testing
- Write unit tests for all new functionality
- Add integration tests for complex workflows
- Ensure all tests pass before submitting PR
- Aim for >80% test coverage

### Documentation
- Update README.md for user-facing changes
- Add docstrings for new functions/classes
- Update CHANGELOG.md for significant changes
- Include examples for new features

## 🏗️ Architecture Guidelines

### Component Design
- Keep components loosely coupled
- Use dependency injection where appropriate
- Follow the existing pattern for new indicators
- Maintain separation between data, logic, and presentation

### Error Handling
- Use specific exception types
- Provide meaningful error messages
- Include fallback mechanisms where appropriate
- Log errors with appropriate levels

### Performance
- Consider memory usage for rolling windows
- Optimize for 30-second polling intervals
- Use efficient data structures
- Profile performance-critical code

## 🧪 Testing Guidelines

### Unit Tests
- Test individual components in isolation
- Mock external dependencies (APIs, file system)
- Test both success and failure scenarios
- Use descriptive test names

### Integration Tests
- Test component interactions
- Use real data when possible
- Test end-to-end workflows
- Verify alert generation

### Running Tests
```bash
# Unit tests
docker-compose run zone-fade-detector-test pytest tests/unit/

# Integration tests
docker-compose run zone-fade-detector-test pytest tests/integration/

# All tests
docker-compose run zone-fade-detector-test pytest
```

## 📝 Pull Request Process

### Before Submitting
1. Ensure all tests pass
2. Update documentation as needed
3. Add/update tests for new functionality
4. Check code style and formatting
5. Update CHANGELOG.md

### PR Description
- Clearly describe the changes
- Explain the motivation
- Reference any related issues
- Include screenshots for UI changes
- List any breaking changes

### Review Process
- All PRs require review
- Address reviewer feedback promptly
- Keep PRs focused and reasonably sized
- Respond to CI/CD failures

## 🎯 Areas for Contribution

### High Priority
- Rolling window management system
- Session state management
- Micro window analysis
- Parallel cross-symbol processing
- ES/NQ/RTY futures integration

### Medium Priority
- Additional indicators
- Performance optimizations
- Enhanced alert formatting
- More comprehensive backtesting
- Additional data sources

### Low Priority
- UI improvements
- Additional documentation
- Code refactoring
- Test coverage improvements

## 🐛 Bug Reports

### Before Reporting
1. Check existing issues
2. Verify the bug with latest code
3. Gather relevant information

### Bug Report Template
```markdown
**Bug Description**
A clear description of the bug.

**Steps to Reproduce**
1. Go to '...'
2. Click on '....'
3. See error

**Expected Behavior**
What you expected to happen.

**Actual Behavior**
What actually happened.

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Docker version: [e.g. 20.10.7]
- Python version: [e.g. 3.11.0]

**Additional Context**
Any other context about the problem.
```

## 💡 Feature Requests

### Before Requesting
1. Check existing feature requests
2. Consider the project's scope
3. Think about implementation complexity

### Feature Request Template
```markdown
**Feature Description**
A clear description of the feature.

**Use Case**
Why is this feature needed?

**Proposed Solution**
How should this feature work?

**Alternatives Considered**
Other solutions you've considered.

**Additional Context**
Any other context about the feature.
```

## 📞 Getting Help

- Create an issue for questions
- Join discussions in issues
- Check existing documentation
- Review code examples

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to Zone Fade Detector! 🚀