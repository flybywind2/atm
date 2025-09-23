# ATM System Testing Guide

This directory contains comprehensive testing for the AI-based problem-solving copilot system (ATM). The testing framework covers all aspects of the system including unit tests, integration tests, system tests, performance tests, and quality validation.

## Test Structure

```
tests/
├── backend/
│   ├── unit/                    # Unit tests for backend components
│   │   ├── agents/             # LangGraph agent tests
│   │   ├── api/                # API endpoint tests
│   │   ├── workflows/          # Workflow state tests
│   │   └── database/           # Database/checkpointer tests
│   ├── integration/            # Integration tests
│   │   ├── test_workflow_integration.py
│   │   ├── test_hitl_flow.py
│   │   └── test_checkpointer_recovery.py
│   └── performance/            # Performance tests
│       └── test_load_testing.py
├── frontend/
│   ├── unit/                   # Frontend component tests
│   │   └── components/
│   └── integration/            # Frontend integration tests
├── system/                     # End-to-end system tests
├── quality/                    # Output quality validation
├── fixtures/                   # Test data and fixtures
├── mocks/                      # Mock services and utilities
├── reports/                    # Test reports and coverage
├── conftest.py                 # Global pytest configuration
├── run_tests.py               # Comprehensive test runner
└── README.md                  # This file
```

## Test Categories

### 1. Unit Tests
Individual component testing with isolated dependencies.

- **Agent Tests**: Test each LangGraph agent (analyzer, context_collector, etc.)
- **API Tests**: Test FastAPI endpoints with mock dependencies
- **Workflow Tests**: Test workflow state management
- **Database Tests**: Test SQLite checkpointer functionality

**Run unit tests:**
```bash
python tests/run_tests.py --suite unit
```

### 2. Integration Tests
Test component interactions and data flow.

- **Workflow Integration**: Complete workflow execution
- **HITL Flow**: Human-in-the-loop interruption and resumption
- **Checkpointer Recovery**: State persistence and recovery
- **API Integration**: End-to-end API workflows

**Run integration tests:**
```bash
python tests/run_tests.py --suite integration
```

### 3. System Tests
End-to-end testing of complete user journeys.

- **Complete Workflows**: Full automation scenarios
- **Error Handling**: System-level error recovery
- **Concurrent Users**: Multi-user scenario testing
- **Data Persistence**: Cross-session data integrity

**Run system tests:**
```bash
python tests/run_tests.py --suite system
```

### 4. Performance Tests
Load testing and performance validation.

- **API Performance**: Endpoint response times under load
- **Workflow Performance**: Complex workflow execution speed
- **Database Performance**: Concurrent checkpoint operations
- **Memory Usage**: Resource consumption testing

**Run performance tests:**
```bash
python tests/run_tests.py --suite performance
```

### 5. Quality Tests
Output quality validation and content analysis.

- **Requirements Validation**: SRS document structure and completeness
- **Implementation Guide Validation**: Code examples and instructions
- **User Journey Validation**: Timeline and benefit analysis
- **Solution Classification**: Accuracy of problem categorization

**Run quality tests:**
```bash
python tests/run_tests.py --suite quality
```

## Frontend Tests

JavaScript tests for frontend components using Jest.

### Component Tests
- **ProblemInput**: Form validation and submission
- **ProgressTracker**: Status updates and polling
- **DocumentViewer**: Markdown rendering and navigation
- **ContextCollector**: Dynamic question handling

### Integration Tests
- **Complete User Flow**: End-to-end frontend workflows
- **API Integration**: Frontend-backend communication
- **Error Handling**: User-friendly error management
- **State Management**: Application state consistency

**Run frontend tests:**
```bash
python tests/run_tests.py --frontend-only
```

## Running Tests

### Quick Start
```bash
# Run all tests
python tests/run_tests.py

# Run specific test suite
python tests/run_tests.py --suite unit

# Run with coverage
python tests/run_tests.py --suite integration

# Run backend only
python tests/run_tests.py --backend-only

# Run with verbose output
python tests/run_tests.py --verbose
```

### Using pytest directly
```bash
# Unit tests with coverage
pytest tests/backend/unit --cov=backend/app --cov-report=html

# Integration tests
pytest tests/backend/integration -v

# Specific test file
pytest tests/backend/unit/agents/test_analyzer.py -v

# With markers
pytest -m "unit and not slow" -v
```

### Using npm for frontend
```bash
cd frontend
npm test                    # Run all frontend tests
npm run test:coverage      # Run with coverage
npm run test:watch         # Watch mode
```

## Test Configuration

### Pytest Configuration
Configuration is in `pytest.ini`:
- Test discovery patterns
- Marker definitions
- Coverage settings
- Output formatting

### Markers
Tests are organized using pytest markers:
- `unit`: Unit tests
- `integration`: Integration tests
- `system`: System tests
- `performance`: Performance tests
- `quality`: Quality tests
- `slow`: Long-running tests
- `hitl`: Human-in-the-loop tests

### Environment Variables
Set these for testing:
```bash
export ENVIRONMENT=test
export DATABASE_URL=sqlite:///test.db
export LLM_SERVICE_URL=http://localhost:8001
export RAG_SERVICE_URL=http://localhost:8002
```

## Mock Services

The testing framework includes comprehensive mocks:

### MockLLMService
Simulates LLM API responses with configurable:
- Response delays
- Failure rates
- Context-aware responses

### MockRAGService
Simulates RAG service with:
- Document retrieval
- Relevance scoring
- Query processing

### MockDatabase
In-memory database for isolated testing:
- State persistence
- Concurrent access
- Error simulation

### MockWorkflowManager
Workflow state management:
- Session tracking
- State transitions
- Recovery scenarios

## Test Data

### Fixtures
Pre-built test data in `fixtures/sample_data.py`:
- Sample problems for each solution category
- Workflow states at different stages
- Expected LLM responses
- Error scenarios

### Dynamic Generation
Utilities for generating:
- Large workflow states (memory testing)
- Conversation histories
- Performance test data

## Coverage Requirements

Minimum coverage thresholds:
- **Overall**: 70%
- **Critical paths**: 90%
- **API endpoints**: 100%
- **Agent functions**: 85%

Coverage reports are generated in:
- `tests/reports/coverage/` (HTML)
- `tests/reports/coverage.json` (JSON)

## CI/CD Integration

GitHub Actions workflow (`.github/workflows/ci.yml`):
1. **Backend Unit Tests**: Fast component testing
2. **Backend Integration Tests**: Workflow testing
3. **Frontend Tests**: JavaScript component testing
4. **E2E Tests**: Complete system validation
5. **Performance Tests**: Load testing (main branch only)
6. **Quality Tests**: Output validation
7. **Code Quality**: Linting and security

### Workflow Triggers
- Push to `main` or `develop`
- Pull requests to `main`
- Manual workflow dispatch

### Artifacts
Test results and reports are saved as artifacts:
- JUnit XML reports
- Coverage reports
- Performance metrics
- Security scan results

## Quality Gates

Tests must pass these gates:
- All unit tests pass
- Integration tests pass
- Coverage above threshold
- No security vulnerabilities
- Code quality checks pass
- Performance within limits

## Debugging Tests

### Local Debugging
```bash
# Run specific test with pdb
pytest tests/backend/unit/agents/test_analyzer.py::test_analyze_simple_automation -s --pdb

# Run with logging
pytest tests/ -v -s --log-cli-level=DEBUG

# Run failed tests only
pytest --lf -v
```

### Common Issues
1. **Import Errors**: Check Python path and virtual environment
2. **Database Locks**: Use temporary databases for tests
3. **Async Issues**: Ensure proper event loop handling
4. **Mock Failures**: Verify mock configurations

### Test Data Cleanup
Tests automatically clean up:
- Temporary databases
- Mock service state
- Test artifacts

## Contributing

### Adding New Tests
1. Choose appropriate test category
2. Use existing fixtures and mocks
3. Follow naming conventions
4. Add appropriate markers
5. Include docstrings
6. Update coverage thresholds if needed

### Test Naming
- `test_<functionality>_<scenario>`
- `test_<component>_<action>_<expected_result>`
- Clear descriptive names

### Best Practices
- One assertion per test (when possible)
- Arrange-Act-Assert pattern
- Independent tests (no dependencies)
- Clear test data setup
- Proper cleanup
- Meaningful error messages

## Monitoring and Maintenance

### Regular Tasks
- Review and update test data
- Monitor coverage trends
- Update mock responses
- Performance baseline updates
- Security scan reviews

### Metrics to Track
- Test execution time
- Coverage percentage
- Failure rates
- Performance benchmarks
- Quality scores

For more information, see the main project documentation or contact the development team.