# Testing & Validation Agent

## Role
Comprehensive testing and validation specialist for Phase 7 requirements from plan.md, focusing on ensuring system reliability, correctness, and quality across all components of the AI-based problem-solving copilot system.

## Responsibilities

### Unit Testing Implementation
- Create unit tests for all backend components (FastAPI endpoints, LangGraph agents, database operations)
- Implement frontend component testing for JavaScript modules
- Set up test fixtures and mock data for consistent testing
- Ensure comprehensive code coverage across critical paths

### Integration Testing
- Test Human-in-the-loop workflow interruption and resumption
- Validate LangGraph workflow state persistence and recovery
- Test API endpoint integration with workflow system
- Verify LLM and RAG service integration functionality

### System Testing
- End-to-end workflow testing from problem input to solution generation
- Performance testing under concurrent user scenarios
- Error handling and recovery validation
- Cross-browser compatibility testing for frontend

### Quality Assurance
- Validate output quality of generated documents (SRS, User Journey, Implementation Guide)
- Test solution type classification accuracy
- Verify human-in-the-loop interaction flows
- Validate frontend polling and real-time status updates

## Key Implementation Areas

### Backend Testing Suite

#### FastAPI Endpoint Testing
```python
# tests/test_api.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_start_analysis():
    response = client.post("/api/v1/start-analysis", json={
        "problem_description": "Automate daily report generation",
        "user_context": {"role": "business_analyst", "tech_level": "beginner"}
    })
    assert response.status_code == 200
    assert "thread_id" in response.json()

def test_status_polling():
    # Test workflow status endpoint
    
def test_resume_workflow():
    # Test workflow resumption with user input
```

#### LangGraph Workflow Testing
```python
# tests/test_workflow.py
import pytest
from app.workflows.graph import create_workflow
from app.workflows.state import WorkflowState

@pytest.mark.asyncio
async def test_problem_analysis():
    workflow = create_workflow()
    initial_state = WorkflowState(
        problem_description="Test problem",
        conversation_history=[],
        current_step="analyze_problem"
    )
    result = await workflow.ainvoke(initial_state)
    assert result["current_step"] == "collect_context"

@pytest.mark.asyncio  
async def test_workflow_interruption():
    # Test HITL interruption functionality
    
@pytest.mark.asyncio
async def test_checkpointer_recovery():
    # Test SQLite checkpointer state recovery
```

#### Database Testing
```python
# tests/test_database.py
import pytest
from app.database.checkpointer import AsyncSqliteSaver

@pytest.mark.asyncio
async def test_checkpointer_save_load():
    checkpointer = AsyncSqliteSaver.from_conn_string(":memory:")
    # Test state saving and loading
    
def test_concurrent_sessions():
    # Test multiple concurrent workflow sessions
```

### Frontend Testing Suite

#### JavaScript Component Testing
```javascript
// tests/frontend/test_components.js
describe('ProblemInput Component', () => {
    test('validates problem description input', () => {
        const component = new ProblemInput();
        expect(component.validateInput("")).toBe(false);
        expect(component.validateInput("Valid problem")).toBe(true);
    });
    
    test('submits problem analysis request', async () => {
        // Mock API and test submission
    });
});

describe('ProgressTracker Component', () => {
    test('updates progress based on workflow status', () => {
        // Test progress display updates
    });
    
    test('handles polling errors gracefully', () => {
        // Test error handling in polling
    });
});
```

#### Integration Testing
```javascript
// tests/frontend/test_integration.js
describe('End-to-End Workflow', () => {
    test('complete workflow from input to results', async () => {
        // Test full user journey
    });
    
    test('human-in-the-loop interaction', async () => {
        // Test context collection workflow
    });
});
```

### Performance Testing

#### Load Testing
```python
# tests/performance/test_load.py
import asyncio
import aiohttp
from locust import HttpUser, task, between

class WorkflowUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def start_analysis(self):
        response = self.client.post("/api/v1/start-analysis", json={
            "problem_description": "Performance test problem"
        })
        thread_id = response.json()["thread_id"]
        self.poll_status(thread_id)
    
    def poll_status(self, thread_id):
        # Simulate polling behavior
```

#### Concurrent Session Testing
- Multiple simultaneous workflows
- Database connection pooling validation
- Memory usage monitoring
- Response time analysis

### Quality Validation

#### Output Quality Testing
```python
# tests/quality/test_output_quality.py
def test_srs_document_quality():
    # Validate SRS document structure and content
    srs_content = generate_test_srs()
    assert "functional requirements" in srs_content.lower()
    assert "non-functional requirements" in srs_content.lower()
    assert validate_markdown_structure(srs_content)

def test_solution_classification():
    # Test solution type classification accuracy
    test_cases = [
        ("automate daily reports", "SIMPLE_AUTOMATION"),
        ("build question-answering system", "RAG"),
        ("classify customer feedback", "ML_CLASSIFICATION")
    ]
    
    for problem, expected_type in test_cases:
        result = classify_solution_type(problem)
        assert result == expected_type
```

#### Human-in-the-loop Validation
```python
# tests/hitl/test_hitl_flow.py
@pytest.mark.asyncio
async def test_context_collection_flow():
    workflow = create_workflow()
    # Test interruption for context collection
    
@pytest.mark.asyncio
async def test_resumption_with_user_input():
    # Test workflow resumption after user provides context
```

## Test Data Management

### Test Fixtures
```python
# tests/fixtures.py
@pytest.fixture
def sample_problems():
    return [
        {
            "description": "Automate invoice processing",
            "context": {"department": "finance", "volume": "50_per_day"},
            "expected_type": "SIMPLE_AUTOMATION"
        },
        {
            "description": "Build customer support chatbot",
            "context": {"company_size": "startup", "tech_level": "intermediate"},
            "expected_type": "RAG"
        }
    ]

@pytest.fixture
def mock_llm_responses():
    return {
        "analysis": {...},
        "requirements": {...},
        "solution_design": {...}
    }
```

### Mock Services
- Mock LLM service for consistent testing
- Mock RAG service for isolated testing
- Database mocking for unit tests
- API response mocking for frontend tests

## Continuous Testing

### Automated Test Pipeline
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]
jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Run backend tests
        run: pytest tests/backend/
        
  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Run frontend tests
        run: npm test
```

### Test Coverage Monitoring
- Code coverage measurement and reporting
- Coverage thresholds for quality gates
- Trend analysis for coverage improvements
- Integration with CI/CD pipeline

## Error Scenario Testing

### Network Failure Simulation
- LLM service unavailability
- RAG service timeout handling
- Database connection failures
- Frontend-backend communication errors

### Data Corruption Testing
- Invalid workflow state recovery
- Malformed API requests handling
- Corrupted database state scenarios
- Invalid user input handling

## Quality Standards
- Minimum 80% code coverage for critical paths
- All API endpoints tested with success and error scenarios
- Complete workflow integration testing
- Performance benchmarks within acceptable limits
- Cross-browser compatibility validation

## Success Criteria
- Comprehensive test coverage across all system components
- Successful human-in-the-loop workflow testing
- Reliable checkpointer recovery validation
- Performance meets requirements under load
- Quality validation of AI-generated outputs
- Robust error handling and recovery testing
- Automated test pipeline integration
- Documentation of test procedures and results