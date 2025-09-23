"""
Global pytest configuration and fixtures
"""

import asyncio
import pytest
import tempfile
import os
from typing import Dict, Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from fastapi.testclient import TestClient


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_db_path():
    """Create a temporary database file for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        temp_path = f.name
    yield temp_path
    if os.path.exists(temp_path):
        os.unlink(temp_path)


@pytest.fixture
def mock_llm_service():
    """Mock LLM service for testing."""
    mock = AsyncMock()
    mock.generate_response.return_value = {
        "content": "Mocked LLM response",
        "usage": {"tokens": 100}
    }
    return mock


@pytest.fixture
def mock_rag_service():
    """Mock RAG service for testing."""
    mock = AsyncMock()
    mock.retrieve_context.return_value = {
        "documents": ["Sample document 1", "Sample document 2"],
        "relevance_scores": [0.8, 0.7]
    }
    return mock


@pytest.fixture
def sample_workflow_state():
    """Sample workflow state for testing."""
    return {
        "problem_description": "Create a simple data processing automation",
        "conversation_history": [],
        "context_data": {},
        "current_step": "start",
        "current_status": "initialized",
        "context_complete": False,
        "requires_user_input": False,
        "retry_count": 0,
        "problem_analysis": None,
        "requirements_doc": None,
        "solution_type": None,
        "technology_stack": None,
        "implementation_plan": None,
        "implementation_guide": None,
        "user_journey": None,
        "progress_percentage": 0
    }


@pytest.fixture
def sample_analysis_request():
    """Sample analysis request for testing."""
    return {
        "problem_description": "I need to automate daily report generation from Excel data",
        "user_context": {
            "technical_level": "beginner",
            "environment": "Windows 10",
            "tools": ["Excel", "Python"],
            "experience": "Basic Python knowledge"
        }
    }


@pytest.fixture
def sample_resume_request():
    """Sample resume request for testing."""
    return {
        "user_input": "I work with financial data, need automated validation, and generate reports for management team.",
        "context_data": {
            "data_type": "financial",
            "validation_requirements": "automated",
            "audience": "management"
        }
    }


@pytest.fixture
def test_client():
    """FastAPI test client."""
    from backend.app.main import app
    return TestClient(app)


@pytest.fixture
def async_client():
    """Async HTTP client for testing."""
    return httpx.AsyncClient(timeout=30.0)


@pytest.fixture
def mock_workflow_state_complete():
    """Complete workflow state for end-to-end testing."""
    return {
        "problem_description": "Automate invoice processing workflow",
        "conversation_history": [
            {"role": "user", "content": "I need to automate invoice processing"},
            {"role": "assistant", "content": "I'll help you analyze this automation task"}
        ],
        "context_data": {
            "department": "finance",
            "volume": "50_invoices_per_day",
            "current_tools": ["Excel", "Email"],
            "pain_points": ["manual_data_entry", "validation_errors"]
        },
        "current_step": "create_guide",
        "current_status": "complete",
        "context_complete": True,
        "requires_user_input": False,
        "retry_count": 0,
        "problem_analysis": {
            "type": "automation",
            "complexity": "medium",
            "solution_category": "SIMPLE_AUTOMATION"
        },
        "requirements_doc": "## Functional Requirements\n- Process invoices automatically\n- Validate data integrity\n- Generate reports",
        "solution_type": "SIMPLE_AUTOMATION",
        "technology_stack": {
            "language": "python",
            "libraries": ["pandas", "openpyxl", "fastapi"],
            "frameworks": ["FastAPI"]
        },
        "implementation_plan": "## Implementation Plan\n1. Data extraction module\n2. Validation engine\n3. Report generator",
        "implementation_guide": "## Implementation Guide\n### Step 1: Setup\n```python\npip install pandas fastapi\n```",
        "user_journey": "## User Journey\n### Current State\n- Manual processing\n### Future State\n- Automated workflow",
        "progress_percentage": 100
    }


class MockAgent:
    """Mock agent class for testing agent functions."""
    
    def __init__(self, response_data: Dict[str, Any]):
        self.response_data = response_data
    
    async def __call__(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {**state, **self.response_data}


@pytest.fixture
def mock_agents():
    """Mock agents for workflow testing."""
    return {
        "analyzer": MockAgent({
            "current_step": "analyze_problem",
            "problem_analysis": {"type": "automation", "complexity": "low"},
            "current_status": "analyzing"
        }),
        "context_collector": MockAgent({
            "current_step": "collect_context",
            "context_data": {"user_needs": "basic automation"},
            "context_complete": True,
            "current_status": "context_collected"
        }),
        "requirements_generator": MockAgent({
            "current_step": "generate_requirements",
            "requirements_doc": "## Requirements\n- Automate data processing",
            "current_status": "requirements_generated"
        }),
        "solution_designer": MockAgent({
            "current_step": "design_solution",
            "solution_type": "SIMPLE_AUTOMATION",
            "technology_stack": {"language": "python"},
            "current_status": "solution_designed"
        }),
        "guide_creator": MockAgent({
            "current_step": "create_guide",
            "implementation_guide": "## Guide\nStep-by-step implementation",
            "current_status": "complete"
        })
    }


@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing."""
    return {
        "llm_timeout": {"error": "LLM service timeout", "retry_after": 5},
        "rag_unavailable": {"error": "RAG service unavailable", "status": 503},
        "invalid_input": {"error": "Invalid problem description", "status": 400},
        "database_error": {"error": "Database connection failed", "status": 500},
        "workflow_corruption": {"error": "Workflow state corrupted", "status": 500}
    }


@pytest.fixture
def performance_config():
    """Performance testing configuration."""
    return {
        "concurrent_users": 10,
        "test_duration": 30,  # seconds
        "request_timeout": 10,  # seconds
        "success_rate_threshold": 0.95,
        "response_time_percentile": 0.95,
        "max_response_time": 5.0  # seconds
    }