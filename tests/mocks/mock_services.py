"""
Mock services for testing
"""

import asyncio
import json
import random
from typing import Dict, Any, List, Optional
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime


class MockLLMService:
    """Mock LLM service for testing."""
    
    def __init__(self, response_delay: float = 0.1, failure_rate: float = 0.0):
        self.response_delay = response_delay
        self.failure_rate = failure_rate
        self.call_count = 0
        self.call_history = []
        
    async def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a mock LLM response."""
        self.call_count += 1
        self.call_history.append({
            "prompt": prompt[:100] + "..." if len(prompt) > 100 else prompt,
            "kwargs": kwargs,
            "timestamp": datetime.now().isoformat()
        })
        
        # Simulate network delay
        await asyncio.sleep(self.response_delay)
        
        # Simulate random failures
        if random.random() < self.failure_rate:
            raise Exception("Mock LLM service failure")
        
        # Generate response based on prompt content
        if "analyze" in prompt.lower() and "problem" in prompt.lower():
            return self._generate_analysis_response(prompt)
        elif "context" in prompt.lower() and "collect" in prompt.lower():
            return self._generate_context_response(prompt)
        elif "requirements" in prompt.lower():
            return self._generate_requirements_response(prompt)
        elif "solution" in prompt.lower() and "design" in prompt.lower():
            return self._generate_solution_response(prompt)
        elif "guide" in prompt.lower() or "implementation" in prompt.lower():
            return self._generate_guide_response(prompt)
        else:
            return self._generate_generic_response(prompt)
    
    def _generate_analysis_response(self, prompt: str) -> Dict[str, Any]:
        """Generate analysis response based on prompt."""
        # Determine solution category from prompt
        if "document" in prompt.lower() and "search" in prompt.lower():
            category = "RAG"
            complexity = "high"
        elif "classify" in prompt.lower() or "machine learning" in prompt.lower():
            category = "ML_CLASSIFICATION"
            complexity = "medium"
        else:
            category = "SIMPLE_AUTOMATION"
            complexity = "low"
        
        return {
            "content": json.dumps({
                "problem_type": "automation" if category == "SIMPLE_AUTOMATION" else "knowledge_management",
                "complexity": complexity,
                "solution_category": category,
                "key_components": self._get_components_for_category(category),
                "estimated_effort": self._get_effort_estimate(complexity)
            }),
            "usage": {"tokens": random.randint(80, 150)}
        }
    
    def _generate_context_response(self, prompt: str) -> Dict[str, Any]:
        """Generate context collection response."""
        # Check if context seems sufficient
        if len(prompt) > 500 and "technical_level" in prompt:
            return {
                "content": json.dumps({
                    "context_sufficient": True,
                    "analysis": "Sufficient context provided for solution design"
                }),
                "usage": {"tokens": random.randint(50, 80)}
            }
        else:
            return {
                "content": json.dumps({
                    "questions": [
                        "What is your technical background?",
                        "What tools do you currently use?",
                        "What is your timeline for implementation?",
                        "Do you have any specific requirements or constraints?"
                    ],
                    "context_required": True,
                    "priority": "high"
                }),
                "usage": {"tokens": random.randint(60, 100)}
            }
    
    def _generate_requirements_response(self, prompt: str) -> Dict[str, Any]:
        """Generate requirements document response."""
        requirements_doc = """# Software Requirements Specification

## Project Overview
Automated solution for the described problem domain.

## Functional Requirements

### FR-001: Core Processing
- System shall process input data according to specified rules
- System shall validate data integrity and format
- System shall handle various input formats

### FR-002: Output Generation
- System shall generate required outputs
- System shall format outputs according to specifications
- System shall provide export capabilities

### FR-003: Error Handling
- System shall log all processing errors
- System shall provide meaningful error messages
- System shall recover gracefully from errors

## Non-Functional Requirements

### NFR-001: Performance
- System shall process data within acceptable timeframes
- System shall handle expected data volumes
- System shall maintain responsive user interface

### NFR-002: Reliability
- System shall maintain 99% uptime
- System shall include backup and recovery mechanisms
- System shall validate all critical operations

### NFR-003: Usability
- System shall provide intuitive user interface
- System shall include comprehensive documentation
- System shall support user training materials

## Acceptance Criteria
- All functional requirements implemented and tested
- Performance meets specified benchmarks
- User acceptance testing completed successfully
- Documentation and training materials delivered"""

        return {
            "content": requirements_doc,
            "usage": {"tokens": random.randint(200, 350)}
        }
    
    def _generate_solution_response(self, prompt: str) -> Dict[str, Any]:
        """Generate solution design response."""
        # Determine technology stack based on context
        if "rag" in prompt.lower() or "document" in prompt.lower():
            tech_stack = {
                "language": "python",
                "framework": "fastapi",
                "libraries": ["langchain", "chromadb", "openai"],
                "vector_db": "chromadb",
                "deployment": "docker"
            }
            solution_type = "RAG"
        elif "classification" in prompt.lower() or "ml" in prompt.lower():
            tech_stack = {
                "language": "python",
                "framework": "fastapi",
                "libraries": ["scikit-learn", "pandas", "numpy"],
                "ml_framework": "scikit_learn",
                "deployment": "docker"
            }
            solution_type = "ML_CLASSIFICATION"
        else:
            tech_stack = {
                "language": "python",
                "framework": "fastapi",
                "libraries": ["pandas", "openpyxl", "requests"],
                "deployment": "docker"
            }
            solution_type = "SIMPLE_AUTOMATION"
        
        return {
            "content": json.dumps({
                "solution_type": solution_type,
                "technology_stack": tech_stack,
                "architecture_pattern": "microservice",
                "estimated_complexity": "medium",
                "development_phases": [
                    "setup_environment",
                    "core_implementation",
                    "testing_validation",
                    "deployment"
                ]
            }),
            "usage": {"tokens": random.randint(150, 250)}
        }
    
    def _generate_guide_response(self, prompt: str) -> Dict[str, Any]:
        """Generate implementation guide response."""
        guide_content = """# Implementation Guide

## Prerequisites
- Development environment setup
- Required libraries and dependencies
- Basic understanding of the chosen technology stack

## Step-by-Step Implementation

### Step 1: Environment Setup
```bash
# Create virtual environment
python -m venv project_env
source project_env/bin/activate  # Linux/Mac
project_env\\Scripts\\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Core Implementation
```python
# Main application structure
from fastapi import FastAPI
import pandas as pd

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API is running"}

@app.post("/process")
def process_data(data: dict):
    # Core processing logic here
    result = {"status": "processed", "data": data}
    return result
```

### Step 3: Testing
```python
# Unit tests
import pytest

def test_core_functionality():
    # Test implementation
    assert True

def test_error_handling():
    # Test error cases
    assert True
```

### Step 4: Deployment
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Best Practices
- Follow code quality standards
- Implement comprehensive error handling
- Include logging and monitoring
- Document all public interfaces

## Next Steps
1. Complete implementation according to requirements
2. Conduct thorough testing
3. Deploy to target environment
4. Monitor and maintain system"""

        return {
            "content": guide_content,
            "usage": {"tokens": random.randint(300, 500)}
        }
    
    def _generate_generic_response(self, prompt: str) -> Dict[str, Any]:
        """Generate generic response for unrecognized prompts."""
        return {
            "content": f"Mock response for prompt: {prompt[:50]}...",
            "usage": {"tokens": random.randint(50, 100)}
        }
    
    def _get_components_for_category(self, category: str) -> List[str]:
        """Get key components for solution category."""
        components_map = {
            "SIMPLE_AUTOMATION": ["data_processing", "file_handling", "report_generation"],
            "RAG": ["document_embedding", "vector_search", "llm_generation"],
            "ML_CLASSIFICATION": ["data_preprocessing", "model_training", "prediction_api"],
            "COMPLEX_AUTOMATION": ["workflow_orchestration", "data_pipeline", "monitoring"]
        }
        return components_map.get(category, ["generic_component"])
    
    def _get_effort_estimate(self, complexity: str) -> str:
        """Get effort estimate based on complexity."""
        estimates = {
            "low": "1-3 days",
            "medium": "1-2 weeks", 
            "high": "2-4 weeks"
        }
        return estimates.get(complexity, "1 week")
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0
        self.call_history = []


class MockRAGService:
    """Mock RAG service for testing."""
    
    def __init__(self, response_delay: float = 0.05, failure_rate: float = 0.0):
        self.response_delay = response_delay
        self.failure_rate = failure_rate
        self.call_count = 0
        self.document_store = {
            "automation": [
                "Best practices for automation include proper error handling and logging",
                "Automation systems should be designed with scalability in mind",
                "Testing is crucial for automation reliability"
            ],
            "python": [
                "Python is excellent for automation tasks with rich library ecosystem",
                "FastAPI provides modern web API development with automatic documentation",
                "pandas is the go-to library for data manipulation in Python"
            ],
            "deployment": [
                "Docker containerization ensures consistent deployment environments",
                "CI/CD pipelines automate testing and deployment processes",
                "Cloud platforms provide scalable hosting solutions"
            ]
        }
    
    async def retrieve_context(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Retrieve relevant context for a query."""
        self.call_count += 1
        
        # Simulate network delay
        await asyncio.sleep(self.response_delay)
        
        # Simulate random failures
        if random.random() < self.failure_rate:
            raise Exception("Mock RAG service failure")
        
        # Simple keyword matching for mock
        relevant_docs = []
        relevance_scores = []
        
        for category, docs in self.document_store.items():
            if category.lower() in query.lower():
                relevant_docs.extend(docs)
                relevance_scores.extend([0.9, 0.8, 0.7][:len(docs)])
        
        # If no specific match, return general docs
        if not relevant_docs:
            relevant_docs = self.document_store["automation"]
            relevance_scores = [0.6, 0.5, 0.4]
        
        # Limit to top_k results
        if len(relevant_docs) > top_k:
            relevant_docs = relevant_docs[:top_k]
            relevance_scores = relevance_scores[:top_k]
        
        return {
            "documents": relevant_docs,
            "relevance_scores": relevance_scores,
            "total_documents": len(self.document_store),
            "query_time": self.response_delay
        }
    
    def add_documents(self, category: str, documents: List[str]):
        """Add documents to the mock store."""
        if category not in self.document_store:
            self.document_store[category] = []
        self.document_store[category].extend(documents)
    
    def reset(self):
        """Reset mock state."""
        self.call_count = 0


class MockDatabase:
    """Mock database for testing."""
    
    def __init__(self, failure_rate: float = 0.0):
        self.failure_rate = failure_rate
        self.data = {}
        self.call_count = 0
        
    async def save(self, key: str, value: Any) -> bool:
        """Save data to mock database."""
        self.call_count += 1
        
        # Simulate random failures
        if random.random() < self.failure_rate:
            raise Exception("Mock database save failure")
        
        self.data[key] = value
        return True
    
    async def load(self, key: str) -> Optional[Any]:
        """Load data from mock database."""
        self.call_count += 1
        
        # Simulate random failures
        if random.random() < self.failure_rate:
            raise Exception("Mock database load failure")
        
        return self.data.get(key)
    
    async def delete(self, key: str) -> bool:
        """Delete data from mock database."""
        self.call_count += 1
        
        if key in self.data:
            del self.data[key]
            return True
        return False
    
    async def list_keys(self, prefix: str = "") -> List[str]:
        """List keys with optional prefix filter."""
        self.call_count += 1
        
        if prefix:
            return [k for k in self.data.keys() if k.startswith(prefix)]
        return list(self.data.keys())
    
    def reset(self):
        """Reset mock database."""
        self.data = {}
        self.call_count = 0


class MockWorkflowManager:
    """Mock workflow manager for testing."""
    
    def __init__(self):
        self.active_workflows = {}
        self.completed_workflows = {}
        self.call_count = 0
    
    async def start_workflow(self, thread_id: str, initial_state: Dict[str, Any]) -> bool:
        """Start a new workflow."""
        self.call_count += 1
        self.active_workflows[thread_id] = {
            "state": initial_state,
            "status": "running",
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }
        return True
    
    async def get_workflow_status(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow status."""
        self.call_count += 1
        
        if thread_id in self.active_workflows:
            workflow = self.active_workflows[thread_id]
            return {
                "thread_id": thread_id,
                "status": workflow["status"],
                "current_step": workflow["state"].get("current_step"),
                "progress_percentage": workflow["state"].get("progress_percentage", 0),
                "requires_input": workflow["state"].get("requires_user_input", False),
                "created_at": workflow["created_at"].isoformat(),
                "updated_at": workflow["updated_at"].isoformat()
            }
        
        if thread_id in self.completed_workflows:
            workflow = self.completed_workflows[thread_id]
            return {
                "thread_id": thread_id,
                "status": "completed",
                "progress_percentage": 100,
                "results": workflow.get("results", {}),
                "created_at": workflow["created_at"].isoformat(),
                "updated_at": workflow["updated_at"].isoformat()
            }
        
        return None
    
    async def update_workflow(self, thread_id: str, new_state: Dict[str, Any]) -> bool:
        """Update workflow state."""
        self.call_count += 1
        
        if thread_id in self.active_workflows:
            self.active_workflows[thread_id]["state"].update(new_state)
            self.active_workflows[thread_id]["updated_at"] = datetime.now()
            
            # Check if workflow is completed
            if new_state.get("current_status") == "complete":
                self.completed_workflows[thread_id] = self.active_workflows.pop(thread_id)
            
            return True
        return False
    
    async def resume_workflow(self, thread_id: str, user_input: Dict[str, Any]) -> bool:
        """Resume workflow with user input."""
        self.call_count += 1
        
        if thread_id in self.active_workflows:
            workflow = self.active_workflows[thread_id]
            workflow["state"]["requires_user_input"] = False
            workflow["state"]["user_input"] = user_input
            workflow["updated_at"] = datetime.now()
            return True
        return False
    
    def list_active_workflows(self) -> List[str]:
        """List all active workflow IDs."""
        return list(self.active_workflows.keys())
    
    def reset(self):
        """Reset mock workflow manager."""
        self.active_workflows = {}
        self.completed_workflows = {}
        self.call_count = 0


class MockAPIClient:
    """Mock API client for testing frontend integration."""
    
    def __init__(self, base_url: str = "http://localhost:8000", delay: float = 0.1):
        self.base_url = base_url
        self.delay = delay
        self.call_count = 0
        self.call_history = []
        
    async def post(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Mock POST request."""
        self.call_count += 1
        self.call_history.append({"method": "POST", "endpoint": endpoint, "data": data})
        
        await asyncio.sleep(self.delay)
        
        if "/start-analysis" in endpoint:
            return {
                "status": "started",
                "thread_id": f"mock_thread_{self.call_count}",
                "message": "Analysis started successfully"
            }
        elif "/resume/" in endpoint:
            return {
                "status": "resumed",
                "message": "Workflow resumed successfully"
            }
        else:
            return {"status": "success", "message": "Mock response"}
    
    async def get(self, endpoint: str) -> Dict[str, Any]:
        """Mock GET request."""
        self.call_count += 1
        self.call_history.append({"method": "GET", "endpoint": endpoint})
        
        await asyncio.sleep(self.delay)
        
        if "/health" in endpoint:
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        elif "/status/" in endpoint:
            # Simulate different status responses based on call count
            responses = [
                {
                    "status": "running",
                    "current_step": "analyze_problem",
                    "progress_percentage": 25,
                    "requires_input": False
                },
                {
                    "status": "awaiting_input",
                    "current_step": "collect_context",
                    "progress_percentage": 40,
                    "requires_input": True,
                    "questions": ["What tools do you use?"]
                },
                {
                    "status": "completed",
                    "current_step": "create_guide",
                    "progress_percentage": 100,
                    "requires_input": False,
                    "results": {
                        "requirements_doc": "# Requirements\n...",
                        "implementation_guide": "# Guide\n..."
                    }
                }
            ]
            return responses[(self.call_count - 1) % len(responses)]
        else:
            return {"status": "success", "data": "Mock data"}
    
    def reset(self):
        """Reset mock API client."""
        self.call_count = 0
        self.call_history = []


# Factory functions for easy mock creation
def create_mock_llm_service(**kwargs) -> MockLLMService:
    """Create a mock LLM service with optional configuration."""
    return MockLLMService(**kwargs)

def create_mock_rag_service(**kwargs) -> MockRAGService:
    """Create a mock RAG service with optional configuration."""
    return MockRAGService(**kwargs)

def create_mock_database(**kwargs) -> MockDatabase:
    """Create a mock database with optional configuration."""
    return MockDatabase(**kwargs)

def create_mock_workflow_manager() -> MockWorkflowManager:
    """Create a mock workflow manager."""
    return MockWorkflowManager()

def create_mock_api_client(**kwargs) -> MockAPIClient:
    """Create a mock API client with optional configuration."""
    return MockAPIClient(**kwargs)

# Utility functions for test setup
def setup_successful_mocks():
    """Set up mocks that always succeed."""
    return {
        "llm_service": create_mock_llm_service(failure_rate=0.0),
        "rag_service": create_mock_rag_service(failure_rate=0.0),
        "database": create_mock_database(failure_rate=0.0),
        "workflow_manager": create_mock_workflow_manager(),
        "api_client": create_mock_api_client()
    }

def setup_failing_mocks(failure_rate: float = 0.3):
    """Set up mocks with specified failure rate."""
    return {
        "llm_service": create_mock_llm_service(failure_rate=failure_rate),
        "rag_service": create_mock_rag_service(failure_rate=failure_rate),
        "database": create_mock_database(failure_rate=failure_rate),
        "workflow_manager": create_mock_workflow_manager(),
        "api_client": create_mock_api_client()
    }

def setup_slow_mocks(delay: float = 1.0):
    """Set up mocks with specified delays for performance testing."""
    return {
        "llm_service": create_mock_llm_service(response_delay=delay),
        "rag_service": create_mock_rag_service(response_delay=delay),
        "database": create_mock_database(),
        "workflow_manager": create_mock_workflow_manager(),
        "api_client": create_mock_api_client(delay=delay)
    }