"""
End-to-end system tests for the complete ATM workflow
"""

import pytest
import asyncio
import httpx
import time
from unittest.mock import patch, AsyncMock
from typing import Dict, Any


class TestEndToEndWorkflow:
    """End-to-end system tests."""

    @pytest.mark.asyncio
    async def test_complete_user_journey_simple_automation(
        self, async_client, mock_llm_service, mock_rag_service
    ):
        """Test complete user journey for simple automation problem."""
        base_url = "http://localhost:8000"
        
        # Mock all LLM interactions
        llm_responses = [
            # Analyzer
            {
                "content": '{"problem_type": "automation", "complexity": "low", "solution_category": "SIMPLE_AUTOMATION"}',
                "usage": {"tokens": 100}
            },
            # Context collector (sufficient context)
            {
                "content": '{"context_sufficient": true, "analysis": "Sufficient context provided"}',
                "usage": {"tokens": 80}
            },
            # Requirements generator
            {
                "content": """# Software Requirements Specification
## Functional Requirements
- FR-001: Process Excel files automatically
- FR-002: Generate daily reports
## Non-Functional Requirements  
- NFR-001: Process files within 30 seconds
## Acceptance Criteria
- Successfully processes Excel files without errors""",
                "usage": {"tokens": 150}
            },
            # Solution designer
            {
                "content": """{"solution_type": "SIMPLE_AUTOMATION", "technology_stack": {"language": "python", "framework": "fastapi", "libraries": ["pandas", "openpyxl"]}, "estimated_complexity": "low"}""",
                "usage": {"tokens": 120}
            },
            # Guide creator
            {
                "content": """# Implementation Guide: Excel Report Automation

## Prerequisites
- Python 3.8+
- Basic Excel knowledge

## Step-by-Step Implementation
### Step 1: Environment Setup
```bash
pip install pandas openpyxl fastapi
```

### Step 2: Data Processing
```python
import pandas as pd

def process_excel(file_path):
    df = pd.read_excel(file_path)
    return df.describe()
```

## Testing Strategy
- Unit tests for data processing
- Integration tests for file handling""",
                "usage": {"tokens": 200}
            }
        ]
        
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service), \
             patch('backend.app.agents.context_collector.llm_service', mock_llm_service), \
             patch('backend.app.agents.requirements_generator.llm_service', mock_llm_service), \
             patch('backend.app.agents.solution_designer.llm_service', mock_llm_service), \
             patch('backend.app.agents.guide_creator.llm_service', mock_llm_service):
            
            mock_llm_service.generate_response.side_effect = llm_responses
            
            # Step 1: Start analysis
            analysis_request = {
                "problem_description": "I need to automate daily Excel report generation. Currently I manually open files, validate data, and create summaries.",
                "user_context": {
                    "technical_level": "beginner",
                    "environment": "Windows 10",
                    "tools": ["Excel", "Outlook"],
                    "daily_volume": "20 files"
                }
            }
            
            start_response = await async_client.post(
                f"{base_url}/api/v1/start-analysis",
                json=analysis_request,
                timeout=30.0
            )
            
            assert start_response.status_code == 202
            start_data = start_response.json()
            thread_id = start_data["thread_id"]
            
            # Step 2: Poll for completion
            max_polls = 30  # 30 seconds max
            final_status = None
            
            for poll_count in range(max_polls):
                status_response = await async_client.get(
                    f"{base_url}/api/v1/status/{thread_id}",
                    timeout=10.0
                )
                
                assert status_response.status_code == 200
                status_data = status_response.json()
                
                print(f"Poll {poll_count + 1}: {status_data['status']} - {status_data['current_step']} ({status_data['progress_percentage']}%)")
                
                if status_data["status"] == "completed":
                    final_status = status_data
                    break
                elif status_data["status"] == "error":
                    pytest.fail(f"Workflow failed: {status_data.get('message')}")
                elif status_data["requires_input"]:
                    # Should not require input for this test case
                    pytest.fail("Unexpected user input required")
                
                await asyncio.sleep(1)
            
            # Verify completion
            assert final_status is not None, "Workflow did not complete in time"
            assert final_status["progress_percentage"] == 100
            assert "results" in final_status
            
            results = final_status["results"]
            assert "requirements_doc" in results
            assert "implementation_guide" in results
            assert "solution_type" in results
            
            # Verify content quality
            requirements = results["requirements_doc"]
            assert "Functional Requirements" in requirements
            assert "Excel" in requirements
            assert "FR-001" in requirements
            
            guide = results["implementation_guide"]
            assert "Implementation Guide" in guide
            assert "python" in guide.lower()
            assert "pandas" in guide
            assert "Step-by-Step" in guide

    @pytest.mark.asyncio
    async def test_complete_user_journey_with_hitl(
        self, async_client, mock_llm_service
    ):
        """Test complete user journey with human-in-the-loop interaction."""
        base_url = "http://localhost:8000"
        
        # First set of responses (will require user input)
        initial_responses = [
            # Analyzer
            {
                "content": '{"problem_type": "complex_automation", "complexity": "medium", "solution_category": "SIMPLE_AUTOMATION"}',
                "usage": {"tokens": 100}
            },
            # Context collector (needs more info)
            {
                "content": '{"questions": ["What specific data validation rules do you need?", "What format should the output reports be in?"], "context_required": true}',
                "usage": {"tokens": 80}
            }
        ]
        
        # Continuation responses (after user input)
        continuation_responses = [
            # Context collector (processes user input)
            {
                "content": '{"context_sufficient": true, "analysis": "User provided detailed validation rules and output requirements"}',
                "usage": {"tokens": 90}
            },
            # Requirements generator
            {
                "content": """# Requirements\n## Functional Requirements\n- Validate data according to business rules\n- Generate PDF reports""",
                "usage": {"tokens": 120}
            },
            # Solution designer
            {
                "content": '{"solution_type": "SIMPLE_AUTOMATION", "technology_stack": {"language": "python", "libraries": ["pandas", "reportlab"]}}',
                "usage": {"tokens": 100}
            },
            # Guide creator
            {
                "content": "# Implementation Guide\n## Validation Module\n```python\ndef validate_data(df): pass\n```",
                "usage": {"tokens": 150}
            }
        ]
        
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service), \
             patch('backend.app.agents.context_collector.llm_service', mock_llm_service), \
             patch('backend.app.agents.requirements_generator.llm_service', mock_llm_service), \
             patch('backend.app.agents.solution_designer.llm_service', mock_llm_service), \
             patch('backend.app.agents.guide_creator.llm_service', mock_llm_service):
            
            mock_llm_service.generate_response.side_effect = initial_responses
            
            # Step 1: Start analysis
            analysis_request = {
                "problem_description": "I need complex data processing automation with specific business rules",
                "user_context": {
                    "technical_level": "intermediate"
                }
            }
            
            start_response = await async_client.post(
                f"{base_url}/api/v1/start-analysis",
                json=analysis_request
            )
            
            assert start_response.status_code == 202
            thread_id = start_response.json()["thread_id"]
            
            # Step 2: Poll until user input required
            awaiting_input = False
            for _ in range(10):
                status_response = await async_client.get(f"{base_url}/api/v1/status/{thread_id}")
                status_data = status_response.json()
                
                if status_data["requires_input"]:
                    awaiting_input = True
                    questions = status_data["questions"]
                    assert len(questions) == 2
                    assert "validation rules" in questions[0]
                    assert "output reports" in questions[1]
                    break
                
                await asyncio.sleep(0.5)
            
            assert awaiting_input, "Workflow should have requested user input"
            
            # Step 3: Provide user input
            mock_llm_service.generate_response.side_effect = continuation_responses
            
            resume_request = {
                "user_input": "I need to validate that all amounts are positive numbers and dates are within the last year. Output should be PDF reports with company branding.",
                "context_data": {
                    "validation_rules": ["positive_amounts", "recent_dates"],
                    "output_format": "PDF",
                    "branding_required": True
                }
            }
            
            resume_response = await async_client.post(
                f"{base_url}/api/v1/resume/{thread_id}",
                json=resume_request
            )
            
            assert resume_response.status_code == 200
            
            # Step 4: Poll for final completion
            final_status = None
            for _ in range(20):
                status_response = await async_client.get(f"{base_url}/api/v1/status/{thread_id}")
                status_data = status_response.json()
                
                if status_data["status"] == "completed":
                    final_status = status_data
                    break
                
                await asyncio.sleep(1)
            
            assert final_status is not None
            assert final_status["progress_percentage"] == 100

    @pytest.mark.asyncio
    async def test_error_handling_system_level(self, async_client):
        """Test system-level error handling."""
        base_url = "http://localhost:8000"
        
        # Test invalid analysis request
        invalid_request = {
            "problem_description": "",  # Empty description
            "user_context": {}
        }
        
        response = await async_client.post(
            f"{base_url}/api/v1/start-analysis",
            json=invalid_request
        )
        
        assert response.status_code == 400
        
        # Test non-existent thread status
        response = await async_client.get(f"{base_url}/api/v1/status/non-existent-thread")
        assert response.status_code == 404
        
        # Test invalid resume request
        response = await async_client.post(
            f"{base_url}/api/v1/resume/non-existent-thread",
            json={"user_input": "test", "context_data": {}}
        )
        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_concurrent_user_sessions(self, async_client, mock_llm_service):
        """Test multiple concurrent user sessions."""
        base_url = "http://localhost:8000"
        
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": '{"problem_type": "automation", "complexity": "low", "solution_category": "SIMPLE_AUTOMATION"}',
                "usage": {"tokens": 100}
            }
            
            # Start multiple concurrent analyses
            tasks = []
            for i in range(5):
                request = {
                    "problem_description": f"Automation problem {i}",
                    "user_context": {"technical_level": "beginner", "user_id": f"user_{i}"}
                }
                task = async_client.post(f"{base_url}/api/v1/start-analysis", json=request)
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks)
            
            # All should succeed
            thread_ids = []
            for response in responses:
                assert response.status_code == 202
                thread_ids.append(response.json()["thread_id"])
            
            # Verify all sessions are independent
            assert len(set(thread_ids)) == 5  # All unique thread IDs

    @pytest.mark.asyncio
    async def test_system_performance_under_load(self, async_client, performance_config):
        """Test system performance under load."""
        base_url = "http://localhost:8000"
        
        # Test health endpoint performance
        start_time = time.time()
        
        tasks = []
        for _ in range(performance_config["concurrent_users"]):
            task = async_client.get(f"{base_url}/api/health")
            tasks.append(task)
        
        responses = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # All health checks should succeed
        for response in responses:
            assert response.status_code == 200
        
        # Performance should be reasonable
        total_time = end_time - start_time
        avg_response_time = total_time / len(responses)
        
        assert avg_response_time < performance_config["max_response_time"]

    @pytest.mark.asyncio
    async def test_data_persistence_across_restarts(self, async_client, mock_llm_service):
        """Test data persistence across system restarts."""
        base_url = "http://localhost:8000"
        
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": '{"problem_type": "automation", "complexity": "low", "solution_category": "SIMPLE_AUTOMATION"}',
                "usage": {"tokens": 100}
            }
            
            # Start analysis
            request = {
                "problem_description": "Test persistence",
                "user_context": {"technical_level": "beginner"}
            }
            
            response = await async_client.post(f"{base_url}/api/v1/start-analysis", json=request)
            assert response.status_code == 202
            thread_id = response.json()["thread_id"]
            
            # Let it run for a bit
            await asyncio.sleep(2)
            
            # Check status (simulates system restart by making new request)
            status_response = await async_client.get(f"{base_url}/api/v1/status/{thread_id}")
            
            # Should still be able to retrieve the session
            assert status_response.status_code == 200
            status_data = status_response.json()
            assert status_data["thread_id"] == thread_id

    @pytest.mark.asyncio
    async def test_workflow_timeout_system_level(self, async_client, mock_llm_service):
        """Test workflow timeout at system level."""
        base_url = "http://localhost:8000"
        
        # Mock very slow LLM response
        async def slow_llm_response(*args, **kwargs):
            await asyncio.sleep(30)  # Very slow
            return {"content": "slow response", "usage": {"tokens": 50}}
        
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.side_effect = slow_llm_response
            
            request = {
                "problem_description": "Test timeout",
                "user_context": {"technical_level": "beginner"}
            }
            
            response = await async_client.post(f"{base_url}/api/v1/start-analysis", json=request)
            thread_id = response.json()["thread_id"]
            
            # Poll with timeout
            start_time = time.time()
            timeout_occurred = False
            
            for _ in range(10):  # 10 second timeout
                if time.time() - start_time > 10:
                    timeout_occurred = True
                    break
                
                status_response = await async_client.get(f"{base_url}/api/v1/status/{thread_id}")
                if status_response.status_code == 200:
                    status_data = status_response.json()
                    if status_data["status"] in ["completed", "error"]:
                        break
                
                await asyncio.sleep(1)
            
            # System should handle timeout gracefully
            assert timeout_occurred or status_response.status_code == 200

    @pytest.mark.asyncio
    async def test_complete_rag_workflow(self, async_client, mock_llm_service, mock_rag_service):
        """Test complete RAG system workflow."""
        base_url = "http://localhost:8000"
        
        rag_responses = [
            # Analyzer
            {
                "content": '{"problem_type": "knowledge_management", "complexity": "high", "solution_category": "RAG"}',
                "usage": {"tokens": 120}
            },
            # Context collector
            {
                "content": '{"context_sufficient": true, "analysis": "RAG requirements clear"}',
                "usage": {"tokens": 80}
            },
            # Requirements with RAG focus
            {
                "content": """# RAG System Requirements
## Functional Requirements
- Document ingestion and indexing
- Vector similarity search
- LLM-based answer generation""",
                "usage": {"tokens": 150}
            },
            # RAG solution design
            {
                "content": '{"solution_type": "RAG", "technology_stack": {"language": "python", "libraries": ["langchain", "chromadb", "openai"]}}',
                "usage": {"tokens": 130}
            },
            # RAG implementation guide
            {
                "content": """# RAG Implementation Guide
## Setup
```python
from langchain import VectorDB
from langchain.chains import RetrievalQA
```""",
                "usage": {"tokens": 180}
            }
        ]
        
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service), \
             patch('backend.app.agents.context_collector.llm_service', mock_llm_service), \
             patch('backend.app.agents.requirements_generator.llm_service', mock_llm_service), \
             patch('backend.app.agents.solution_designer.llm_service', mock_llm_service), \
             patch('backend.app.agents.guide_creator.llm_service', mock_llm_service):
            
            mock_llm_service.generate_response.side_effect = rag_responses
            
            request = {
                "problem_description": "Build a document question-answering system for our knowledge base",
                "user_context": {
                    "technical_level": "advanced",
                    "document_types": ["PDF", "Word", "Text"],
                    "expected_queries": "Technical documentation questions"
                }
            }
            
            response = await async_client.post(f"{base_url}/api/v1/start-analysis", json=request)
            assert response.status_code == 202
            thread_id = response.json()["thread_id"]
            
            # Poll for completion
            for _ in range(20):
                status_response = await async_client.get(f"{base_url}/api/v1/status/{thread_id}")
                status_data = status_response.json()
                
                if status_data["status"] == "completed":
                    results = status_data["results"]
                    assert results["solution_type"] == "RAG"
                    assert "langchain" in results["implementation_guide"]
                    assert "Vector" in results["requirements_doc"]
                    break
                
                await asyncio.sleep(1)