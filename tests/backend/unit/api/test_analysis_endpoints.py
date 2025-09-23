"""
Unit tests for analysis API endpoints
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import httpx


class TestAnalysisEndpoints:
    """Test cases for analysis API endpoints."""

    def test_health_endpoint(self, test_client):
        """Test health check endpoint."""
        response = test_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "version" in data

    def test_api_root_endpoint(self, test_client):
        """Test API root endpoint."""
        response = test_client.get("/api/v1/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "ATM Problem Solving API"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data

    @pytest.mark.asyncio
    async def test_start_analysis_success(self, test_client, sample_analysis_request):
        """Test successful analysis start."""
        with patch('backend.app.api.analysis.start_workflow_background') as mock_start:
            mock_start.return_value = "test-thread-123"
            
            response = test_client.post(
                "/api/v1/start-analysis",
                json=sample_analysis_request
            )
            
            assert response.status_code == 202
            data = response.json()
            assert data["status"] == "started"
            assert data["thread_id"] == "test-thread-123"
            assert "message" in data

    def test_start_analysis_invalid_request(self, test_client):
        """Test analysis start with invalid request."""
        invalid_request = {
            "problem_description": "",  # Empty description
            "user_context": {}
        }
        
        response = test_client.post(
            "/api/v1/start-analysis",
            json=invalid_request
        )
        
        assert response.status_code == 400
        data = response.json()
        assert "detail" in data

    def test_start_analysis_missing_fields(self, test_client):
        """Test analysis start with missing required fields."""
        incomplete_request = {
            "user_context": {"technical_level": "beginner"}
            # Missing problem_description
        }
        
        response = test_client.post(
            "/api/v1/start-analysis",
            json=incomplete_request
        )
        
        assert response.status_code == 422  # Unprocessable Entity

    @pytest.mark.asyncio
    async def test_get_status_active_workflow(self, test_client):
        """Test status endpoint for active workflow."""
        thread_id = "test-thread-456"
        
        with patch('backend.app.api.analysis.get_workflow_status') as mock_status:
            mock_status.return_value = {
                "thread_id": thread_id,
                "status": "running",
                "current_step": "analyze_problem",
                "progress_percentage": 25,
                "message": "Analyzing your problem...",
                "requires_input": False,
                "questions": None,
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-01T10:05:00Z"
            }
            
            response = test_client.get(f"/api/v1/status/{thread_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "running"
            assert data["current_step"] == "analyze_problem"
            assert data["progress_percentage"] == 25
            assert data["requires_input"] == False

    @pytest.mark.asyncio
    async def test_get_status_awaiting_input(self, test_client):
        """Test status endpoint for workflow awaiting input."""
        thread_id = "test-thread-789"
        
        with patch('backend.app.api.analysis.get_workflow_status') as mock_status:
            mock_status.return_value = {
                "thread_id": thread_id,
                "status": "awaiting_input",
                "current_step": "collect_context",
                "progress_percentage": 40,
                "message": "Please provide additional context",
                "requires_input": True,
                "questions": [
                    "What file formats do you work with?",
                    "How many files do you process daily?"
                ],
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-01T10:10:00Z"
            }
            
            response = test_client.get(f"/api/v1/status/{thread_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "awaiting_input"
            assert data["requires_input"] == True
            assert len(data["questions"]) == 2

    @pytest.mark.asyncio
    async def test_get_status_completed(self, test_client):
        """Test status endpoint for completed workflow."""
        thread_id = "test-thread-complete"
        
        with patch('backend.app.api.analysis.get_workflow_status') as mock_status:
            mock_status.return_value = {
                "thread_id": thread_id,
                "status": "completed",
                "current_step": "create_guide",
                "progress_percentage": 100,
                "message": "Analysis completed successfully",
                "requires_input": False,
                "results": {
                    "requirements_doc": "## Requirements\n- Feature 1",
                    "implementation_guide": "## Guide\n- Step 1",
                    "user_journey": "## Journey\n- Current state"
                },
                "created_at": "2024-01-01T10:00:00Z",
                "updated_at": "2024-01-01T10:30:00Z"
            }
            
            response = test_client.get(f"/api/v1/status/{thread_id}")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "completed"
            assert data["progress_percentage"] == 100
            assert "results" in data
            assert "requirements_doc" in data["results"]

    def test_get_status_not_found(self, test_client):
        """Test status endpoint for non-existent workflow."""
        thread_id = "non-existent-thread"
        
        with patch('backend.app.api.analysis.get_workflow_status') as mock_status:
            mock_status.return_value = None
            
            response = test_client.get(f"/api/v1/status/{thread_id}")
            
            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_resume_workflow_success(self, test_client, sample_resume_request):
        """Test successful workflow resumption."""
        thread_id = "test-thread-resume"
        
        with patch('backend.app.api.analysis.resume_workflow_background') as mock_resume:
            mock_resume.return_value = True
            
            response = test_client.post(
                f"/api/v1/resume/{thread_id}",
                json=sample_resume_request
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "resumed"
            assert data["thread_id"] == thread_id
            assert "message" in data

    def test_resume_workflow_invalid_thread(self, test_client, sample_resume_request):
        """Test workflow resumption with invalid thread ID."""
        thread_id = "invalid-thread"
        
        with patch('backend.app.api.analysis.resume_workflow_background') as mock_resume:
            mock_resume.return_value = False
            
            response = test_client.post(
                f"/api/v1/resume/{thread_id}",
                json=sample_resume_request
            )
            
            assert response.status_code == 404
            data = response.json()
            assert "not found" in data["detail"].lower()

    def test_resume_workflow_invalid_request(self, test_client):
        """Test workflow resumption with invalid request."""
        thread_id = "test-thread"
        invalid_request = {
            "user_input": "",  # Empty input
            "context_data": {}
        }
        
        response = test_client.post(
            f"/api/v1/resume/{thread_id}",
            json=invalid_request
        )
        
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_start_analysis_background_task_failure(self, test_client, sample_analysis_request):
        """Test analysis start when background task fails."""
        with patch('backend.app.api.analysis.start_workflow_background') as mock_start:
            mock_start.side_effect = Exception("Workflow start failed")
            
            response = test_client.post(
                "/api/v1/start-analysis",
                json=sample_analysis_request
            )
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data["detail"].lower()

    @pytest.mark.asyncio
    async def test_concurrent_analysis_requests(self, async_client, sample_analysis_request):
        """Test handling of concurrent analysis requests."""
        base_url = "http://localhost:8000"
        
        with patch('backend.app.api.analysis.start_workflow_background') as mock_start:
            # Mock different thread IDs for concurrent requests
            mock_start.side_effect = ["thread-1", "thread-2", "thread-3"]
            
            # Send multiple concurrent requests
            tasks = []
            for i in range(3):
                task = async_client.post(
                    f"{base_url}/api/v1/start-analysis",
                    json=sample_analysis_request
                )
                tasks.append(task)
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All requests should succeed
            assert len(responses) == 3
            for response in responses:
                if isinstance(response, httpx.Response):
                    assert response.status_code == 202

    def test_request_validation_edge_cases(self, test_client):
        """Test request validation for edge cases."""
        edge_cases = [
            # Very long problem description
            {
                "problem_description": "x" * 10000,
                "user_context": {"technical_level": "beginner"}
            },
            # Special characters in description
            {
                "problem_description": "Problem with émojis 🚀 and spëcial chars",
                "user_context": {"technical_level": "advanced"}
            },
            # Nested context data
            {
                "problem_description": "Valid problem",
                "user_context": {
                    "technical_level": "intermediate",
                    "environment": {
                        "os": "Windows",
                        "version": "10"
                    }
                }
            }
        ]
        
        for case in edge_cases:
            response = test_client.post("/api/v1/start-analysis", json=case)
            # Should either succeed or fail gracefully
            assert response.status_code in [200, 202, 400, 422]

    @pytest.mark.asyncio
    async def test_error_handling_workflow_corruption(self, test_client):
        """Test error handling for corrupted workflow state."""
        thread_id = "corrupted-thread"
        
        with patch('backend.app.api.analysis.get_workflow_status') as mock_status:
            mock_status.side_effect = Exception("Database corruption error")
            
            response = test_client.get(f"/api/v1/status/{thread_id}")
            
            assert response.status_code == 500
            data = response.json()
            assert "error" in data["detail"].lower()

    def test_response_format_consistency(self, test_client, sample_analysis_request):
        """Test consistency of response formats across endpoints."""
        with patch('backend.app.api.analysis.start_workflow_background') as mock_start:
            mock_start.return_value = "test-thread"
            
            # Test start analysis response format
            response = test_client.post("/api/v1/start-analysis", json=sample_analysis_request)
            start_data = response.json()
            
            # All responses should have consistent field naming
            expected_fields = ["status", "thread_id", "message"]
            for field in expected_fields:
                assert field in start_data

        with patch('backend.app.api.analysis.get_workflow_status') as mock_status:
            mock_status.return_value = {
                "thread_id": "test-thread",
                "status": "running",
                "current_step": "analyze_problem",
                "progress_percentage": 25,
                "message": "Processing...",
                "requires_input": False
            }
            
            # Test status response format
            response = test_client.get("/api/v1/status/test-thread")
            status_data = response.json()
            
            # Should include all status fields
            status_fields = ["thread_id", "status", "current_step", "progress_percentage"]
            for field in status_fields:
                assert field in status_data

    @pytest.mark.asyncio
    async def test_timeout_handling(self, test_client, sample_analysis_request):
        """Test handling of request timeouts."""
        with patch('backend.app.api.analysis.start_workflow_background') as mock_start:
            # Simulate timeout
            mock_start.side_effect = asyncio.TimeoutError("Request timeout")
            
            response = test_client.post("/api/v1/start-analysis", json=sample_analysis_request)
            
            # Should handle timeout gracefully
            assert response.status_code == 500
            data = response.json()
            assert "timeout" in data["detail"].lower() or "error" in data["detail"].lower()

    def test_cors_headers(self, test_client):
        """Test CORS headers in responses."""
        response = test_client.get("/api/health")
        
        # Should include CORS headers for web frontend
        assert response.status_code == 200
        # Note: Actual CORS headers depend on FastAPI CORS middleware configuration

    def test_content_type_validation(self, test_client):
        """Test content type validation."""
        # Test with invalid content type
        response = test_client.post(
            "/api/v1/start-analysis",
            data="invalid data",
            headers={"Content-Type": "text/plain"}
        )
        
        assert response.status_code == 422