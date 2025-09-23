"""
Unit tests for the problem analyzer agent
"""

import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any


class TestAnalyzerAgent:
    """Test cases for the analyzer agent."""

    @pytest.mark.asyncio
    async def test_analyze_simple_automation(self, sample_workflow_state, mock_llm_service):
        """Test analysis of simple automation problem."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            # Mock LLM response for analysis
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "problem_type": "automation",
                    "complexity": "low",
                    "solution_category": "SIMPLE_AUTOMATION",
                    "key_components": ["data_processing", "file_handling"],
                    "estimated_effort": "2-4 hours"
                }
                """,
                "usage": {"tokens": 150}
            }
            
            from backend.app.agents.analyzer import analyze_problem
            
            state = {
                **sample_workflow_state,
                "problem_description": "Automate daily Excel report generation"
            }
            
            result = await analyze_problem(state)
            
            # Verify state updates
            assert result["current_step"] == "analyze_problem"
            assert result["current_status"] == "analyzing"
            assert "problem_analysis" in result
            assert result["problem_analysis"]["solution_category"] == "SIMPLE_AUTOMATION"
            
            # Verify LLM was called
            mock_llm_service.generate_response.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_complex_rag_problem(self, sample_workflow_state, mock_llm_service):
        """Test analysis of complex RAG problem."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "problem_type": "knowledge_management",
                    "complexity": "high",
                    "solution_category": "RAG",
                    "key_components": ["document_embedding", "vector_search", "llm_generation"],
                    "estimated_effort": "2-3 weeks"
                }
                """,
                "usage": {"tokens": 200}
            }
            
            from backend.app.agents.analyzer import analyze_problem
            
            state = {
                **sample_workflow_state,
                "problem_description": "Build a document Q&A system for company knowledge base"
            }
            
            result = await analyze_problem(state)
            
            assert result["problem_analysis"]["solution_category"] == "RAG"
            assert result["problem_analysis"]["complexity"] == "high"

    @pytest.mark.asyncio
    async def test_analyze_ml_classification_problem(self, sample_workflow_state, mock_llm_service):
        """Test analysis of ML classification problem."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "problem_type": "classification",
                    "complexity": "medium",
                    "solution_category": "ML_CLASSIFICATION",
                    "key_components": ["data_preprocessing", "model_training", "prediction_api"],
                    "estimated_effort": "1-2 weeks"
                }
                """,
                "usage": {"tokens": 180}
            }
            
            from backend.app.agents.analyzer import analyze_problem
            
            state = {
                **sample_workflow_state,
                "problem_description": "Classify customer feedback into categories"
            }
            
            result = await analyze_problem(state)
            
            assert result["problem_analysis"]["solution_category"] == "ML_CLASSIFICATION"
            assert result["problem_analysis"]["complexity"] == "medium"

    @pytest.mark.asyncio
    async def test_analyze_with_llm_error(self, sample_workflow_state, mock_llm_service):
        """Test analyzer behavior when LLM service fails."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            # Mock LLM service failure
            mock_llm_service.generate_response.side_effect = Exception("LLM service unavailable")
            
            from backend.app.agents.analyzer import analyze_problem
            
            state = sample_workflow_state.copy()
            
            result = await analyze_problem(state)
            
            # Should handle error gracefully
            assert result["current_status"] == "error"
            assert "error" in result
            assert result["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_analyze_with_invalid_response(self, sample_workflow_state, mock_llm_service):
        """Test analyzer behavior with invalid LLM response."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            # Mock invalid JSON response
            mock_llm_service.generate_response.return_value = {
                "content": "Invalid JSON response",
                "usage": {"tokens": 50}
            }
            
            from backend.app.agents.analyzer import analyze_problem
            
            state = sample_workflow_state.copy()
            
            result = await analyze_problem(state)
            
            # Should handle parsing error
            assert result["current_status"] == "error"
            assert "error" in result

    def test_extract_problem_keywords(self):
        """Test problem keyword extraction."""
        from backend.app.agents.analyzer import extract_problem_keywords
        
        test_cases = [
            ("Automate daily Excel reports", ["automate", "excel", "reports"]),
            ("Build chatbot for customer support", ["build", "chatbot", "customer", "support"]),
            ("Classify text documents using ML", ["classify", "text", "documents", "ml"])
        ]
        
        for description, expected_keywords in test_cases:
            keywords = extract_problem_keywords(description)
            for keyword in expected_keywords:
                assert keyword.lower() in [k.lower() for k in keywords]

    def test_classify_solution_category(self):
        """Test solution category classification logic."""
        from backend.app.agents.analyzer import classify_solution_category
        
        test_cases = [
            ("automate daily reports", "SIMPLE_AUTOMATION"),
            ("build document search system", "RAG"),
            ("classify customer feedback", "ML_CLASSIFICATION"),
            ("process data files", "SIMPLE_AUTOMATION")
        ]
        
        for description, expected_category in test_cases:
            category = classify_solution_category(description)
            assert category == expected_category

    @pytest.mark.asyncio
    async def test_analyze_with_context_data(self, sample_workflow_state, mock_llm_service):
        """Test analyzer with existing context data."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "problem_type": "automation",
                    "complexity": "low",
                    "solution_category": "SIMPLE_AUTOMATION",
                    "key_components": ["data_processing"],
                    "estimated_effort": "1-2 hours"
                }
                """,
                "usage": {"tokens": 120}
            }
            
            from backend.app.agents.analyzer import analyze_problem
            
            state = {
                **sample_workflow_state,
                "context_data": {
                    "technical_level": "beginner",
                    "environment": "Windows",
                    "tools": ["Excel"]
                }
            }
            
            result = await analyze_problem(state)
            
            # Should consider existing context
            assert result["current_step"] == "analyze_problem"
            assert "problem_analysis" in result
            
            # Verify context was passed to LLM
            call_args = mock_llm_service.generate_response.call_args
            assert "technical_level" in str(call_args)

    @pytest.mark.asyncio
    async def test_analyze_progress_tracking(self, sample_workflow_state, mock_llm_service):
        """Test progress percentage tracking during analysis."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "problem_type": "automation",
                    "complexity": "low",
                    "solution_category": "SIMPLE_AUTOMATION"
                }
                """,
                "usage": {"tokens": 100}
            }
            
            from backend.app.agents.analyzer import analyze_problem
            
            result = await analyze_problem(sample_workflow_state)
            
            # Should update progress
            assert result["progress_percentage"] > 0
            assert result["progress_percentage"] <= 20  # First step should be ~20%