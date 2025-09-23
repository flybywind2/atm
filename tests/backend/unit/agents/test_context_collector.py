"""
Unit tests for the context collector agent
"""

import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any


class TestContextCollectorAgent:
    """Test cases for the context collector agent."""

    @pytest.mark.asyncio
    async def test_collect_context_basic_automation(self, sample_workflow_state, mock_llm_service):
        """Test context collection for basic automation scenario."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "questions": [
                        "What file formats do you work with?",
                        "How many files do you process daily?",
                        "What is your current workflow?"
                    ],
                    "context_required": true,
                    "priority": "high"
                }
                """,
                "usage": {"tokens": 120}
            }
            
            from backend.app.agents.context_collector import collect_context
            
            state = {
                **sample_workflow_state,
                "problem_analysis": {
                    "solution_category": "SIMPLE_AUTOMATION",
                    "complexity": "low"
                }
            }
            
            result = await collect_context(state)
            
            # Should generate questions and require user input
            assert result["current_step"] == "collect_context"
            assert result["requires_user_input"] == True
            assert "questions" in result
            assert len(result["questions"]) > 0
            assert result["context_complete"] == False

    @pytest.mark.asyncio
    async def test_collect_context_with_sufficient_data(self, sample_workflow_state, mock_llm_service):
        """Test context collection when sufficient context already exists."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "context_sufficient": true,
                    "analysis": "Sufficient context provided for automation task",
                    "next_step": "requirements"
                }
                """,
                "usage": {"tokens": 80}
            }
            
            from backend.app.agents.context_collector import collect_context
            
            state = {
                **sample_workflow_state,
                "context_data": {
                    "file_format": "Excel",
                    "daily_volume": "10 files",
                    "current_process": "Manual copy-paste",
                    "environment": "Windows 10",
                    "technical_level": "beginner"
                },
                "problem_analysis": {
                    "solution_category": "SIMPLE_AUTOMATION"
                }
            }
            
            result = await collect_context(state)
            
            # Should not require additional input
            assert result["context_complete"] == True
            assert result["requires_user_input"] == False
            assert result["current_status"] == "context_collected"

    @pytest.mark.asyncio
    async def test_collect_context_rag_system(self, sample_workflow_state, mock_llm_service):
        """Test context collection for RAG system."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "questions": [
                        "What types of documents will be indexed?",
                        "What is the expected query volume?",
                        "Do you need real-time updates?",
                        "What is your preferred deployment environment?"
                    ],
                    "context_required": true,
                    "technical_focus": ["document_formats", "infrastructure", "performance"]
                }
                """,
                "usage": {"tokens": 150}
            }
            
            from backend.app.agents.context_collector import collect_context
            
            state = {
                **sample_workflow_state,
                "problem_analysis": {
                    "solution_category": "RAG",
                    "complexity": "high"
                }
            }
            
            result = await collect_context(state)
            
            # Should generate technical questions for RAG
            assert result["requires_user_input"] == True
            assert len(result["questions"]) >= 3
            assert any("document" in q.lower() for q in result["questions"])

    @pytest.mark.asyncio
    async def test_process_user_input_valid(self, sample_workflow_state):
        """Test processing valid user input."""
        from backend.app.agents.context_collector import process_user_input
        
        state = {
            **sample_workflow_state,
            "questions": [
                "What file formats do you work with?",
                "How many files daily?"
            ]
        }
        
        user_input = "I work with Excel files, about 20 files per day"
        context_data = {
            "file_format": "Excel",
            "daily_volume": "20 files"
        }
        
        result = process_user_input(state, user_input, context_data)
        
        assert result["context_complete"] == True
        assert result["requires_user_input"] == False
        assert "file_format" in result["context_data"]
        assert result["context_data"]["file_format"] == "Excel"

    @pytest.mark.asyncio
    async def test_collect_context_ml_classification(self, sample_workflow_state, mock_llm_service):
        """Test context collection for ML classification task."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "questions": [
                        "What are you trying to classify?",
                        "Do you have labeled training data?",
                        "What categories do you want to predict?",
                        "What is the data volume?"
                    ],
                    "context_required": true,
                    "ml_focus": ["training_data", "categories", "performance_requirements"]
                }
                """,
                "usage": {"tokens": 140}
            }
            
            from backend.app.agents.context_collector import collect_context
            
            state = {
                **sample_workflow_state,
                "problem_analysis": {
                    "solution_category": "ML_CLASSIFICATION",
                    "complexity": "medium"
                }
            }
            
            result = await collect_context(state)
            
            # Should focus on ML-specific questions
            assert result["requires_user_input"] == True
            assert any("training" in q.lower() or "data" in q.lower() for q in result["questions"])
            assert any("categor" in q.lower() for q in result["questions"])

    def test_validate_context_completeness(self):
        """Test context completeness validation."""
        from backend.app.agents.context_collector import validate_context_completeness
        
        # Test complete context
        complete_context = {
            "technical_level": "intermediate",
            "environment": "Windows",
            "file_format": "Excel",
            "daily_volume": "50 files"
        }
        assert validate_context_completeness(complete_context, "SIMPLE_AUTOMATION") == True
        
        # Test incomplete context
        incomplete_context = {
            "technical_level": "beginner"
        }
        assert validate_context_completeness(incomplete_context, "SIMPLE_AUTOMATION") == False

    def test_generate_followup_questions(self):
        """Test follow-up question generation."""
        from backend.app.agents.context_collector import generate_followup_questions
        
        existing_context = {
            "file_format": "Excel"
        }
        
        problem_type = "SIMPLE_AUTOMATION"
        
        questions = generate_followup_questions(existing_context, problem_type)
        
        assert len(questions) > 0
        assert not any("excel" in q.lower() and "format" in q.lower() for q in questions)

    @pytest.mark.asyncio
    async def test_collect_context_with_error(self, sample_workflow_state, mock_llm_service):
        """Test context collection with LLM error."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            mock_llm_service.generate_response.side_effect = Exception("LLM service error")
            
            from backend.app.agents.context_collector import collect_context
            
            result = await collect_context(sample_workflow_state)
            
            # Should handle error gracefully
            assert result["current_status"] == "error"
            assert "error" in result
            assert result["retry_count"] == 1

    @pytest.mark.asyncio
    async def test_context_priority_assessment(self, sample_workflow_state, mock_llm_service):
        """Test context priority assessment."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "questions": [
                        "What is your technical background?",
                        "What tools do you currently use?"
                    ],
                    "priority_order": ["technical_background", "current_tools"],
                    "optional_questions": ["What is your budget?"]
                }
                """,
                "usage": {"tokens": 100}
            }
            
            from backend.app.agents.context_collector import collect_context
            
            result = await collect_context(sample_workflow_state)
            
            # Should prioritize essential questions
            assert len(result["questions"]) >= 2
            assert result["requires_user_input"] == True

    def test_context_data_sanitization(self):
        """Test context data sanitization and validation."""
        from backend.app.agents.context_collector import sanitize_context_data
        
        raw_context = {
            "file_format": "Excel  ",  # Extra whitespace
            "daily_volume": "ABOUT 20 FILES",  # Inconsistent case
            "technical_level": "Beginner",
            "environment": "windows 10"
        }
        
        sanitized = sanitize_context_data(raw_context)
        
        assert sanitized["file_format"] == "Excel"
        assert sanitized["daily_volume"] == "about 20 files"
        assert sanitized["technical_level"] == "beginner"
        assert sanitized["environment"] == "windows 10"

    @pytest.mark.asyncio
    async def test_collect_context_progress_update(self, sample_workflow_state, mock_llm_service):
        """Test progress tracking during context collection."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """{"questions": ["What tools do you use?"], "context_required": true}""",
                "usage": {"tokens": 50}
            }
            
            from backend.app.agents.context_collector import collect_context
            
            result = await collect_context(sample_workflow_state)
            
            # Should update progress
            assert result["progress_percentage"] > 20  # Should be higher than analysis step
            assert result["progress_percentage"] <= 40