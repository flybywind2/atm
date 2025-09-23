"""
Integration tests for Human-in-the-Loop (HITL) functionality
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any


class TestHITLFlow:
    """Integration tests for Human-in-the-Loop workflow."""

    @pytest.mark.asyncio
    async def test_hitl_interruption_simple_case(self, temp_db_path, mock_llm_service):
        """Test basic HITL interruption when additional context is needed."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service), \
             patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            
            # Mock analyzer to complete successfully
            analyzer_response = {
                "content": '{"problem_type": "automation", "complexity": "medium", "solution_category": "SIMPLE_AUTOMATION"}',
                "usage": {"tokens": 100}
            }
            
            # Mock context collector to require user input
            context_response = {
                "content": '{"questions": ["What file formats do you work with?", "How many files per day?"], "context_required": true, "priority": "high"}',
                "usage": {"tokens": 80}
            }
            
            mock_llm_service.generate_response.side_effect = [
                analyzer_response, context_response
            ]
            
            from backend.app.workflows.graph import get_compiled_workflow, get_workflow_config
            from backend.app.database.checkpointer import create_new_workflow_session
            
            thread_id = create_new_workflow_session(
                user_id="test-hitl-basic",
                db_path=temp_db_path
            )
            
            # Enable human-in-the-loop
            workflow = await get_compiled_workflow(
                db_path=temp_db_path,
                enable_human_loop=True
            )
            
            config = get_workflow_config(thread_id)
            
            initial_state = {
                "problem_description": "I need to automate something but need help with details",
                "conversation_history": [],
                "context_data": {"technical_level": "beginner"},
                "current_step": "start",
                "current_status": "initialized",
                "context_complete": False,
                "requires_user_input": False,
                "retry_count": 0
            }
            
            result = await workflow.ainvoke(initial_state, config)
            
            # Should be interrupted for user input
            assert result["requires_user_input"] == True
            assert result["current_step"] == "collect_context"
            assert result["current_status"] == "awaiting_input"
            assert "questions" in result
            assert len(result["questions"]) == 2
            assert "file formats" in result["questions"][0]

    @pytest.mark.asyncio
    async def test_hitl_resume_with_user_response(self, temp_db_path, mock_llm_service):
        """Test resuming workflow after user provides input."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service), \
             patch('backend.app.agents.requirements_generator.llm_service', mock_llm_service):
            
            # Setup initial interrupted state
            from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
            from backend.app.workflows.graph import get_compiled_workflow, get_workflow_config
            
            checkpointer = await create_checkpointer(temp_db_path)
            thread_id = "test-hitl-resume"
            
            interrupted_state = {
                "problem_description": "Automate daily data processing",
                "conversation_history": [
                    {"role": "user", "content": "I need to automate daily data processing"},
                    {"role": "assistant", "content": "I'll help you with that. I need some more details."}
                ],
                "context_data": {"technical_level": "intermediate"},
                "current_step": "collect_context",
                "current_status": "awaiting_input",
                "context_complete": False,
                "requires_user_input": True,
                "questions": [
                    "What type of data do you work with?",
                    "What is the current manual process?"
                ],
                "retry_count": 0,
                "progress_percentage": 30
            }
            
            await save_checkpoint(checkpointer, thread_id, interrupted_state)
            
            # Mock context collector to process user input and continue
            context_response = {
                "content": '{"context_sufficient": true, "analysis": "User provided sufficient context about Excel data processing"}',
                "usage": {"tokens": 90}
            }
            
            # Mock requirements generator
            requirements_response = {
                "content": """# Requirements Document
## Functional Requirements
- Process Excel files automatically
- Generate daily summary reports
- Handle data validation""",
                "usage": {"tokens": 150}
            }
            
            mock_llm_service.generate_response.side_effect = [
                context_response, requirements_response
            ]
            
            workflow = await get_compiled_workflow(db_path=temp_db_path)
            config = get_workflow_config(thread_id)
            
            # Resume with user input
            resumed_state = {
                **interrupted_state,
                "user_input": "I work with Excel files containing sales data. Currently I manually open each file, validate data, and create summary reports.",
                "context_data": {
                    **interrupted_state["context_data"],
                    "data_type": "Excel sales data",
                    "current_process": "manual validation and reporting",
                    "file_format": "Excel"
                },
                "requires_user_input": False,
                "questions": None
            }
            
            result = await workflow.ainvoke(resumed_state, config)
            
            # Should progress beyond context collection
            assert result["current_step"] != "collect_context"
            assert result["context_complete"] == True
            assert result["progress_percentage"] > 30
            assert "requirements_doc" in result
            assert "Excel files" in result["requirements_doc"]

    @pytest.mark.asyncio
    async def test_hitl_multiple_interruptions(self, temp_db_path, mock_llm_service):
        """Test workflow with multiple HITL interruptions."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            
            # Simulate multiple rounds of context collection
            responses = [
                # First context collection - still needs more info
                {
                    "content": '{"questions": ["What is your technical background?"], "context_required": true}',
                    "usage": {"tokens": 60}
                },
                # Second context collection - needs clarification
                {
                    "content": '{"questions": ["How many files do you process daily?"], "context_required": true}',
                    "usage": {"tokens": 70}
                },
                # Third context collection - sufficient
                {
                    "content": '{"context_sufficient": true, "analysis": "Now have sufficient context"}',
                    "usage": {"tokens": 80}
                }
            ]
            
            mock_llm_service.generate_response.side_effect = responses
            
            from backend.app.workflows.graph import get_compiled_workflow, get_workflow_config
            from backend.app.database.checkpointer import create_new_workflow_session
            
            thread_id = create_new_workflow_session(
                user_id="test-multiple-hitl",
                db_path=temp_db_path
            )
            
            workflow = await get_compiled_workflow(
                db_path=temp_db_path,
                enable_human_loop=True
            )
            config = get_workflow_config(thread_id)
            
            # Start with minimal context
            state = {
                "problem_description": "I need automation help",
                "conversation_history": [],
                "context_data": {},
                "current_step": "collect_context",
                "current_status": "initialized",
                "context_complete": False,
                "requires_user_input": False
            }
            
            # First execution - should ask for technical background
            result1 = await workflow.ainvoke(state, config)
            assert result1["requires_user_input"] == True
            assert "technical background" in result1["questions"][0]
            
            # Resume with first answer
            state2 = {
                **result1,
                "user_input": "I'm a beginner with basic Excel skills",
                "context_data": {"technical_level": "beginner", "tools": ["Excel"]},
                "requires_user_input": False
            }
            
            result2 = await workflow.ainvoke(state2, config)
            assert result2["requires_user_input"] == True
            assert "how many files" in result2["questions"][0].lower()
            
            # Resume with second answer
            state3 = {
                **result2,
                "user_input": "About 20 files per day",
                "context_data": {
                    **result2["context_data"],
                    "daily_volume": "20 files"
                },
                "requires_user_input": False
            }
            
            result3 = await workflow.ainvoke(state3, config)
            # Should now have sufficient context
            assert result3["context_complete"] == True
            assert result3["requires_user_input"] == False

    @pytest.mark.asyncio
    async def test_hitl_timeout_handling(self, temp_db_path, mock_llm_service):
        """Test HITL timeout when user doesn't respond."""
        from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
        from backend.app.workflows.graph import check_hitl_timeout
        from datetime import datetime, timedelta
        
        checkpointer = await create_checkpointer(temp_db_path)
        thread_id = "test-hitl-timeout"
        
        # Create state that's been waiting for user input for too long
        old_timestamp = (datetime.now() - timedelta(hours=25)).isoformat()
        
        waiting_state = {
            "problem_description": "Test timeout",
            "current_step": "collect_context",
            "current_status": "awaiting_input",
            "requires_user_input": True,
            "questions": ["What do you need help with?"],
            "last_activity": old_timestamp,
            "timeout_count": 0
        }
        
        await save_checkpoint(checkpointer, thread_id, waiting_state)
        
        # Check for timeout
        is_timed_out = await check_hitl_timeout(thread_id, temp_db_path, timeout_hours=24)
        
        assert is_timed_out == True

    @pytest.mark.asyncio
    async def test_hitl_context_validation(self, temp_db_path, mock_llm_service):
        """Test validation of user-provided context."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            
            # Mock context validation responses
            validation_responses = [
                # Invalid/insufficient context
                {
                    "content": '{"context_valid": false, "issues": ["Missing technical details"], "follow_up_questions": ["What programming experience do you have?"]}',
                    "usage": {"tokens": 80}
                },
                # Valid context
                {
                    "content": '{"context_valid": true, "context_sufficient": true, "analysis": "Sufficient context provided"}',
                    "usage": {"tokens": 70}
                }
            ]
            
            mock_llm_service.generate_response.side_effect = validation_responses
            
            from backend.app.agents.context_collector import validate_user_context
            
            # Test invalid context
            invalid_context = {
                "user_input": "I need help",
                "context_data": {}
            }
            
            validation1 = await validate_user_context(invalid_context)
            assert validation1["context_valid"] == False
            assert len(validation1["issues"]) > 0
            
            # Test valid context
            valid_context = {
                "user_input": "I'm a Python developer with 3 years experience working on data automation",
                "context_data": {
                    "technical_level": "intermediate",
                    "experience": "3 years",
                    "domain": "data automation"
                }
            }
            
            validation2 = await validate_user_context(valid_context)
            assert validation2["context_valid"] == True

    @pytest.mark.asyncio
    async def test_hitl_conversation_tracking(self, temp_db_path, mock_llm_service):
        """Test conversation history tracking during HITL interactions."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            
            mock_llm_service.generate_response.return_value = {
                "content": '{"questions": ["What tools do you currently use?"], "context_required": true}',
                "usage": {"tokens": 60}
            }
            
            from backend.app.workflows.state import add_conversation_entry
            
            state = {
                "conversation_history": [],
                "current_step": "collect_context"
            }
            
            # Add user message
            state = add_conversation_entry(state, "user", "I need help automating my daily tasks")
            
            # Add assistant response
            state = add_conversation_entry(state, "assistant", "I'd be happy to help. What tools do you currently use?")
            
            # Add user response
            state = add_conversation_entry(state, "user", "I mainly use Excel and email")
            
            assert len(state["conversation_history"]) == 3
            assert state["conversation_history"][0]["role"] == "user"
            assert state["conversation_history"][1]["role"] == "assistant"
            assert state["conversation_history"][2]["content"] == "I mainly use Excel and email"
            
            # All entries should have timestamps
            for entry in state["conversation_history"]:
                assert "timestamp" in entry

    @pytest.mark.asyncio
    async def test_hitl_dynamic_question_generation(self, temp_db_path, mock_llm_service):
        """Test dynamic question generation based on problem context."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            
            # Different question sets based on problem type
            automation_questions = {
                "content": '{"questions": ["What tasks are you automating?", "What file formats are involved?"], "focus": "automation"}',
                "usage": {"tokens": 80}
            }
            
            rag_questions = {
                "content": '{"questions": ["What documents will be indexed?", "What types of queries do you expect?"], "focus": "rag"}',
                "usage": {"tokens": 85}
            }
            
            from backend.app.agents.context_collector import generate_dynamic_questions
            
            # Test automation context
            automation_context = {
                "problem_analysis": {"solution_category": "SIMPLE_AUTOMATION"},
                "problem_description": "Automate daily report generation"
            }
            
            mock_llm_service.generate_response.return_value = automation_questions
            automation_result = await generate_dynamic_questions(automation_context)
            
            assert "automating" in automation_result["questions"][0]
            assert "file formats" in automation_result["questions"][1]
            
            # Test RAG context
            rag_context = {
                "problem_analysis": {"solution_category": "RAG"},
                "problem_description": "Build document search system"
            }
            
            mock_llm_service.generate_response.return_value = rag_questions
            rag_result = await generate_dynamic_questions(rag_context)
            
            assert "documents" in rag_result["questions"][0]
            assert "queries" in rag_result["questions"][1]

    @pytest.mark.asyncio
    async def test_hitl_context_prioritization(self, temp_db_path):
        """Test prioritization of context collection questions."""
        from backend.app.agents.context_collector import prioritize_context_questions
        
        all_questions = [
            {"question": "What is your technical level?", "priority": "high", "category": "technical"},
            {"question": "What is your budget?", "priority": "low", "category": "business"},
            {"question": "What tools do you use?", "priority": "high", "category": "technical"},
            {"question": "What is your timeline?", "priority": "medium", "category": "project"},
            {"question": "Do you have security requirements?", "priority": "low", "category": "security"}
        ]
        
        prioritized = prioritize_context_questions(all_questions, max_questions=3)
        
        # Should return top 3 by priority
        assert len(prioritized) == 3
        
        # High priority questions should be first
        high_priority_count = sum(1 for q in prioritized if q["priority"] == "high")
        assert high_priority_count >= 2

    @pytest.mark.asyncio
    async def test_hitl_error_recovery(self, temp_db_path, mock_llm_service):
        """Test error recovery during HITL interactions."""
        with patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            
            # Simulate LLM service failure during context collection
            mock_llm_service.generate_response.side_effect = [
                Exception("Service temporarily unavailable"),
                {
                    "content": '{"questions": ["Recovered question"], "context_required": true}',
                    "usage": {"tokens": 50}
                }
            ]
            
            from backend.app.agents.context_collector import collect_context_with_retry
            
            state = {
                "problem_description": "Test error recovery",
                "current_step": "collect_context",
                "retry_count": 0
            }
            
            result = await collect_context_with_retry(state, max_retries=2)
            
            # Should recover and provide questions
            assert result["retry_count"] == 1
            assert "questions" in result
            assert result["questions"][0] == "Recovered question"

    @pytest.mark.asyncio
    async def test_hitl_context_merging(self, temp_db_path):
        """Test merging of context data from multiple HITL interactions."""
        from backend.app.workflows.state import merge_context_data
        
        # Initial context
        initial_context = {
            "technical_level": "beginner",
            "environment": "Windows"
        }
        
        # First user input
        first_input = {
            "file_format": "Excel",
            "daily_volume": "10 files"
        }
        
        # Second user input (with some overlap)
        second_input = {
            "daily_volume": "15 files",  # Updated value
            "process_complexity": "medium",
            "deadline": "urgent"
        }
        
        state = {"context_data": initial_context}
        
        # Merge first input
        state = merge_context_data(state, first_input)
        assert state["context_data"]["file_format"] == "Excel"
        assert state["context_data"]["daily_volume"] == "10 files"
        
        # Merge second input
        state = merge_context_data(state, second_input)
        assert state["context_data"]["daily_volume"] == "15 files"  # Should update
        assert state["context_data"]["process_complexity"] == "medium"
        assert state["context_data"]["technical_level"] == "beginner"  # Should preserve