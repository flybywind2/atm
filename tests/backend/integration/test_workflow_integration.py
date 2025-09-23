"""
Integration tests for complete workflow execution
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Dict, Any


class TestWorkflowIntegration:
    """Integration tests for complete workflow execution."""

    @pytest.mark.asyncio
    async def test_complete_workflow_simple_automation(
        self, temp_db_path, sample_analysis_request, mock_llm_service, mock_rag_service
    ):
        """Test complete workflow for simple automation problem."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service), \
             patch('backend.app.agents.context_collector.llm_service', mock_llm_service), \
             patch('backend.app.agents.requirements_generator.llm_service', mock_llm_service), \
             patch('backend.app.agents.solution_designer.llm_service', mock_llm_service), \
             patch('backend.app.agents.guide_creator.llm_service', mock_llm_service):
            
            # Mock LLM responses for each step
            mock_responses = [
                # Analyzer response
                {
                    "content": '{"problem_type": "automation", "complexity": "low", "solution_category": "SIMPLE_AUTOMATION"}',
                    "usage": {"tokens": 100}
                },
                # Context collector - sufficient context
                {
                    "content": '{"context_sufficient": true, "analysis": "Sufficient context provided"}',
                    "usage": {"tokens": 80}
                },
                # Requirements generator
                {
                    "content": "# Requirements\n## Functional Requirements\n- Process Excel files\n- Generate reports",
                    "usage": {"tokens": 150}
                },
                # Solution designer
                {
                    "content": '{"solution_type": "SIMPLE_AUTOMATION", "technology_stack": {"language": "python", "framework": "fastapi"}}',
                    "usage": {"tokens": 120}
                },
                # Guide creator
                {
                    "content": "# Implementation Guide\n## Step 1: Setup\n```python\nimport pandas\n```",
                    "usage": {"tokens": 200}
                }
            ]
            
            mock_llm_service.generate_response.side_effect = mock_responses
            
            from backend.app.workflows.graph import get_compiled_workflow, get_workflow_config
            from backend.app.database.checkpointer import create_new_workflow_session
            
            # Create workflow session
            thread_id = create_new_workflow_session(
                user_id="test-user",
                workflow_type="problem_solving",
                db_path=temp_db_path
            )
            
            # Get compiled workflow
            workflow = await get_compiled_workflow(
                db_path=temp_db_path,
                enable_human_loop=False
            )
            
            config = get_workflow_config(thread_id)
            
            # Initial state
            initial_state = {
                "problem_description": sample_analysis_request["problem_description"],
                "user_context": sample_analysis_request["user_context"],
                "conversation_history": [],
                "context_data": sample_analysis_request["user_context"],
                "current_step": "start",
                "current_status": "initialized",
                "context_complete": False,
                "requires_user_input": False,
                "retry_count": 0,
                "progress_percentage": 0
            }
            
            # Execute workflow
            result = await workflow.ainvoke(initial_state, config)
            
            # Verify completion
            assert result["current_status"] == "complete"
            assert result["progress_percentage"] == 100
            assert "requirements_doc" in result
            assert "implementation_guide" in result
            assert "solution_type" in result
            assert result["solution_type"] == "SIMPLE_AUTOMATION"

    @pytest.mark.asyncio
    async def test_workflow_with_human_in_the_loop(
        self, temp_db_path, mock_llm_service
    ):
        """Test workflow with human-in-the-loop interruption."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service), \
             patch('backend.app.agents.context_collector.llm_service', mock_llm_service):
            
            # Mock responses
            analyzer_response = {
                "content": '{"problem_type": "automation", "complexity": "medium", "solution_category": "SIMPLE_AUTOMATION"}',
                "usage": {"tokens": 100}
            }
            
            context_collector_response = {
                "content": '{"questions": ["What file formats?", "How many files daily?"], "context_required": true}',
                "usage": {"tokens": 80}
            }
            
            mock_llm_service.generate_response.side_effect = [
                analyzer_response, context_collector_response
            ]
            
            from backend.app.workflows.graph import get_compiled_workflow, get_workflow_config
            from backend.app.database.checkpointer import create_new_workflow_session
            
            # Create workflow session
            thread_id = create_new_workflow_session(
                user_id="test-user-hitl",
                db_path=temp_db_path
            )
            
            # Get compiled workflow with HITL enabled
            workflow = await get_compiled_workflow(
                db_path=temp_db_path,
                enable_human_loop=True
            )
            
            config = get_workflow_config(thread_id)
            
            initial_state = {
                "problem_description": "Automate daily reports with complex requirements",
                "conversation_history": [],
                "context_data": {},
                "current_step": "start",
                "current_status": "initialized",
                "context_complete": False,
                "requires_user_input": False,
                "retry_count": 0
            }
            
            # Execute workflow - should stop for user input
            result = await workflow.ainvoke(initial_state, config)
            
            # Should be interrupted waiting for user input
            assert result["requires_user_input"] == True
            assert "questions" in result
            assert len(result["questions"]) > 0
            assert result["current_step"] == "collect_context"

    @pytest.mark.asyncio
    async def test_workflow_resumption_after_user_input(
        self, temp_db_path, mock_llm_service
    ):
        """Test workflow resumption after receiving user input."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service), \
             patch('backend.app.agents.context_collector.llm_service', mock_llm_service), \
             patch('backend.app.agents.requirements_generator.llm_service', mock_llm_service):
            
            # Setup workflow state that's waiting for user input
            from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
            from backend.app.workflows.graph import get_compiled_workflow, get_workflow_config
            
            checkpointer = await create_checkpointer(temp_db_path)
            thread_id = "test-resume-thread"
            
            # State waiting for user input
            waiting_state = {
                "problem_description": "Automate invoice processing",
                "conversation_history": [],
                "context_data": {"technical_level": "beginner"},
                "current_step": "collect_context",
                "current_status": "awaiting_input",
                "context_complete": False,
                "requires_user_input": True,
                "questions": ["What file formats?", "How many invoices daily?"],
                "retry_count": 0,
                "progress_percentage": 40
            }
            
            await save_checkpoint(checkpointer, thread_id, waiting_state)
            
            # Mock user input processing
            mock_llm_service.generate_response.side_effect = [
                # Context collector processes user input
                {
                    "content": '{"context_sufficient": true, "analysis": "Sufficient context received"}',
                    "usage": {"tokens": 70}
                },
                # Requirements generator
                {
                    "content": "# Requirements\n## Functional Requirements\n- Process invoices",
                    "usage": {"tokens": 120}
                }
            ]
            
            # Resume workflow with user input
            workflow = await get_compiled_workflow(db_path=temp_db_path)
            config = get_workflow_config(thread_id)
            
            # Simulate user providing context
            resumed_state = {
                **waiting_state,
                "user_input": "I work with PDF invoices, about 50 per day",
                "context_data": {
                    **waiting_state["context_data"],
                    "file_format": "PDF",
                    "daily_volume": "50 invoices"
                },
                "requires_user_input": False,
                "context_complete": True
            }
            
            result = await workflow.ainvoke(resumed_state, config)
            
            # Should progress beyond context collection
            assert result["current_step"] != "collect_context"
            assert result["context_complete"] == True
            assert result["progress_percentage"] > 40

    @pytest.mark.asyncio
    async def test_workflow_error_recovery(self, temp_db_path, mock_llm_service):
        """Test workflow error handling and recovery."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            
            # First call fails, second succeeds
            mock_llm_service.generate_response.side_effect = [
                Exception("LLM service temporary failure"),
                {
                    "content": '{"problem_type": "automation", "complexity": "low", "solution_category": "SIMPLE_AUTOMATION"}',
                    "usage": {"tokens": 100}
                }
            ]
            
            from backend.app.workflows.graph import get_compiled_workflow, get_workflow_config
            from backend.app.database.checkpointer import create_new_workflow_session
            
            thread_id = create_new_workflow_session(
                user_id="test-error-recovery",
                db_path=temp_db_path
            )
            
            workflow = await get_compiled_workflow(db_path=temp_db_path)
            config = get_workflow_config(thread_id)
            
            initial_state = {
                "problem_description": "Test error recovery",
                "conversation_history": [],
                "context_data": {},
                "current_step": "start",
                "current_status": "initialized",
                "retry_count": 0
            }
            
            # First execution should handle error
            result = await workflow.ainvoke(initial_state, config)
            
            # Should either retry or be in error state
            assert result["retry_count"] >= 1 or result["current_status"] == "error"

    @pytest.mark.asyncio
    async def test_workflow_state_persistence(self, temp_db_path, mock_llm_service):
        """Test workflow state persistence across multiple invocations."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            
            mock_llm_service.generate_response.return_value = {
                "content": '{"problem_type": "automation", "complexity": "low", "solution_category": "SIMPLE_AUTOMATION"}',
                "usage": {"tokens": 100}
            }
            
            from backend.app.workflows.graph import get_compiled_workflow, get_workflow_config
            from backend.app.database.checkpointer import create_new_workflow_session, load_checkpoint, create_checkpointer
            
            thread_id = create_new_workflow_session(
                user_id="test-persistence",
                db_path=temp_db_path
            )
            
            workflow = await get_compiled_workflow(db_path=temp_db_path)
            config = get_workflow_config(thread_id)
            
            # First execution
            initial_state = {
                "problem_description": "Test persistence",
                "conversation_history": [],
                "context_data": {},
                "current_step": "start",
                "current_status": "initialized",
                "retry_count": 0
            }
            
            result1 = await workflow.ainvoke(initial_state, config)
            
            # Load state from database
            checkpointer = await create_checkpointer(temp_db_path)
            persisted_state = await load_checkpoint(checkpointer, thread_id)
            
            assert persisted_state is not None
            assert persisted_state["problem_description"] == "Test persistence"
            
            # Continue from persisted state
            result2 = await workflow.ainvoke(persisted_state, config)
            
            # Should maintain continuity
            assert result2["conversation_history"] == result1["conversation_history"]

    @pytest.mark.asyncio
    async def test_concurrent_workflows(self, temp_db_path, mock_llm_service):
        """Test multiple concurrent workflows."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            
            mock_llm_service.generate_response.return_value = {
                "content": '{"problem_type": "automation", "complexity": "low", "solution_category": "SIMPLE_AUTOMATION"}',
                "usage": {"tokens": 100}
            }
            
            from backend.app.workflows.graph import get_compiled_workflow, get_workflow_config
            from backend.app.database.checkpointer import create_new_workflow_session
            
            # Create multiple workflow sessions
            thread_ids = []
            for i in range(3):
                thread_id = create_new_workflow_session(
                    user_id=f"test-user-{i}",
                    db_path=temp_db_path
                )
                thread_ids.append(thread_id)
            
            workflow = await get_compiled_workflow(db_path=temp_db_path)
            
            # Execute workflows concurrently
            tasks = []
            for i, thread_id in enumerate(thread_ids):
                config = get_workflow_config(thread_id)
                initial_state = {
                    "problem_description": f"Test problem {i}",
                    "conversation_history": [],
                    "context_data": {},
                    "current_step": "start",
                    "current_status": "initialized"
                }
                task = workflow.ainvoke(initial_state, config)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # All workflows should execute independently
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    pytest.fail(f"Workflow {i} failed: {result}")
                assert result["problem_description"] == f"Test problem {i}"

    @pytest.mark.asyncio
    async def test_workflow_timeout_handling(self, temp_db_path, mock_llm_service):
        """Test workflow timeout handling."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            
            # Mock a very slow LLM response
            async def slow_response(*args, **kwargs):
                await asyncio.sleep(10)  # Simulate slow response
                return {"content": '{"result": "slow"}', "usage": {"tokens": 50}}
            
            mock_llm_service.generate_response.side_effect = slow_response
            
            from backend.app.workflows.graph import get_compiled_workflow, get_workflow_config
            from backend.app.database.checkpointer import create_new_workflow_session
            
            thread_id = create_new_workflow_session(
                user_id="test-timeout",
                db_path=temp_db_path
            )
            
            workflow = await get_compiled_workflow(db_path=temp_db_path)
            config = get_workflow_config(thread_id)
            
            initial_state = {
                "problem_description": "Test timeout",
                "conversation_history": [],
                "context_data": {},
                "current_step": "start",
                "current_status": "initialized"
            }
            
            # Execute with timeout
            try:
                result = await asyncio.wait_for(
                    workflow.ainvoke(initial_state, config),
                    timeout=5.0  # 5 second timeout
                )
                pytest.fail("Expected timeout but workflow completed")
            except asyncio.TimeoutError:
                # Expected behavior
                pass

    @pytest.mark.asyncio
    async def test_workflow_step_validation(self, temp_db_path):
        """Test workflow step validation and routing."""
        from backend.app.workflows.graph import create_workflow_graph
        
        # Create workflow without mocking to test actual routing
        workflow_graph = create_workflow_graph(enable_human_loop=False)
        
        # Verify all expected nodes exist
        expected_nodes = [
            "analyze_problem",
            "collect_context", 
            "generate_requirements",
            "design_solution",
            "create_guide"
        ]
        
        for node in expected_nodes:
            assert node in workflow_graph.nodes
        
        # Verify conditional edges exist
        assert "check_context_complete" in [edge.path for edge in workflow_graph.edges if hasattr(edge, 'path')]

    @pytest.mark.asyncio
    async def test_workflow_data_flow(self, temp_db_path, mock_agents):
        """Test data flow between workflow steps."""
        with patch('backend.app.agents.analyzer.analyze_problem', mock_agents["analyzer"]), \
             patch('backend.app.agents.context_collector.collect_context', mock_agents["context_collector"]), \
             patch('backend.app.agents.requirements_generator.generate_requirements', mock_agents["requirements_generator"]):
            
            from backend.app.workflows.graph import get_compiled_workflow, get_workflow_config
            from backend.app.database.checkpointer import create_new_workflow_session
            
            thread_id = create_new_workflow_session(
                user_id="test-data-flow",
                db_path=temp_db_path
            )
            
            workflow = await get_compiled_workflow(db_path=temp_db_path)
            config = get_workflow_config(thread_id)
            
            initial_state = {
                "problem_description": "Test data flow",
                "conversation_history": [],
                "context_data": {},
                "current_step": "start",
                "current_status": "initialized"
            }
            
            result = await workflow.ainvoke(initial_state, config)
            
            # Verify data flows through steps
            assert "problem_analysis" in result  # From analyzer
            assert "context_data" in result      # From context collector
            assert result["context_complete"] == True  # From context collector