"""
Unit tests for workflow state management
"""

import pytest
from typing import Dict, Any


class TestWorkflowState:
    """Test cases for workflow state management."""

    def test_initial_state_creation(self):
        """Test creation of initial workflow state."""
        from backend.app.workflows.state import create_initial_state
        
        problem_description = "Automate daily reports"
        user_context = {"technical_level": "beginner"}
        
        state = create_initial_state(problem_description, user_context)
        
        assert state["problem_description"] == problem_description
        assert state["user_context"] == user_context
        assert state["current_step"] == "start"
        assert state["current_status"] == "initialized"
        assert state["context_complete"] == False
        assert state["requires_user_input"] == False
        assert state["retry_count"] == 0
        assert state["progress_percentage"] == 0
        assert state["conversation_history"] == []

    def test_state_validation_valid(self):
        """Test validation of valid workflow state."""
        from backend.app.workflows.state import validate_workflow_state
        
        valid_state = {
            "problem_description": "Test problem",
            "conversation_history": [],
            "context_data": {},
            "current_step": "analyze_problem",
            "current_status": "running",
            "context_complete": False,
            "requires_user_input": False,
            "retry_count": 0,
            "progress_percentage": 25
        }
        
        assert validate_workflow_state(valid_state) == True

    def test_state_validation_invalid(self):
        """Test validation of invalid workflow state."""
        from backend.app.workflows.state import validate_workflow_state
        
        invalid_states = [
            # Missing required fields
            {"problem_description": "Test"},
            # Invalid data types
            {
                "problem_description": 123,  # Should be string
                "conversation_history": [],
                "current_step": "start"
            },
            # Invalid step name
            {
                "problem_description": "Test",
                "current_step": "invalid_step",
                "current_status": "running"
            }
        ]
        
        for invalid_state in invalid_states:
            assert validate_workflow_state(invalid_state) == False

    def test_state_update_progress(self):
        """Test updating state progress."""
        from backend.app.workflows.state import update_progress
        
        state = {
            "current_step": "analyze_problem",
            "progress_percentage": 0
        }
        
        updated_state = update_progress(state, "collect_context", 40)
        
        assert updated_state["current_step"] == "collect_context"
        assert updated_state["progress_percentage"] == 40

    def test_state_add_conversation_entry(self):
        """Test adding conversation history entry."""
        from backend.app.workflows.state import add_conversation_entry
        
        state = {"conversation_history": []}
        
        updated_state = add_conversation_entry(
            state, "user", "I need help with automation"
        )
        
        assert len(updated_state["conversation_history"]) == 1
        entry = updated_state["conversation_history"][0]
        assert entry["role"] == "user"
        assert entry["content"] == "I need help with automation"
        assert "timestamp" in entry

    def test_state_merge_context_data(self):
        """Test merging context data."""
        from backend.app.workflows.state import merge_context_data
        
        state = {
            "context_data": {
                "technical_level": "beginner",
                "environment": "Windows"
            }
        }
        
        new_context = {
            "file_format": "Excel",
            "environment": "Linux"  # Should override
        }
        
        updated_state = merge_context_data(state, new_context)
        
        assert updated_state["context_data"]["technical_level"] == "beginner"
        assert updated_state["context_data"]["environment"] == "Linux"
        assert updated_state["context_data"]["file_format"] == "Excel"

    def test_state_set_error(self):
        """Test setting error state."""
        from backend.app.workflows.state import set_error_state
        
        state = {
            "current_status": "running",
            "retry_count": 0,
            "error": None
        }
        
        error_message = "LLM service unavailable"
        updated_state = set_error_state(state, error_message)
        
        assert updated_state["current_status"] == "error"
        assert updated_state["error"] == error_message
        assert updated_state["retry_count"] == 1

    def test_state_clear_error(self):
        """Test clearing error state."""
        from backend.app.workflows.state import clear_error_state
        
        state = {
            "current_status": "error",
            "error": "Previous error",
            "retry_count": 2
        }
        
        updated_state = clear_error_state(state)
        
        assert updated_state["current_status"] == "running"
        assert updated_state["error"] is None
        # retry_count should remain for tracking

    def test_state_step_transitions(self):
        """Test valid state step transitions."""
        from backend.app.workflows.state import is_valid_transition
        
        valid_transitions = [
            ("start", "analyze_problem"),
            ("analyze_problem", "collect_context"),
            ("collect_context", "generate_requirements"),
            ("generate_requirements", "design_solution"),
            ("design_solution", "create_guide"),
            ("create_guide", "complete")
        ]
        
        for from_step, to_step in valid_transitions:
            assert is_valid_transition(from_step, to_step) == True
        
        invalid_transitions = [
            ("start", "create_guide"),  # Skip steps
            ("complete", "analyze_problem"),  # Backward
            ("invalid_step", "analyze_problem")  # Invalid step
        ]
        
        for from_step, to_step in invalid_transitions:
            assert is_valid_transition(from_step, to_step) == False

    def test_state_serialization(self):
        """Test state serialization and deserialization."""
        from backend.app.workflows.state import serialize_state, deserialize_state
        
        original_state = {
            "problem_description": "Test problem",
            "conversation_history": [
                {"role": "user", "content": "Hello", "timestamp": "2024-01-01T10:00:00Z"}
            ],
            "context_data": {"level": "beginner"},
            "current_step": "analyze_problem",
            "progress_percentage": 25
        }
        
        # Serialize
        serialized = serialize_state(original_state)
        assert isinstance(serialized, str)
        
        # Deserialize
        deserialized = deserialize_state(serialized)
        assert deserialized == original_state

    def test_state_deep_copy(self):
        """Test deep copying of state."""
        from backend.app.workflows.state import copy_state
        
        original_state = {
            "context_data": {"nested": {"value": 123}},
            "conversation_history": [{"role": "user", "content": "test"}]
        }
        
        copied_state = copy_state(original_state)
        
        # Modify copied state
        copied_state["context_data"]["nested"]["value"] = 456
        copied_state["conversation_history"].append({"role": "assistant", "content": "response"})
        
        # Original should be unchanged
        assert original_state["context_data"]["nested"]["value"] == 123
        assert len(original_state["conversation_history"]) == 1

    def test_state_get_step_index(self):
        """Test getting step index for progress calculation."""
        from backend.app.workflows.state import get_step_index
        
        step_indices = {
            "start": 0,
            "analyze_problem": 1,
            "collect_context": 2,
            "generate_requirements": 3,
            "design_solution": 4,
            "create_guide": 5,
            "complete": 6
        }
        
        for step, expected_index in step_indices.items():
            assert get_step_index(step) == expected_index

    def test_state_calculate_progress_percentage(self):
        """Test progress percentage calculation."""
        from backend.app.workflows.state import calculate_progress_percentage
        
        test_cases = [
            ("start", 0),
            ("analyze_problem", 17),  # 1/6 * 100 ≈ 17
            ("collect_context", 33),   # 2/6 * 100 ≈ 33
            ("generate_requirements", 50),  # 3/6 * 100 = 50
            ("design_solution", 67),   # 4/6 * 100 ≈ 67
            ("create_guide", 83),      # 5/6 * 100 ≈ 83
            ("complete", 100)
        ]
        
        for step, expected_percentage in test_cases:
            actual_percentage = calculate_progress_percentage(step)
            assert abs(actual_percentage - expected_percentage) <= 1  # Allow 1% variance

    def test_state_context_completeness_check(self):
        """Test context completeness checking."""
        from backend.app.workflows.state import is_context_complete
        
        complete_contexts = [
            {
                "technical_level": "beginner",
                "environment": "Windows",
                "file_format": "Excel",
                "daily_volume": "10 files"
            },
            {
                "technical_level": "advanced",
                "solution_preference": "cloud_based",
                "budget": "unlimited"
            }
        ]
        
        incomplete_contexts = [
            {"technical_level": "beginner"},  # Missing key info
            {},  # Empty
            {"random_field": "value"}  # Irrelevant info only
        ]
        
        for context in complete_contexts:
            state = {"context_data": context}
            assert is_context_complete(state) == True
        
        for context in incomplete_contexts:
            state = {"context_data": context}
            assert is_context_complete(state) == False

    def test_state_cleanup_expired_data(self):
        """Test cleanup of expired or stale data."""
        from backend.app.workflows.state import cleanup_expired_data
        from datetime import datetime, timedelta
        
        # Create state with old timestamp
        old_timestamp = (datetime.now() - timedelta(hours=25)).isoformat()
        
        state = {
            "temporary_data": {"key": "value"},
            "last_activity": old_timestamp,
            "cache": {"expired_entry": "old_data"}
        }
        
        cleaned_state = cleanup_expired_data(state)
        
        # Should remove expired temporary data
        assert "temporary_data" not in cleaned_state
        assert "cache" not in cleaned_state
        # Should keep essential data
        assert "last_activity" in cleaned_state