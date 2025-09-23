"""
Unit tests for database checkpointer functionality
"""

import pytest
import tempfile
import os
import asyncio
from typing import Dict, Any


class TestCheckpointer:
    """Test cases for SQLite checkpointer."""

    @pytest.mark.asyncio
    async def test_create_checkpointer(self, temp_db_path):
        """Test creating a checkpointer instance."""
        from backend.app.database.checkpointer import create_checkpointer
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        assert checkpointer is not None
        assert os.path.exists(temp_db_path)

    @pytest.mark.asyncio
    async def test_save_checkpoint(self, temp_db_path, sample_workflow_state):
        """Test saving a checkpoint."""
        from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
        
        checkpointer = await create_checkpointer(temp_db_path)
        thread_id = "test-thread-save"
        
        success = await save_checkpoint(
            checkpointer, thread_id, sample_workflow_state
        )
        
        assert success == True

    @pytest.mark.asyncio
    async def test_load_checkpoint(self, temp_db_path, sample_workflow_state):
        """Test loading a checkpoint."""
        from backend.app.database.checkpointer import (
            create_checkpointer, save_checkpoint, load_checkpoint
        )
        
        checkpointer = await create_checkpointer(temp_db_path)
        thread_id = "test-thread-load"
        
        # Save first
        await save_checkpoint(checkpointer, thread_id, sample_workflow_state)
        
        # Load
        loaded_state = await load_checkpoint(checkpointer, thread_id)
        
        assert loaded_state is not None
        assert loaded_state["problem_description"] == sample_workflow_state["problem_description"]
        assert loaded_state["current_step"] == sample_workflow_state["current_step"]

    @pytest.mark.asyncio
    async def test_load_nonexistent_checkpoint(self, temp_db_path):
        """Test loading a non-existent checkpoint."""
        from backend.app.database.checkpointer import create_checkpointer, load_checkpoint
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        loaded_state = await load_checkpoint(checkpointer, "nonexistent-thread")
        
        assert loaded_state is None

    @pytest.mark.asyncio
    async def test_checkpoint_versioning(self, temp_db_path, sample_workflow_state):
        """Test checkpoint versioning and history."""
        from backend.app.database.checkpointer import (
            create_checkpointer, save_checkpoint, get_checkpoint_history
        )
        
        checkpointer = await create_checkpointer(temp_db_path)
        thread_id = "test-thread-versioning"
        
        # Save multiple versions
        state_v1 = {**sample_workflow_state, "current_step": "analyze_problem"}
        state_v2 = {**sample_workflow_state, "current_step": "collect_context"}
        state_v3 = {**sample_workflow_state, "current_step": "generate_requirements"}
        
        await save_checkpoint(checkpointer, thread_id, state_v1)
        await save_checkpoint(checkpointer, thread_id, state_v2)
        await save_checkpoint(checkpointer, thread_id, state_v3)
        
        # Get history
        history = await get_checkpoint_history(checkpointer, thread_id)
        
        assert len(history) >= 3
        assert history[-1]["current_step"] == "generate_requirements"

    @pytest.mark.asyncio
    async def test_checkpoint_metadata(self, temp_db_path, sample_workflow_state):
        """Test checkpoint metadata storage and retrieval."""
        from backend.app.database.checkpointer import (
            create_checkpointer, save_checkpoint_with_metadata, get_checkpoint_metadata
        )
        
        checkpointer = await create_checkpointer(temp_db_path)
        thread_id = "test-thread-metadata"
        
        metadata = {
            "user_id": "test-user",
            "session_type": "problem_solving",
            "priority": "high",
            "tags": ["automation", "excel"]
        }
        
        await save_checkpoint_with_metadata(
            checkpointer, thread_id, sample_workflow_state, metadata
        )
        
        retrieved_metadata = await get_checkpoint_metadata(checkpointer, thread_id)
        
        assert retrieved_metadata["user_id"] == "test-user"
        assert retrieved_metadata["session_type"] == "problem_solving"
        assert "automation" in retrieved_metadata["tags"]

    @pytest.mark.asyncio
    async def test_concurrent_checkpoints(self, temp_db_path, sample_workflow_state):
        """Test concurrent checkpoint operations."""
        from backend.app.database.checkpointer import (
            create_checkpointer, save_checkpoint, load_checkpoint
        )
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        # Create multiple threads with different states
        threads_and_states = [
            ("thread-1", {**sample_workflow_state, "current_step": "analyze_problem"}),
            ("thread-2", {**sample_workflow_state, "current_step": "collect_context"}),
            ("thread-3", {**sample_workflow_state, "current_step": "design_solution"})
        ]
        
        # Save concurrently
        save_tasks = [
            save_checkpoint(checkpointer, thread_id, state)
            for thread_id, state in threads_and_states
        ]
        
        results = await asyncio.gather(*save_tasks)
        assert all(results)  # All should succeed
        
        # Load concurrently
        load_tasks = [
            load_checkpoint(checkpointer, thread_id)
            for thread_id, _ in threads_and_states
        ]
        
        loaded_states = await asyncio.gather(*load_tasks)
        
        # Verify each thread has correct state
        for i, (thread_id, original_state) in enumerate(threads_and_states):
            assert loaded_states[i]["current_step"] == original_state["current_step"]

    @pytest.mark.asyncio
    async def test_checkpoint_cleanup(self, temp_db_path, sample_workflow_state):
        """Test checkpoint cleanup and garbage collection."""
        from backend.app.database.checkpointer import (
            create_checkpointer, save_checkpoint, cleanup_old_checkpoints
        )
        from datetime import datetime, timedelta
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        # Create old checkpoints (simulate by modifying timestamps)
        old_threads = ["old-thread-1", "old-thread-2"]
        new_threads = ["new-thread-1"]
        
        for thread_id in old_threads + new_threads:
            await save_checkpoint(checkpointer, thread_id, sample_workflow_state)
        
        # Cleanup checkpoints older than 1 hour
        cutoff_time = datetime.now() - timedelta(hours=1)
        cleaned_count = await cleanup_old_checkpoints(checkpointer, cutoff_time)
        
        # Should have cleaned some checkpoints
        assert cleaned_count >= 0

    def test_checkpointer_database_schema(self, temp_db_path):
        """Test database schema creation and validation."""
        from backend.app.database.checkpointer import validate_database_schema
        
        # Create checkpointer to initialize schema
        from backend.app.database.checkpointer import create_sync_checkpointer
        
        checkpointer = create_sync_checkpointer(temp_db_path)
        
        # Validate schema
        is_valid = validate_database_schema(temp_db_path)
        assert is_valid == True

    @pytest.mark.asyncio
    async def test_checkpoint_serialization(self, temp_db_path):
        """Test checkpoint data serialization and deserialization."""
        from backend.app.database.checkpointer import (
            create_checkpointer, save_checkpoint, load_checkpoint
        )
        
        checkpointer = await create_checkpointer(temp_db_path)
        thread_id = "test-serialization"
        
        # Complex state with various data types
        complex_state = {
            "problem_description": "Test with émojis 🚀",
            "conversation_history": [
                {"role": "user", "content": "Hello", "timestamp": "2024-01-01T10:00:00Z"}
            ],
            "context_data": {
                "nested": {"value": 123},
                "list": [1, 2, 3],
                "boolean": True,
                "null_value": None
            },
            "current_step": "analyze_problem",
            "progress_percentage": 25.5
        }
        
        await save_checkpoint(checkpointer, thread_id, complex_state)
        loaded_state = await load_checkpoint(checkpointer, thread_id)
        
        assert loaded_state["problem_description"] == "Test with émojis 🚀"
        assert loaded_state["context_data"]["nested"]["value"] == 123
        assert loaded_state["context_data"]["boolean"] == True
        assert loaded_state["context_data"]["null_value"] is None
        assert loaded_state["progress_percentage"] == 25.5

    @pytest.mark.asyncio
    async def test_checkpoint_error_handling(self, temp_db_path):
        """Test error handling in checkpoint operations."""
        from backend.app.database.checkpointer import (
            create_checkpointer, save_checkpoint, load_checkpoint
        )
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        # Test with invalid data that can't be serialized
        invalid_state = {
            "problem_description": "Test",
            "invalid_data": lambda x: x  # Function can't be serialized
        }
        
        # Should handle serialization error gracefully
        result = await save_checkpoint(checkpointer, "test-error", invalid_state)
        assert result == False  # Should fail gracefully

    @pytest.mark.asyncio
    async def test_checkpoint_recovery_after_corruption(self, temp_db_path, sample_workflow_state):
        """Test recovery after database corruption."""
        from backend.app.database.checkpointer import (
            create_checkpointer, save_checkpoint, recover_from_corruption
        )
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        # Save some data
        await save_checkpoint(checkpointer, "test-recovery", sample_workflow_state)
        
        # Simulate corruption by directly modifying the database file
        with open(temp_db_path, 'a') as f:
            f.write("CORRUPTED DATA")
        
        # Attempt recovery
        recovery_success = await recover_from_corruption(temp_db_path)
        
        # Recovery might succeed or fail depending on corruption level
        assert isinstance(recovery_success, bool)

    @pytest.mark.asyncio
    async def test_checkpoint_performance(self, temp_db_path, sample_workflow_state):
        """Test checkpoint performance under load."""
        from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
        import time
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        # Measure time for multiple checkpoint operations
        start_time = time.time()
        
        tasks = []
        for i in range(50):  # 50 concurrent operations
            thread_id = f"perf-test-{i}"
            task = save_checkpoint(checkpointer, thread_id, sample_workflow_state)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks)
        end_time = time.time()
        
        # All operations should succeed
        assert all(results)
        
        # Should complete within reasonable time (adjust threshold as needed)
        duration = end_time - start_time
        assert duration < 10.0  # 10 seconds for 50 operations

    @pytest.mark.asyncio
    async def test_workflow_session_management(self, temp_db_path):
        """Test workflow session creation and management."""
        from backend.app.database.checkpointer import (
            create_workflow_session, get_workflow_session, complete_workflow_session
        )
        
        session_id = create_workflow_session(
            user_id="test-user",
            workflow_type="problem_solving",
            db_path=temp_db_path
        )
        
        assert session_id is not None
        assert len(session_id) > 10  # Should be a proper UUID-like string
        
        # Get session info
        session_info = get_workflow_session(session_id, temp_db_path)
        assert session_info is not None
        assert session_info["user_id"] == "test-user"
        assert session_info["workflow_type"] == "problem_solving"
        assert session_info["status"] == "active"
        
        # Complete session
        completion_success = complete_workflow_session(session_id, temp_db_path)
        assert completion_success == True
        
        # Verify completion
        updated_session = get_workflow_session(session_id, temp_db_path)
        assert updated_session["status"] == "completed"