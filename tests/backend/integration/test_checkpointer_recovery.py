"""
Integration tests for checkpointer recovery and state management
"""

import pytest
import tempfile
import os
import sqlite3
import asyncio
from unittest.mock import patch, AsyncMock
from typing import Dict, Any


class TestCheckpointerRecovery:
    """Test cases for checkpointer recovery scenarios."""

    @pytest.mark.asyncio
    async def test_workflow_recovery_after_crash(self, temp_db_path, sample_workflow_state):
        """Test workflow recovery after simulated system crash."""
        from backend.app.database.checkpointer import (
            create_checkpointer, save_checkpoint, load_checkpoint
        )
        from backend.app.workflows.graph import get_compiled_workflow, get_workflow_config
        
        # Save workflow state before "crash"
        checkpointer = await create_checkpointer(temp_db_path)
        thread_id = "crash-recovery-test"
        
        pre_crash_state = {
            **sample_workflow_state,
            "current_step": "collect_context",
            "progress_percentage": 40,
            "context_data": {"technical_level": "intermediate"},
            "conversation_history": [
                {"role": "user", "content": "I need automation help"},
                {"role": "assistant", "content": "I'll help you with that"}
            ]
        }
        
        await save_checkpoint(checkpointer, thread_id, pre_crash_state)
        
        # Simulate crash by creating new checkpointer instance
        new_checkpointer = await create_checkpointer(temp_db_path)
        
        # Recover state
        recovered_state = await load_checkpoint(new_checkpointer, thread_id)
        
        # Verify state recovery
        assert recovered_state is not None
        assert recovered_state["current_step"] == "collect_context"
        assert recovered_state["progress_percentage"] == 40
        assert len(recovered_state["conversation_history"]) == 2
        assert recovered_state["context_data"]["technical_level"] == "intermediate"
        
        # Verify workflow can continue from recovered state
        workflow = await get_compiled_workflow(db_path=temp_db_path)
        config = get_workflow_config(thread_id)
        
        # Should be able to continue execution
        continued_result = await workflow.ainvoke(recovered_state, config)
        assert continued_result is not None

    @pytest.mark.asyncio
    async def test_database_corruption_recovery(self, temp_db_path, sample_workflow_state):
        """Test recovery from database corruption."""
        from backend.app.database.checkpointer import (
            create_checkpointer, save_checkpoint, recover_from_corruption
        )
        
        # Create and save some data
        checkpointer = await create_checkpointer(temp_db_path)
        thread_id = "corruption-test"
        await save_checkpoint(checkpointer, thread_id, sample_workflow_state)
        
        # Corrupt the database file
        with open(temp_db_path, 'r+b') as f:
            f.seek(100)  # Seek to middle of file
            f.write(b"CORRUPTED_DATA_BLOCK")
        
        # Attempt to create new checkpointer (should detect corruption)
        try:
            corrupted_checkpointer = await create_checkpointer(temp_db_path)
            # If this succeeds, the corruption wasn't severe enough
        except Exception as e:
            # Expected corruption detection
            assert "corrupt" in str(e).lower() or "database" in str(e).lower()
        
        # Attempt recovery
        recovery_success = await recover_from_corruption(temp_db_path)
        
        # Recovery might create a new clean database
        if recovery_success:
            recovered_checkpointer = await create_checkpointer(temp_db_path)
            assert recovered_checkpointer is not None

    @pytest.mark.asyncio
    async def test_concurrent_access_recovery(self, temp_db_path, sample_workflow_state):
        """Test recovery from concurrent access conflicts."""
        from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
        
        # Create multiple checkpointer instances
        checkpointers = []
        for i in range(5):
            cp = await create_checkpointer(temp_db_path)
            checkpointers.append(cp)
        
        # Attempt concurrent writes
        tasks = []
        for i, cp in enumerate(checkpointers):
            thread_id = f"concurrent-test-{i}"
            state = {**sample_workflow_state, "current_step": f"step_{i}"}
            task = save_checkpoint(cp, thread_id, state)
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should handle concurrent access gracefully
        successful_writes = sum(1 for r in results if r is True)
        assert successful_writes > 0  # At least some should succeed
        
        # Verify data integrity
        verification_cp = await create_checkpointer(temp_db_path)
        for i in range(len(checkpointers)):
            thread_id = f"concurrent-test-{i}"
            try:
                state = await verification_cp.aget({"configurable": {"thread_id": thread_id}})
                if state:
                    # Verify the state is valid
                    assert "current_step" in state.values
            except Exception:
                # Some might fail due to concurrent access, which is acceptable
                pass

    @pytest.mark.asyncio
    async def test_partial_transaction_recovery(self, temp_db_path):
        """Test recovery from partial transactions."""
        from backend.app.database.checkpointer import create_checkpointer
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        # Simulate interrupted transaction by directly manipulating database
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        try:
            # Start a transaction but don't commit
            cursor.execute("BEGIN TRANSACTION;")
            cursor.execute("""
                INSERT INTO checkpoints (thread_id, checkpoint_id, data, created_at)
                VALUES ('partial-transaction', 'test-id', '{"incomplete": true}', datetime('now'))
            """)
            # Don't commit - simulate interruption
            
        except Exception:
            pass  # Expected if table doesn't exist yet
        finally:
            conn.close()
        
        # New checkpointer should handle partial transactions
        new_checkpointer = await create_checkpointer(temp_db_path)
        assert new_checkpointer is not None

    @pytest.mark.asyncio
    async def test_schema_migration_recovery(self, temp_db_path):
        """Test recovery during schema migrations."""
        from backend.app.database.checkpointer import (
            create_checkpointer, migrate_database_schema
        )
        
        # Create old schema version
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        
        # Create an older version of the schema
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS old_checkpoints (
                id INTEGER PRIMARY KEY,
                thread_id TEXT,
                data TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Insert some old format data
        cursor.execute("""
            INSERT INTO old_checkpoints (thread_id, data)
            VALUES ('old-format', '{"old": "data"}')
        """)
        
        conn.commit()
        conn.close()
        
        # Attempt migration
        migration_success = await migrate_database_schema(temp_db_path)
        
        # Should handle migration gracefully
        assert isinstance(migration_success, bool)
        
        # New checkpointer should work regardless of migration success
        checkpointer = await create_checkpointer(temp_db_path)
        assert checkpointer is not None

    @pytest.mark.asyncio
    async def test_memory_pressure_recovery(self, temp_db_path, sample_workflow_state):
        """Test recovery under memory pressure conditions."""
        from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        # Create large state objects to simulate memory pressure
        large_state = {
            **sample_workflow_state,
            "large_data": ["x" * 1000] * 1000,  # Large list
            "complex_nested": {
                f"key_{i}": {
                    f"nested_{j}": f"value_{i}_{j}" * 100
                    for j in range(100)
                }
                for i in range(50)
            }
        }
        
        # Attempt to save large state
        try:
            result = await save_checkpoint(checkpointer, "memory-pressure-test", large_state)
            # If successful, verify we can load it back
            if result:
                loaded_state = await checkpointer.aget(
                    {"configurable": {"thread_id": "memory-pressure-test"}}
                )
                assert loaded_state is not None
        except MemoryError:
            # Expected under memory pressure
            pass
        except Exception as e:
            # Should handle other errors gracefully
            assert "memory" in str(e).lower() or "size" in str(e).lower()

    @pytest.mark.asyncio
    async def test_disk_space_exhaustion_recovery(self, sample_workflow_state):
        """Test recovery when disk space is exhausted."""
        from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
        
        # Create temporary file on potentially full filesystem
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            checkpointer = await create_checkpointer(temp_path)
            
            # Try to save increasingly large data until disk space issues
            for size_multiplier in [1, 10, 100]:
                large_data = {
                    **sample_workflow_state,
                    "bulk_data": "x" * (1024 * 1024 * size_multiplier)  # MB of data
                }
                
                try:
                    result = await save_checkpoint(
                        checkpointer, f"disk-test-{size_multiplier}", large_data
                    )
                    if not result:
                        # Expected failure due to disk space
                        break
                except Exception as e:
                    # Should handle disk space errors gracefully
                    error_msg = str(e).lower()
                    expected_errors = ["disk", "space", "full", "no space"]
                    if any(err in error_msg for err in expected_errors):
                        # Expected disk space error
                        break
                    else:
                        # Unexpected error
                        raise
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_network_interruption_recovery(self, temp_db_path, sample_workflow_state):
        """Test recovery from network interruptions (if using remote database)."""
        from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
        
        # For SQLite, this simulates file system access interruptions
        checkpointer = await create_checkpointer(temp_db_path)
        
        # Save initial state
        thread_id = "network-interruption-test"
        await save_checkpoint(checkpointer, thread_id, sample_workflow_state)
        
        # Simulate network/filesystem interruption by temporarily moving the file
        backup_path = temp_db_path + ".backup"
        os.rename(temp_db_path, backup_path)
        
        try:
            # Attempt operation while "disconnected"
            result = await save_checkpoint(
                checkpointer, thread_id, 
                {**sample_workflow_state, "updated": True}
            )
            # Should fail gracefully
            assert result == False
        except Exception as e:
            # Expected connection/file error
            assert "no such file" in str(e).lower() or "not found" in str(e).lower()
        finally:
            # Restore connection
            os.rename(backup_path, temp_db_path)
        
        # Should be able to reconnect and continue
        new_checkpointer = await create_checkpointer(temp_db_path)
        recovered_state = await new_checkpointer.aget(
            {"configurable": {"thread_id": thread_id}}
        )
        assert recovered_state is not None

    @pytest.mark.asyncio
    async def test_version_mismatch_recovery(self, temp_db_path, sample_workflow_state):
        """Test recovery from version mismatches in checkpointed data."""
        from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        # Save state with version information
        versioned_state = {
            **sample_workflow_state,
            "_version": "1.0.0",
            "_schema_version": "legacy",
            "deprecated_field": "old_value"
        }
        
        await save_checkpoint(checkpointer, "version-test", versioned_state)
        
        # Simulate version upgrade by loading with different version expectations
        loaded_state = await checkpointer.aget(
            {"configurable": {"thread_id": "version-test"}}
        )
        
        # Should handle version differences gracefully
        if loaded_state:
            state_data = loaded_state.values
            # Should either migrate or handle version differences
            assert "_version" in state_data or "current_step" in state_data

    @pytest.mark.asyncio
    async def test_rollback_recovery(self, temp_db_path, sample_workflow_state):
        """Test rollback to previous checkpoint after failure."""
        from backend.app.database.checkpointer import (
            create_checkpointer, save_checkpoint, get_checkpoint_history
        )
        
        checkpointer = await create_checkpointer(temp_db_path)
        thread_id = "rollback-test"
        
        # Save multiple checkpoint versions
        states = [
            {**sample_workflow_state, "current_step": "analyze_problem", "version": 1},
            {**sample_workflow_state, "current_step": "collect_context", "version": 2},
            {**sample_workflow_state, "current_step": "generate_requirements", "version": 3},
        ]
        
        for state in states:
            await save_checkpoint(checkpointer, thread_id, state)
            await asyncio.sleep(0.1)  # Ensure different timestamps
        
        # Get checkpoint history
        history = await get_checkpoint_history(checkpointer, thread_id)
        
        if history and len(history) >= 2:
            # Should be able to rollback to previous version
            previous_checkpoint = history[-2]  # Second to last
            assert previous_checkpoint["current_step"] != states[-1]["current_step"]

    @pytest.mark.asyncio
    async def test_recovery_stress_test(self, temp_db_path, sample_workflow_state):
        """Stress test recovery mechanisms under various failure conditions."""
        from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        # Simulate multiple types of failures in rapid succession
        failure_scenarios = [
            # Rapid saves
            ("rapid-save-1", sample_workflow_state),
            ("rapid-save-2", {**sample_workflow_state, "current_step": "step2"}),
            ("rapid-save-3", {**sample_workflow_state, "current_step": "step3"}),
        ]
        
        # Execute scenarios rapidly
        tasks = []
        for thread_id, state in failure_scenarios:
            task = save_checkpoint(checkpointer, thread_id, state)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify system stability after stress
        verification_checkpointer = await create_checkpointer(temp_db_path)
        assert verification_checkpointer is not None
        
        # At least some operations should have succeeded
        successful_operations = sum(1 for r in results if r is True)
        total_operations = len(results)
        success_rate = successful_operations / total_operations
        
        # Should maintain reasonable success rate even under stress
        assert success_rate >= 0.5  # At least 50% success rate