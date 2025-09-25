"""
SQLite 체크포인터 설정

비개발자 요약:
- 워크플로 진행 상태를 SQLite DB에 저장/복원하여, 중단되더라도 이어서 진행할 수 있게 합니다.
- 세션(사용자 작업) 관리, 무결성 점검, 청소(오래된 데이터 정리) 기능이 포함됩니다.
"""

import asyncio
import sqlite3
import logging
import threading
from pathlib import Path
from typing import Optional, Dict, Any, List, AsyncGenerator
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime, timedelta

try:
    # Try new import structure for langgraph-checkpoint-sqlite 2.x
    from langgraph_checkpoint_sqlite import SqliteSaver
    from langgraph_checkpoint_sqlite.aio import AsyncSqliteSaver
except ImportError:
    try:
        # Fallback to old import structure
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError:
        # Create mock classes for development
        class SqliteSaver:
            def __init__(self, connection):
                self.connection = connection
            @classmethod
            def from_conn_string(cls, conn_string):
                return cls(None)
        
        class AsyncSqliteSaver:
            def __init__(self, connection):
                self.connection = connection
            @classmethod
            def from_conn_string(cls, conn_string):
                return cls(None)

# Configure logging
logger = logging.getLogger(__name__)


class WorkflowSessionManager:
    """
    Manages workflow session metadata and status tracking
    """
    
    def __init__(self, connection: sqlite3.Connection):
        self.connection = connection
        self._setup_session_tables()
    
    def _setup_session_tables(self) -> None:
        """
        Create workflow session management tables
        """
        cursor = self.connection.cursor()
        
        # Workflow sessions table for tracking active workflows
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS workflow_sessions (
                thread_id TEXT PRIMARY KEY,
                status TEXT NOT NULL DEFAULT 'active',
                current_step TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                user_id TEXT,
                workflow_type TEXT,
                metadata TEXT
            )
        """)
        
        # Index for performance
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_sessions_status 
            ON workflow_sessions(status)
        """)
        
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_workflow_sessions_created 
            ON workflow_sessions(created_at)
        """)
        
        self.connection.commit()
    
    def create_session(self, thread_id: str, user_id: Optional[str] = None, 
                      workflow_type: str = "problem_solving") -> None:
        """
        Create a new workflow session
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO workflow_sessions 
            (thread_id, status, user_id, workflow_type, updated_at)
            VALUES (?, 'active', ?, ?, CURRENT_TIMESTAMP)
        """, (thread_id, user_id, workflow_type))
        self.connection.commit()
    
    def update_session_step(self, thread_id: str, current_step: str) -> None:
        """
        Update the current step of a workflow session
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            UPDATE workflow_sessions 
            SET current_step = ?, updated_at = CURRENT_TIMESTAMP 
            WHERE thread_id = ?
        """, (current_step, thread_id))
        self.connection.commit()
    
    def complete_session(self, thread_id: str) -> None:
        """
        Mark a workflow session as completed
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            UPDATE workflow_sessions 
            SET status = 'completed', completed_at = CURRENT_TIMESTAMP, 
                updated_at = CURRENT_TIMESTAMP 
            WHERE thread_id = ?
        """, (thread_id,))
        self.connection.commit()
    
    def get_session_info(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get workflow session information
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            SELECT thread_id, status, current_step, created_at, updated_at, 
                   completed_at, user_id, workflow_type, metadata
            FROM workflow_sessions 
            WHERE thread_id = ?
        """, (thread_id,))
        
        result = cursor.fetchone()
        if result:
            columns = [desc[0] for desc in cursor.description]
            return dict(zip(columns, result))
        return None
    
    def get_active_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all active workflow sessions
        """
        cursor = self.connection.cursor()
        
        if user_id:
            cursor.execute("""
                SELECT thread_id, status, current_step, created_at, updated_at,
                       user_id, workflow_type
                FROM workflow_sessions 
                WHERE status = 'active' AND user_id = ?
                ORDER BY updated_at DESC
            """, (user_id,))
        else:
            cursor.execute("""
                SELECT thread_id, status, current_step, created_at, updated_at,
                       user_id, workflow_type
                FROM workflow_sessions 
                WHERE status = 'active'
                ORDER BY updated_at DESC
            """)
        
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        return [dict(zip(columns, row)) for row in results]
    
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        """
        Clean up old completed sessions
        """
        cursor = self.connection.cursor()
        cursor.execute("""
            DELETE FROM workflow_sessions 
            WHERE status = 'completed' 
            AND completed_at < datetime('now', '-{} days')
        """.format(days_old))
        
        deleted_count = cursor.rowcount
        self.connection.commit()
        return deleted_count


class CheckpointerManager:
    """
    Enhanced SQLite checkpointer manager with full state persistence,
    concurrent session handling, and recovery capabilities
    """
    
    def __init__(self, db_path: str = "workflow_checkpoints.db", 
                 enable_wal: bool = True):
        """
        Initialize checkpointer manager
        
        Args:
            db_path: Path to SQLite database file
            enable_wal: Enable WAL mode for better concurrency
        """
        self.db_path = Path(db_path)
        self.enable_wal = enable_wal
        self.connection: Optional[sqlite3.Connection] = None
        self.checkpointer: Optional[SqliteSaver] = None
        self.session_manager: Optional[WorkflowSessionManager] = None
        self._lock = threading.RLock()
        
    def initialize_database(self) -> None:
        """
        Initialize SQLite database with proper configuration
        """
        with self._lock:
            try:
                # Ensure database directory exists
                self.db_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Create database connection with optimized settings
                self.connection = sqlite3.connect(
                    str(self.db_path), 
                    check_same_thread=False,
                    timeout=30.0  # 30 second timeout
                )
                
                # Configure SQLite for better concurrency and performance
                cursor = self.connection.cursor()
                
                if self.enable_wal:
                    # Enable WAL mode for better concurrency
                    cursor.execute("PRAGMA journal_mode=WAL")
                
                # Optimize SQLite settings
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.execute("PRAGMA cache_size=10000")
                cursor.execute("PRAGMA temp_store=memory")
                cursor.execute("PRAGMA mmap_size=268435456")  # 256MB
                
                self.connection.commit()
                
                # Initialize SqliteSaver with the connection
                self.checkpointer = SqliteSaver(self.connection)
                
                # Initialize session manager
                self.session_manager = WorkflowSessionManager(self.connection)
                
                logger.info(f"Database initialized at {self.db_path} with WAL mode: {self.enable_wal}")
                
            except Exception as e:
                logger.error(f"Failed to initialize database: {e}")
                if self.connection:
                    self.connection.close()
                    self.connection = None
                raise
    
    def get_checkpointer(self) -> SqliteSaver:
        """
        Get the SQLite checkpointer instance
        
        Returns:
            SqliteSaver instance for workflow checkpointing
            
        Raises:
            RuntimeError: If checkpointer is not initialized
        """
        if self.checkpointer is None:
            self.initialize_database()
        
        if self.checkpointer is None:
            raise RuntimeError("Failed to initialize checkpointer")
            
        return self.checkpointer
    
    def get_session_manager(self) -> WorkflowSessionManager:
        """
        Get the workflow session manager
        """
        if self.session_manager is None:
            self.initialize_database()
        
        if self.session_manager is None:
            raise RuntimeError("Failed to initialize session manager")
            
        return self.session_manager
    
    def create_workflow_session(self, thread_id: str, user_id: Optional[str] = None,
                               workflow_type: str = "problem_solving") -> None:
        """
        Create a new workflow session with tracking
        """
        session_manager = self.get_session_manager()
        session_manager.create_session(thread_id, user_id, workflow_type)
        logger.info(f"Created workflow session: {thread_id}")
    
    def update_workflow_step(self, thread_id: str, current_step: str) -> None:
        """
        Update the current step of a workflow
        """
        session_manager = self.get_session_manager()
        session_manager.update_session_step(thread_id, current_step)
    
    def complete_workflow(self, thread_id: str) -> None:
        """
        Mark a workflow as completed
        """
        session_manager = self.get_session_manager()
        session_manager.complete_session(thread_id)
        logger.info(f"Completed workflow session: {thread_id}")
    
    def cleanup_old_checkpoints(self, days_old: int = 30) -> Dict[str, int]:
        """
        Clean up old checkpoint data and completed sessions
        
        Args:
            days_old: Remove checkpoints older than this many days
            
        Returns:
            Dictionary with cleanup statistics
        """
        if self.connection is None:
            logger.warning("No database connection available for cleanup")
            return {"checkpoints_deleted": 0, "sessions_deleted": 0}
            
        cleanup_stats = {"checkpoints_deleted": 0, "sessions_deleted": 0}
        
        try:
            cursor = self.connection.cursor()
            
            # Clean up old checkpoints
            # Note: SqliteSaver creates its own tables, we need to check the actual column name
            try:
                # First try with 'ts' column (older versions)
                cursor.execute("""
                    DELETE FROM checkpoints 
                    WHERE ts < datetime('now', '-{} days')
                """.format(days_old))
            except sqlite3.OperationalError:
                try:
                    # Try with 'created_at' column (newer versions)
                    cursor.execute("""
                        DELETE FROM checkpoints 
                        WHERE created_at < datetime('now', '-{} days')
                    """.format(days_old))
                except sqlite3.OperationalError:
                    # If no timestamp column found, skip checkpoint cleanup
                    logger.warning("Could not find timestamp column in checkpoints table")
                    pass
            cleanup_stats["checkpoints_deleted"] = cursor.rowcount
            
            # Clean up old sessions through session manager
            if self.session_manager:
                cleanup_stats["sessions_deleted"] = self.session_manager.cleanup_old_sessions(days_old)
            
            self.connection.commit()
            
            logger.info(f"Cleanup completed: {cleanup_stats}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup old data: {e}")
            # Rollback any partial changes
            if self.connection:
                self.connection.rollback()
                
        return cleanup_stats
    
    def get_workflow_status(self, thread_id: str) -> Optional[Dict[str, Any]]:
        """
        Get comprehensive workflow status including session info and latest checkpoint
        
        Args:
            thread_id: Workflow thread identifier
            
        Returns:
            Comprehensive workflow status or None if not found
        """
        try:
            # Get session information
            session_info = None
            if self.session_manager:
                session_info = self.session_manager.get_session_info(thread_id)
            
            # Get latest checkpoint information
            checkpoint_info = None
            if self.checkpointer:
                try:
                    config = {"configurable": {"thread_id": thread_id}}
                    state = self.checkpointer.get(config)
                    if state:
                        checkpoint_info = {
                            "checkpoint_id": state.config.get("configurable", {}).get("checkpoint_id"),
                            "timestamp": getattr(state, "created_at", None),
                            "values": getattr(state, "values", {})
                        }
                except Exception as e:
                    logger.debug(f"Could not retrieve checkpoint for {thread_id}: {e}")
            
            if session_info or checkpoint_info:
                return {
                    "thread_id": thread_id,
                    "session_info": session_info,
                    "checkpoint_info": checkpoint_info,
                    "last_updated": datetime.now().isoformat()
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get workflow status for {thread_id}: {e}")
            return None
    
    def get_active_workflows(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all active workflows, optionally filtered by user
        """
        if self.session_manager is None:
            return []
            
        try:
            return self.session_manager.get_active_sessions(user_id)
        except Exception as e:
            logger.error(f"Failed to get active workflows: {e}")
            return []
    
    def validate_database_integrity(self) -> Dict[str, Any]:
        """
        Validate database integrity and return health status
        """
        if self.connection is None:
            return {"status": "error", "message": "No database connection"}
            
        try:
            cursor = self.connection.cursor()
            
            # Check database integrity
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            
            # Get database stats
            cursor.execute("SELECT COUNT(*) FROM checkpoints")
            checkpoint_count = cursor.fetchone()[0]
            
            cursor.execute("SELECT COUNT(*) FROM workflow_sessions")
            session_count = cursor.fetchone()[0]
            
            # Check for orphaned sessions (sessions without recent checkpoints)
            cursor.execute("""
                SELECT COUNT(*) FROM workflow_sessions ws
                LEFT JOIN checkpoints cp ON ws.thread_id = cp.thread_id
                WHERE ws.status = 'active' AND cp.thread_id IS NULL
            """)
            orphaned_sessions = cursor.fetchone()[0]
            
            return {
                "status": "healthy" if integrity_result == "ok" else "warning",
                "integrity_check": integrity_result,
                "checkpoint_count": checkpoint_count,
                "session_count": session_count,
                "orphaned_sessions": orphaned_sessions,
                "database_path": str(self.db_path),
                "wal_enabled": self.enable_wal
            }
            
        except Exception as e:
            logger.error(f"Database integrity check failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @contextmanager
    def transaction(self):
        """
        Context manager for database transactions with proper rollback
        """
        if self.connection is None:
            raise RuntimeError("No database connection")
            
        cursor = self.connection.cursor()
        try:
            cursor.execute("BEGIN")
            yield cursor
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
    
    def close(self) -> None:
        """
        Close database connection and cleanup resources
        """
        with self._lock:
            if self.connection:
                try:
                    # Ensure any pending transactions are committed
                    self.connection.commit()
                    self.connection.close()
                except Exception as e:
                    logger.warning(f"Error during database close: {e}")
                finally:
                    self.connection = None
                    self.checkpointer = None
                    self.session_manager = None
                    logger.info("Database connection closed")
    
    def __enter__(self):
        """Context manager entry"""
        self.initialize_database()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()


class AsyncCheckpointerManager:
    """
    Async version of CheckpointerManager for high-performance async workflows
    """
    
    def __init__(self, db_path: str = "workflow_checkpoints.db"):
        self.db_path = db_path
        self._checkpointer: Optional[AsyncSqliteSaver] = None
        self._lock = asyncio.Lock()
    
    async def get_async_checkpointer(self) -> AsyncSqliteSaver:
        """
        Get or create async SQLite checkpointer
        """
        async with self._lock:
            if self._checkpointer is None:
                self._checkpointer = AsyncSqliteSaver.from_conn_string(
                    f"sqlite:///{self.db_path}"
                )
                logger.info(f"Async checkpointer initialized for {self.db_path}")
            
            return self._checkpointer
    
    @asynccontextmanager
    async def get_checkpointer_context(self) -> AsyncGenerator[AsyncSqliteSaver, None]:
        """
        Async context manager for checkpointer
        """
        checkpointer = await self.get_async_checkpointer()
        try:
            yield checkpointer
        finally:
            # AsyncSqliteSaver handles its own cleanup
            pass
    
    async def close(self) -> None:
        """
        Close async checkpointer
        """
        async with self._lock:
            if self._checkpointer:
                # AsyncSqliteSaver handles cleanup automatically
                self._checkpointer = None
                logger.info("Async checkpointer closed")


# Global managers
_checkpointer_manager: Optional[CheckpointerManager] = None
_async_checkpointer_manager: Optional[AsyncCheckpointerManager] = None
_manager_lock = threading.Lock()


def get_checkpointer_manager(db_path: str = "workflow_checkpoints.db") -> CheckpointerManager:
    """
    Get the global checkpointer manager instance with thread safety
    
    Args:
        db_path: Database path (only used on first call)
        
    Returns:
        CheckpointerManager instance
    """
    global _checkpointer_manager
    
    with _manager_lock:
        if _checkpointer_manager is None:
            _checkpointer_manager = CheckpointerManager(db_path)
        
        return _checkpointer_manager


def get_async_checkpointer_manager(db_path: str = "workflow_checkpoints.db") -> AsyncCheckpointerManager:
    """
    Get the global async checkpointer manager instance
    
    Args:
        db_path: Database path (only used on first call)
        
    Returns:
        AsyncCheckpointerManager instance
    """
    global _async_checkpointer_manager
    
    with _manager_lock:
        if _async_checkpointer_manager is None:
            _async_checkpointer_manager = AsyncCheckpointerManager(db_path)
        
        return _async_checkpointer_manager


def get_checkpointer(db_path: str = "workflow_checkpoints.db") -> SqliteSaver:
    """
    Get the SQLite checkpointer for workflow persistence
    
    Args:
        db_path: Database path
        
    Returns:
        SqliteSaver instance
    """
    manager = get_checkpointer_manager(db_path)
    return manager.get_checkpointer()


async def get_async_checkpointer(db_path: str = "workflow_checkpoints.db") -> AsyncSqliteSaver:
    """
    Get the async SQLite checkpointer for workflow persistence
    
    Args:
        db_path: Database path
        
    Returns:
        AsyncSqliteSaver instance
    """
    manager = get_async_checkpointer_manager(db_path)
    return await manager.get_async_checkpointer()


def create_workflow_session(thread_id: str, user_id: Optional[str] = None,
                           workflow_type: str = "problem_solving",
                           db_path: str = "workflow_checkpoints.db") -> None:
    """
    Create a new workflow session with tracking
    """
    manager = get_checkpointer_manager(db_path)
    manager.create_workflow_session(thread_id, user_id, workflow_type)


def get_workflow_status(thread_id: str, db_path: str = "workflow_checkpoints.db") -> Optional[Dict[str, Any]]:
    """
    Get comprehensive workflow status
    """
    manager = get_checkpointer_manager(db_path)
    return manager.get_workflow_status(thread_id)


def cleanup_checkpoints(days_old: int = 30, db_path: str = "workflow_checkpoints.db") -> Dict[str, int]:
    """
    Clean up old checkpoints and sessions
    
    Args:
        days_old: Remove data older than this many days
        db_path: Database path
        
    Returns:
        Cleanup statistics
    """
    manager = get_checkpointer_manager(db_path)
    return manager.cleanup_old_checkpoints(days_old)


def validate_database_health(db_path: str = "workflow_checkpoints.db") -> Dict[str, Any]:
    """
    Validate database integrity and health
    """
    manager = get_checkpointer_manager(db_path)
    return manager.validate_database_integrity()


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    def test_sync_checkpointer():
        """Test synchronous checkpointer functionality"""
        print("Testing synchronous checkpointer...")
        
        try:
            # Test checkpointer initialization
            with CheckpointerManager("test_workflow.db") as manager:
                checkpointer = manager.get_checkpointer()
                print("OK - Sync checkpointer initialized successfully")
                
                # Test session management
                test_thread_id = "test_thread_123"
                manager.create_workflow_session(test_thread_id, "test_user")
                print(f"OK - Created workflow session: {test_thread_id}")
                
                # Test status retrieval
                status = manager.get_workflow_status(test_thread_id)
                print(f"OK - Retrieved workflow status: {status is not None}")
                
                # Test database health
                health = manager.validate_database_integrity()
                print(f"OK - Database health check: {health['status']}")
                
                # Test cleanup
                cleanup_stats = manager.cleanup_old_checkpoints(0)  # Clean everything for test
                print(f"OK - Cleanup completed: {cleanup_stats}")
                
        except Exception as e:
            print(f"ERROR - Sync test error: {e}")
    
    async def test_async_checkpointer():
        """Test asynchronous checkpointer functionality"""
        print("\nTesting asynchronous checkpointer...")
        
        try:
            manager = AsyncCheckpointerManager("test_async_workflow.db")
            
            async with manager.get_checkpointer_context() as checkpointer:
                print("OK - Async checkpointer initialized successfully")
                
                # Test basic functionality
                config = {"configurable": {"thread_id": "async_test_123"}}
                # Note: Actual checkpoint operations would require a proper LangGraph state
                print("OK - Async checkpointer ready for operations")
                
            await manager.close()
            print("OK - Async checkpointer closed successfully")
            
        except Exception as e:
            print(f"ERROR - Async test error: {e}")
    
    def test_convenience_functions():
        """Test convenience functions"""
        print("\nTesting convenience functions...")
        
        try:
            # Test global functions
            test_thread_id = "convenience_test_456"
            
            create_workflow_session(test_thread_id, "test_user", "problem_solving")
            print(f"OK - Created session via convenience function")
            
            status = get_workflow_status(test_thread_id)
            print(f"OK - Retrieved status via convenience function: {status is not None}")
            
            health = validate_database_health()
            print(f"OK - Database health via convenience function: {health['status']}")
            
            cleanup_stats = cleanup_checkpoints(0)
            print(f"OK - Cleanup via convenience function: {cleanup_stats}")
            
        except Exception as e:
            print(f"ERROR - Convenience function test error: {e}")
        finally:
            # Cleanup global manager
            global _checkpointer_manager
            if _checkpointer_manager:
                _checkpointer_manager.close()
                _checkpointer_manager = None
    
    # Run tests
    test_sync_checkpointer()
    asyncio.run(test_async_checkpointer())
    test_convenience_functions()
    
    print("\nOK - All tests completed")
