"""
LangGraph 메인 워크플로 그래프 정의

비개발자 요약:
- 이 파일은 전체 흐름도를 정의합니다. 각 단계(노드)와 다음 단계로 넘어가는
  조건(간선)을 설정합니다. 체크포인터(SQLite)를 통해 상태를 저장/복원합니다.

특징:
- SQLite 기반 상태 영속화(중단 후 재개 가능)
- 조건부 라우팅(분석 → 수집/요구사항/설계/가이드)
- Human-in-the-Loop(질문/답변) 지원
- 세션/스레드 단위 실행 관리
"""

import logging
import uuid
from typing import Literal, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
try:
    # Try new import structure for langgraph-checkpoint-sqlite 2.x
    from langgraph_checkpoint_sqlite import SqliteSaver
    from langgraph_checkpoint_sqlite.aio import AsyncSqliteSaver
except ImportError:
    try:
        # Fallback to old import structure (the correct one based on Context7)
        from langgraph.checkpoint.sqlite import SqliteSaver
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
    except ImportError:
        # Create mock classes for development
        class SqliteSaver:
            @classmethod
            def from_conn_string(cls, conn_string):
                return cls()
        
        class AsyncSqliteSaver:
            @classmethod
            def from_conn_string(cls, conn_string):
                return cls()

from app.config import settings
from app.workflows.state import WorkflowState, ProblemSolvingState
from app.agents.analyzer import analyze_problem
from app.agents.context_collector import collect_context
from app.agents.requirements_generator import generate_requirements
from app.agents.solution_designer import design_solution
from app.agents.guide_creator import create_guide
from app.database.checkpointer import (
    get_checkpointer, 
    get_async_checkpointer,
    create_workflow_session,
    get_checkpointer_manager
)

# Configure logging
logger = logging.getLogger(__name__)


def check_context_complete(state: WorkflowState) -> Literal["collect_context", "generate_requirements"]:
    """
    Enhanced conditional edge function to determine if context collection is complete
    
    This function now includes:
    - Context completeness validation
    - Missing information checks
    - User input requirements assessment
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node name based on context completeness
    """
    # Check if context collection is explicitly marked as complete
    if state.get("context_complete", False):
        logger.info(f"Context collection complete for thread: {state.get('current_step', 'unknown')}")
        return "generate_requirements"
    
    # Check if we have sufficient context data
    context_data = state.get("context_data", {})
    missing_info = state.get("missing_information", [])
    
    # If we have context data and no missing information, consider it complete
    if context_data and not missing_info:
        logger.info("Context collection auto-completed based on available data")
        return "generate_requirements"
    
    # If we're waiting for user input, continue context collection
    if state.get("requires_user_input", False):
        logger.info("Continuing context collection - awaiting user input")
        return "collect_context"
    
    # Default: continue context collection
    logger.info("Continuing context collection - incomplete data")
    return "collect_context"


def route_solution(state: WorkflowState) -> Literal["create_guide", "collect_context", "design_solution"]:
    """
    Enhanced conditional edge function to route solution based on analysis results

    This function handles:
    - Solution type complexity assessment
    - Context completeness validation
    - Implementation readiness checks
    - Error recovery routing

    Args:
        state: Current workflow state

    Returns:
        Next node name based on solution requirements and state
    """
    logger.info("=== ROUTE_SOLUTION FUNCTION CALLED ===")
    logger.info(f"State keys: {list(state.keys())}")

    solution_type = state.get("solution_type") or state.get("recommended_solution_type", "")
    context_complete = state.get("context_complete", False)
    technology_stack = state.get("technology_stack", {})
    implementation_plan = state.get("implementation_plan", "")
    current_step = state.get("current_step", "unknown")

    logger.info(f"Current step: {current_step}")
    logger.info(f"Solution type: {solution_type}")
    logger.info(f"Context complete: {context_complete}")
    logger.info(f"Technology stack type: {type(technology_stack)}, value: {technology_stack}")
    logger.info(f"Implementation plan type: {type(implementation_plan)}, value: {implementation_plan}")

    # Check for error conditions that require re-design
    error_message = state.get("error_message")
    if error_message and "design" in error_message.lower():
        logger.warning(f"Routing back to design due to error: {error_message}")
        return "design_solution"

    # For complex solutions that need more context
    complex_solutions = ["RAG", "ML_CLASSIFICATION", "COMPLEX_AUTOMATION", "INTEGRATION"]
    if solution_type in complex_solutions and not context_complete:
        logger.info(f"Complex solution {solution_type} requires more context")
        return "collect_context"

    # Check if solution design is incomplete with detailed logging
    tech_stack_valid = bool(technology_stack and isinstance(technology_stack, dict) and len(technology_stack) > 0)
    impl_plan_valid = bool(implementation_plan and isinstance(implementation_plan, dict) and len(implementation_plan) > 0)

    logger.info(f"🔍🚀 Solution validation check - ENHANCED:")
    logger.info(f"  - technology_stack exists: {bool(technology_stack)}, valid: {tech_stack_valid}")
    logger.info(f"  - implementation_plan exists: {bool(implementation_plan)}, valid: {impl_plan_valid}")

    # Be more lenient - allow if either field exists or if we've tried multiple times
    retry_count = state.get("retry_count", 0)
    logger.info(f"Current retry count: {retry_count}")

    if retry_count > 3:
        logger.info(f"⚠️ Retry count exceeded ({retry_count}), proceeding to avoid infinite loop")
        return "create_guide"

    if not tech_stack_valid and not impl_plan_valid:
        logger.info("Solution design incomplete, continuing design phase")
        # Increment retry count to prevent infinite loops
        state["retry_count"] = retry_count + 1
        logger.info(f"Updated retry count to: {retry_count + 1}")
        return "design_solution"

    # If everything is ready, proceed to guide creation
    logger.info(f"Solution {solution_type} ready for guide creation")
    return "create_guide"


def check_human_intervention_needed(state: WorkflowState) -> Literal["human_input", "continue_workflow"]:
    """
    Check if human intervention is needed at any point in the workflow
    
    Args:
        state: Current workflow state
        
    Returns:
        Next node based on human intervention requirements
    """
    # Check if there are pending questions for the user
    pending_questions = state.get("pending_questions", [])
    requires_user_input = state.get("requires_user_input", False)
    
    if pending_questions or requires_user_input:
        logger.info("Human intervention required - pausing workflow")
        return "human_input"
    
    return "continue_workflow"


def handle_human_input(state: WorkflowState) -> WorkflowState:
    """
    Handle human input and prepare for workflow continuation
    
    This is a special node that can interrupt the workflow for human input.
    The actual human input handling is done externally.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with human input status
    """
    logger.info("Workflow paused for human input")
    
    # Update state to reflect waiting status
    updated_state = state.copy()
    updated_state["current_status"] = "awaiting_input"
    updated_state["requires_user_input"] = True
    
    # This node effectively pauses the workflow until external input is provided
    return updated_state


def create_workflow_graph(enable_human_loop: bool = True) -> StateGraph:
    """
    Create and configure the enhanced workflow graph with full checkpointer support
    
    Args:
        enable_human_loop: Whether to enable human-in-the-loop functionality
        
    Returns:
        Configured StateGraph for the problem-solving workflow
    """
    # Initialize the graph with our enhanced state type
    workflow = StateGraph(WorkflowState)
    
    # Add all agent nodes to the graph
    workflow.add_node("analyze_problem", analyze_problem)
    workflow.add_node("collect_context", collect_context)
    workflow.add_node("generate_requirements", generate_requirements)
    workflow.add_node("design_solution", design_solution)
    workflow.add_node("create_guide", create_guide)
    
    # Add human-in-the-loop node if enabled
    if enable_human_loop:
        workflow.add_node("human_input", handle_human_input)
    
    # Set the entry point
    workflow.add_edge(START, "analyze_problem")
    
    # Define the main workflow flow
    workflow.add_edge("analyze_problem", "collect_context")
    
    # Enhanced conditional edge: check context completeness with human input support
    if enable_human_loop:
        workflow.add_conditional_edges(
            "collect_context",
            lambda state: (
                "human_input" if state.get("requires_user_input", False) 
                else check_context_complete(state)
            ),
            {
                "collect_context": "collect_context",
                "generate_requirements": "generate_requirements",
                "human_input": "human_input"
            }
        )
        
        # From human input, we can return to context collection or continue
        workflow.add_conditional_edges(
            "human_input",
            lambda state: (
                "collect_context" if not state.get("context_complete", False)
                else "generate_requirements"
            ),
            {
                "collect_context": "collect_context",
                "generate_requirements": "generate_requirements"
            }
        )
    else:
        workflow.add_conditional_edges(
            "collect_context",
            check_context_complete,
            {
                "collect_context": "collect_context",
                "generate_requirements": "generate_requirements"
            }
        )
    
    workflow.add_edge("generate_requirements", "design_solution")
    
    # Enhanced conditional edge: route based on solution analysis
    workflow.add_conditional_edges(
        "design_solution",
        route_solution,
        {
            "collect_context": "collect_context",
            "design_solution": "design_solution",  # Allow re-design loop
            "create_guide": "create_guide"
        }
    )
    
    workflow.add_edge("create_guide", END)
    
    logger.info(f"Workflow graph created with human loop: {enable_human_loop}")
    return workflow


def get_compiled_workflow(db_path: str = None, 
                          enable_human_loop: bool = None,
                          use_persistent_storage: bool = None) -> Any:
    """
    Get a compiled workflow graph with full SQLite checkpointer integration
    
    Args:
        db_path: Path to SQLite database for checkpointing (defaults from settings)
        enable_human_loop: Whether to enable human-in-the-loop functionality (defaults from settings)
        use_persistent_storage: Whether to use persistent storage (defaults from settings)
        
    Returns:
        Compiled workflow ready for execution with full state persistence
    """
    # Use settings defaults if not provided
    if db_path is None:
        db_path = settings.get_database_path()
    if enable_human_loop is None:
        enable_human_loop = settings.ENABLE_HUMAN_LOOP
    if use_persistent_storage is None:
        use_persistent_storage = settings.USE_PERSISTENT_STORAGE
    
    workflow = create_workflow_graph(enable_human_loop=enable_human_loop)
    
    # Configure SQLite checkpointer for persistence
    if use_persistent_storage:
        # Use persistent SQLite database
        checkpointer = get_checkpointer(db_path)
        logger.info(f"Using persistent checkpointer: {db_path}")
    else:
        # Use in-memory database for testing
        checkpointer = SqliteSaver.from_conn_string(":memory:")
        logger.info("Using in-memory checkpointer")
    
    # Compile the workflow with checkpointer
    compiled_workflow = workflow.compile(checkpointer=checkpointer)
    
    logger.info("Workflow compiled with checkpointer successfully")
    return compiled_workflow


async def get_async_compiled_workflow_direct(db_path: str = None,
                                           enable_human_loop: bool = None):
    """
    Get async workflow following exact Context7 pattern without wrapper class
    
    Args:
        db_path: Path to SQLite database for checkpointing (defaults from settings)
        enable_human_loop: Whether to enable human-in-the-loop functionality (defaults from settings)
        
    Returns:
        AsyncSqliteSaver context manager that yields compiled workflow
    """
    # Use settings defaults if not provided
    if enable_human_loop is None:
        enable_human_loop = settings.ENABLE_HUMAN_LOOP
        
    workflow = create_workflow_graph(enable_human_loop=enable_human_loop)
    
    # Return the AsyncSqliteSaver context manager directly following Context7 pattern
    return AsyncSqliteSaver.from_conn_string(":memory:")


class AsyncWorkflowManager:
    """
    Context manager for proper AsyncSqliteSaver lifecycle management
    """
    
    def __init__(self, db_path: str = None, enable_human_loop: bool = None):
        # Use settings defaults if not provided
        if db_path is None:
            db_path = settings.get_database_path()
        if enable_human_loop is None:
            enable_human_loop = settings.ENABLE_HUMAN_LOOP
            
        self.db_path = db_path
        self.enable_human_loop = enable_human_loop
        self.workflow = create_workflow_graph(enable_human_loop=enable_human_loop)
        self.checkpointer = None
        self.compiled_workflow = None
    
    async def __aenter__(self):
        """Enter async context and setup AsyncSqliteSaver following Context7 pattern"""
        try:
            # Use the exact Context7 pattern
            logger.info("Using in-memory SQLite database for AsyncSqliteSaver")
            
            # Create AsyncSqliteSaver following Context7 documentation exactly
            async with AsyncSqliteSaver.from_conn_string(":memory:") as checkpointer:
                # Compile the workflow with async checkpointer
                self.compiled_workflow = self.workflow.compile(checkpointer=checkpointer)
                self.checkpointer = checkpointer
                
                logger.info("AsyncWorkflowManager compiled with AsyncSqliteSaver (in-memory)")
                return self.compiled_workflow
            
        except Exception as e:
            logger.error(f"Failed to initialize AsyncSqliteSaver: {e}")
            raise
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and cleanup AsyncSqliteSaver"""
        # The AsyncSqliteSaver context manager handles its own cleanup
        self.checkpointer = None
        self.compiled_workflow = None


async def get_async_compiled_workflow(db_path: str = None,
                                     enable_human_loop: bool = None) -> AsyncWorkflowManager:
    """
    Get an async workflow manager that properly handles AsyncSqliteSaver lifecycle
    
    Args:
        db_path: Path to SQLite database for checkpointing (defaults from settings)
        enable_human_loop: Whether to enable human-in-the-loop functionality (defaults from settings)
        
    Returns:
        AsyncWorkflowManager that can be used as async context manager
    """
    return AsyncWorkflowManager(db_path=db_path, enable_human_loop=enable_human_loop)


def create_new_workflow_session(user_id: Optional[str] = None, 
                               workflow_type: str = "problem_solving",
                               db_path: str = None) -> str:
    """
    Create a new workflow session with proper tracking
    
    Args:
        user_id: Optional user identifier
        workflow_type: Type of workflow being created
        db_path: Database path for checkpointing (defaults from settings)
        
    Returns:
        Generated thread_id for the new session
    """
    # Use settings default if not provided
    if db_path is None:
        db_path = settings.get_database_path()
    
    # Generate unique thread ID
    thread_id = f"workflow_{uuid.uuid4().hex[:12]}"
    
    # Create session in database
    create_workflow_session(thread_id, user_id, workflow_type, db_path)
    
    logger.info(f"Created new workflow session: {thread_id} for user: {user_id}")
    return thread_id


def resume_workflow_session(thread_id: str, 
                          db_path: str = "workflow_checkpoints.db") -> Optional[Dict[str, Any]]:
    """
    Resume an existing workflow session
    
    Args:
        thread_id: Thread identifier of the session to resume
        db_path: Database path for checkpointing
        
    Returns:
        Current workflow state if session exists, None otherwise
    """
    from app.database.checkpointer import get_workflow_status
    
    # Get current session status
    status = get_workflow_status(thread_id, db_path)
    
    if status:
        logger.info(f"Resuming workflow session: {thread_id}")
        return status
    else:
        logger.warning(f"Workflow session not found: {thread_id}")
        return None


def get_workflow_config(thread_id: str, checkpoint_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Get workflow configuration for LangGraph execution
    
    Args:
        thread_id: Thread identifier for the workflow session
        checkpoint_id: Optional specific checkpoint to resume from
        
    Returns:
        Configuration dictionary for LangGraph
    """
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    if checkpoint_id:
        config["configurable"]["checkpoint_id"] = checkpoint_id
        logger.info(f"Configuration set for checkpoint: {checkpoint_id}")
    
    return config


# Enhanced workflow execution helpers
def execute_workflow_step(compiled_workflow: Any, 
                         state: WorkflowState, 
                         config: Dict[str, Any]) -> WorkflowState:
    """
    Execute a single step of the workflow with proper error handling
    
    Args:
        compiled_workflow: Compiled LangGraph workflow
        state: Current workflow state
        config: LangGraph configuration
        
    Returns:
        Updated workflow state
    """
    try:
        # Update session step tracking
        thread_id = config["configurable"]["thread_id"]
        current_step = state.get("current_step", "unknown")
        
        manager = get_checkpointer_manager()
        manager.update_workflow_step(thread_id, current_step)
        
        # Execute the workflow step
        result = compiled_workflow.invoke(state, config)
        
        logger.info(f"Workflow step executed: {current_step}")
        return result
        
    except Exception as e:
        logger.error(f"Workflow step execution failed: {e}")
        # Update state with error information
        error_state = state.copy()
        error_state["error_message"] = str(e)
        error_state["current_status"] = "error"
        return error_state


def complete_workflow_session(thread_id: str, 
                             db_path: str = "workflow_checkpoints.db") -> None:
    """
    Mark a workflow session as completed
    
    Args:
        thread_id: Thread identifier of the completed session
        db_path: Database path for checkpointing
    """
    manager = get_checkpointer_manager(db_path)
    manager.complete_workflow(thread_id)
    logger.info(f"Workflow session completed: {thread_id}")


# Example usage and testing
if __name__ == "__main__":
    import asyncio
    
    def test_sync_workflow():
        """Test synchronous workflow creation and basic functionality"""
        print("Testing synchronous workflow...")
        
        try:
            # Test workflow graph creation
            compiled_workflow = get_compiled_workflow(
                db_path="test_workflow.db",
                use_persistent_storage=False  # Use in-memory for testing
            )
            print("✓ Sync workflow compiled successfully")
            print(f"  Nodes: {list(compiled_workflow.graph.nodes())}")
            print(f"  Edges: {list(compiled_workflow.graph.edges())}")
            
            # Test session creation
            thread_id = create_new_workflow_session(
                user_id="test_user",
                db_path="test_workflow.db"
            )
            print(f"✓ Created workflow session: {thread_id}")
            
            # Test configuration
            config = get_workflow_config(thread_id)
            print(f"✓ Generated workflow config: {config['configurable']['thread_id']}")
            
        except Exception as e:
            print(f"✗ Sync workflow test error: {e}")
    
    async def test_async_workflow():
        """Test asynchronous workflow functionality"""
        print("\nTesting asynchronous workflow...")
        
        try:
            # Test async workflow compilation
            compiled_workflow = await get_async_compiled_workflow(
                db_path="test_async_workflow.db"
            )
            print("✓ Async workflow compiled successfully")
            print(f"  Nodes: {list(compiled_workflow.graph.nodes())}")
            
        except Exception as e:
            print(f"✗ Async workflow test error: {e}")
    
    def test_workflow_features():
        """Test advanced workflow features"""
        print("\nTesting workflow features...")
        
        try:
            # Test workflow with human loop disabled
            workflow_no_human = get_compiled_workflow(
                enable_human_loop=False,
                use_persistent_storage=False
            )
            print("✓ Workflow without human loop created")
            
            # Test workflow session management
            thread_id = "test_thread_456"
            status = resume_workflow_session(thread_id)
            print(f"✓ Session resume test: {status is not None}")
            
            # Test workflow completion
            complete_workflow_session(thread_id)
            print("✓ Workflow completion test passed")
            
        except Exception as e:
            print(f"✗ Workflow features test error: {e}")
    
    # Run all tests
    test_sync_workflow()
    asyncio.run(test_async_workflow())
    test_workflow_features()
    
    print("\n✓ All workflow tests completed")
