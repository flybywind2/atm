"""
Test script for workflow checkpointer integration without agent dependencies
"""

import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Mock agent functions for testing
def mock_analyze_problem(state: Dict[str, Any]) -> Dict[str, Any]:
    """Mock problem analysis agent"""
    logger.info("Mock: Analyzing problem")
    return {
        **state,
        "current_step": "analyze_problem",
        "problem_analysis": {"type": "simple", "complexity": "low"},
        "current_status": "analyzing"
    }

def mock_collect_context(state: Dict[str, Any]) -> Dict[str, Any]:
    """Mock context collection agent"""
    logger.info("Mock: Collecting context")
    return {
        **state,
        "current_step": "collect_context",
        "context_data": {"user_needs": "basic automation"},
        "context_complete": True,
        "current_status": "context_collected"
    }

def mock_generate_requirements(state: Dict[str, Any]) -> Dict[str, Any]:
    """Mock requirements generation agent"""
    logger.info("Mock: Generating requirements")
    return {
        **state,
        "current_step": "generate_requirements",
        "requirements_doc": "Simple automation requirements",
        "current_status": "requirements_generated"
    }

def mock_design_solution(state: Dict[str, Any]) -> Dict[str, Any]:
    """Mock solution design agent"""
    logger.info("Mock: Designing solution")
    return {
        **state,
        "current_step": "design_solution",
        "solution_type": "SIMPLE_AUTOMATION",
        "technology_stack": {"language": "python", "framework": "fastapi"},
        "implementation_plan": "Create simple API endpoint",
        "current_status": "solution_designed"
    }

def mock_create_guide(state: Dict[str, Any]) -> Dict[str, Any]:
    """Mock guide creation agent"""
    logger.info("Mock: Creating implementation guide")
    return {
        **state,
        "current_step": "create_guide",
        "implementation_guide": "Step-by-step implementation guide",
        "current_status": "complete"
    }

# Test the workflow with mocked agents
def test_workflow_with_checkpointer():
    """Test complete workflow with checkpointer"""
    print("Testing workflow with checkpointer integration...")
    
    try:
        # Import workflow functions
        from app.workflows.graph import (
            create_workflow_graph, 
            get_compiled_workflow,
            create_new_workflow_session,
            get_workflow_config,
            complete_workflow_session
        )
        from langgraph.graph import StateGraph
        
        # Replace agent imports with mocks
        import app.workflows.graph as graph_module
        graph_module.analyze_problem = mock_analyze_problem
        graph_module.collect_context = mock_collect_context
        graph_module.generate_requirements = mock_generate_requirements
        graph_module.design_solution = mock_design_solution
        graph_module.create_guide = mock_create_guide
        
        # Create workflow graph
        workflow = create_workflow_graph(enable_human_loop=False)
        print("OK - Workflow graph created")
        print(f"   Nodes: {list(workflow.nodes.keys())}")
        
        # Compile workflow with checkpointer
        compiled_workflow = get_compiled_workflow(
            db_path="test_full_workflow.db",
            enable_human_loop=False,
            use_persistent_storage=True
        )
        print("OK - Workflow compiled with checkpointer")
        
        # Create new session
        thread_id = create_new_workflow_session(
            user_id="test_user",
            workflow_type="problem_solving",
            db_path="test_full_workflow.db"
        )
        print(f"OK - Created workflow session: {thread_id}")
        
        # Get workflow config
        config = get_workflow_config(thread_id)
        print(f"OK - Generated workflow config for thread: {config['configurable']['thread_id']}")
        
        # Test workflow execution with initial state
        initial_state = {
            "problem_description": "Create a simple API endpoint",
            "conversation_history": [],
            "context_data": {},
            "current_step": "start",
            "current_status": "initialized",
            "context_complete": False,
            "requires_user_input": False,
            "retry_count": 0
        }
        
        print("OK - Running workflow...")
        
        # Execute workflow
        result = compiled_workflow.invoke(initial_state, config)
        print(f"OK - Workflow completed with status: {result.get('current_status')}")
        print(f"   Final step: {result.get('current_step')}")
        print(f"   Solution type: {result.get('solution_type')}")
        
        # Mark session as complete
        complete_workflow_session(thread_id, "test_full_workflow.db")
        print("OK - Session marked as completed")
        
        return True
        
    except Exception as e:
        print(f"ERROR - Workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_workflow_resumption():
    """Test workflow interruption and resumption"""
    print("\nTesting workflow resumption...")
    
    try:
        from app.workflows.graph import (
            get_compiled_workflow,
            create_new_workflow_session,
            get_workflow_config,
            resume_workflow_session
        )
        from app.database.checkpointer import get_workflow_status
        
        # Create new session for resumption test
        thread_id = create_new_workflow_session(
            user_id="resume_test_user",
            db_path="test_resume_workflow.db"
        )
        print(f"OK - Created session for resumption test: {thread_id}")
        
        # Check initial status
        status = get_workflow_status(thread_id, "test_resume_workflow.db")
        print(f"OK - Initial session status retrieved: {status is not None}")
        
        # Resume session (should work even for new sessions)
        resumed_status = resume_workflow_session(thread_id, "test_resume_workflow.db")
        print(f"OK - Session resumption test: {resumed_status is not None}")
        
        return True
        
    except Exception as e:
        print(f"ERROR - Resumption test failed: {e}")
        return False

async def test_async_workflow():
    """Test async workflow functionality"""
    print("\nTesting async workflow...")
    
    try:
        from app.workflows.graph import get_async_compiled_workflow
        
        # Create async workflow
        async_workflow = await get_async_compiled_workflow(
            db_path="test_async_full_workflow.db",
            enable_human_loop=False
        )
        print("OK - Async workflow compiled successfully")
        
        return True
        
    except Exception as e:
        print(f"ERROR - Async workflow test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("TESTING WORKFLOW CHECKPOINTER INTEGRATION")
    print("="*60)
    
    results = []
    
    # Test basic workflow
    results.append(test_workflow_with_checkpointer())
    
    # Test resumption
    results.append(test_workflow_resumption())
    
    # Test async workflow
    results.append(asyncio.run(test_async_workflow()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    passed_tests = sum(results)
    
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    
    if all(results):
        print("\nOK - All tests passed! Workflow checkpointer integration is working.")
    else:
        print("\nERROR - Some tests failed. Check the output above for details.")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)