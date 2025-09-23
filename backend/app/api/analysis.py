"""
FastAPI endpoints for analysis workflow

This module implements the REST API endpoints for the problem-solving workflow,
including start analysis, status checking, and resume functionality.
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException, Path, status
from typing import Dict, Any, Optional
import uuid
import asyncio
import logging
from datetime import datetime

from app.models.requests import AnalysisRequest, ResumeRequest
from app.models.responses import (
    StatusResponse, 
    StartAnalysisResponse, 
    ErrorResponse, 
    WorkflowStatus
)
from app.workflows.graph import get_compiled_workflow
from app.workflows.state import ProblemSolvingState
from app.database.checkpointer import get_checkpointer_manager

# Configure logging
logger = logging.getLogger(__name__)

# Router for analysis endpoints
analysis_router = APIRouter()

# Global storage for active workflows
active_workflows: Dict[str, Dict[str, Any]] = {}


@analysis_router.post("/start-analysis", response_model=StatusResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_analysis(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks
):
    """
    Start a new problem analysis workflow
    
    Args:
        request: Analysis request with problem description
        background_tasks: FastAPI background tasks for async processing
    
    Returns:
        StatusResponse: Initial status with thread_id
    
    Raises:
        HTTPException: If workflow initialization fails
    """
    thread_id = str(uuid.uuid4())
    
    try:
        # Initialize workflow state
        initial_state: ProblemSolvingState = {
            "problem_description": request.problem_description,
            "user_context": request.user_context or {},
            "current_step": "analyze_problem",
            "conversation_history": [],
            "requires_user_input": False,
            "context_complete": False,
            "retry_count": 0
        }
        
        # Store workflow metadata
        active_workflows[thread_id] = {
            "status": WorkflowStatus.STARTED,
            "current_step": "analyze_problem",
            "progress_percentage": 0,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "requires_input": False,
            "message": "Initializing workflow...",
            "state": initial_state
        }
        
        # Start background workflow with LangGraph
        background_tasks.add_task(run_langgraph_workflow, thread_id, initial_state)
        
        logger.info(f"Started analysis workflow {thread_id}")
        
        return StatusResponse(
            thread_id=thread_id,
            status=WorkflowStatus.STARTED,
            current_step="analyze_problem",
            progress_percentage=0,
            message="Analysis workflow started successfully",
            requires_input=False
        )
        
    except Exception as e:
        logger.error(f"Failed to start workflow {thread_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start analysis workflow: {str(e)}"
        )


@analysis_router.get("/status/{thread_id}", response_model=StatusResponse)
async def get_status(
    thread_id: str = Path(..., description="Workflow thread identifier")
):
    """
    Get current status of a workflow thread
    
    Args:
        thread_id: Unique workflow thread identifier
    
    Returns:
        StatusResponse: Current workflow status
    
    Raises:
        HTTPException: If thread_id not found
    """
    if thread_id not in active_workflows:
        # Try to get status from checkpointer
        try:
            checkpointer_manager = get_checkpointer_manager()
            workflow_data = checkpointer_manager.get_workflow_status(thread_id)
            
            if workflow_data is None:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Workflow thread not found"
                )
                
            # Convert checkpointer data to status response
            return StatusResponse(
                thread_id=thread_id,
                status=WorkflowStatus(workflow_data.get("status", "unknown")),
                current_step=workflow_data.get("current_step", "unknown"),
                progress_percentage=workflow_data.get("progress_percentage", 0),
                message=workflow_data.get("message", ""),
                requires_input=workflow_data.get("requires_input", False)
            )
            
        except Exception as e:
            logger.error(f"Failed to get workflow status from checkpointer: {e}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Workflow thread not found"
            )
    
    workflow_data = active_workflows[thread_id]
    workflow_state = workflow_data.get("state", {})
    
    # Check if there are questions waiting for user input
    questions = []
    if workflow_data.get("requires_input"):
        questions = workflow_state.get("context_questions", [])
    
    # Get partial results if available
    results = {}
    if workflow_state.get("structured_problem"):
        results["problem_analysis"] = workflow_state["structured_problem"]
    if workflow_state.get("requirements_document"):
        results["requirements_document"] = workflow_state["requirements_document"]
    if workflow_state.get("user_journey_map"):
        results["user_journey_map"] = workflow_state["user_journey_map"]
    if workflow_state.get("implementation_guide"):
        results["implementation_guide"] = workflow_state["implementation_guide"]
    
    return StatusResponse(
        thread_id=thread_id,
        status=workflow_data["status"],
        current_step=workflow_data["current_step"],
        progress_percentage=workflow_data["progress_percentage"],
        message=workflow_data.get("message"),
        requires_input=workflow_data.get("requires_input", False),
        questions=questions if questions else None,
        results=results if results else None
    )


@analysis_router.post("/resume/{thread_id}", response_model=StatusResponse)
async def resume_workflow(
    request: ResumeRequest,
    background_tasks: BackgroundTasks,
    thread_id: str = Path(..., description="Workflow thread identifier")
):
    """
    Resume a paused workflow with user input
    
    Args:
        request: Resume request with user input
        background_tasks: FastAPI background tasks for async processing
        thread_id: Unique workflow thread identifier
    
    Returns:
        StatusResponse: Updated workflow status
    
    Raises:
        HTTPException: If thread_id not found or workflow not paused
    """
    if thread_id not in active_workflows:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Workflow thread not found"
        )
    
    workflow_data = active_workflows[thread_id]
    
    if not workflow_data.get("requires_input"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Workflow is not paused waiting for input"
        )
    
    try:
        # Update workflow state with user input
        workflow_state = workflow_data.get("state", {})
        workflow_state["user_input"] = request.user_input
        workflow_state["requires_user_input"] = False
        
        # Add context data if provided
        if request.context_data:
            collected_context = workflow_state.get("collected_context", {})
            collected_context.update(request.context_data)
            workflow_state["collected_context"] = collected_context
        
        # Add to conversation history
        workflow_state["conversation_history"].append({
            "sender": "user",
            "message": request.user_input,
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Update workflow metadata
        workflow_data["requires_input"] = False
        workflow_data["status"] = WorkflowStatus.PROCESSING
        workflow_data["message"] = "Processing user input..."
        workflow_data["updated_at"] = datetime.utcnow().isoformat()
        
        # Resume background workflow with LangGraph
        background_tasks.add_task(resume_langgraph_workflow, thread_id, workflow_state)
        
        logger.info(f"Resumed workflow {thread_id} with user input")
        
        return StatusResponse(
            thread_id=thread_id,
            status=WorkflowStatus.PROCESSING,
            current_step=workflow_data["current_step"],
            progress_percentage=workflow_data["progress_percentage"],
            message="Workflow resumed, processing user input...",
            requires_input=False
        )
        
    except Exception as e:
        logger.error(f"Failed to resume workflow {thread_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to resume workflow: {str(e)}"
        )


async def run_langgraph_workflow(thread_id: str, initial_state: ProblemSolvingState):
    """
    Background task to run the LangGraph analysis workflow
    
    Args:
        thread_id: Unique workflow thread identifier
        initial_state: Initial state for the workflow
    """
    workflow_data = active_workflows.get(thread_id)
    if not workflow_data:
        logger.error(f"Workflow data not found for thread {thread_id}")
        return
    
    try:
        # Get compiled workflow
        compiled_workflow = get_compiled_workflow()
        
        # Configure thread for checkpointing
        thread_config = {"configurable": {"thread_id": thread_id}}
        
        # Update status to analyzing
        workflow_data["status"] = WorkflowStatus.ANALYZING
        workflow_data["current_step"] = "analyze_problem"
        workflow_data["progress_percentage"] = 10
        workflow_data["message"] = "Analyzing problem..."
        
        # Stream the workflow execution
        async for output in compiled_workflow.astream(
            initial_state, 
            config=thread_config,
            stream_mode="values"
        ):
            # Update workflow state and progress
            await update_workflow_progress(thread_id, output)
            
            # Check if workflow is interrupted (awaiting user input)
            if output.get("requires_user_input"):
                workflow_data["requires_input"] = True
                workflow_data["status"] = WorkflowStatus.AWAITING_INPUT
                workflow_data["message"] = "Waiting for user input..."
                logger.info(f"Workflow {thread_id} paused for user input")
                return
        
        # Workflow completed successfully
        workflow_data["status"] = WorkflowStatus.COMPLETED
        workflow_data["current_step"] = "completed"
        workflow_data["progress_percentage"] = 100
        workflow_data["message"] = "Analysis completed successfully"
        workflow_data["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Workflow {thread_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Workflow {thread_id} failed: {e}")
        workflow_data["status"] = WorkflowStatus.ERROR
        workflow_data["message"] = f"Workflow failed: {str(e)}"
        workflow_data["updated_at"] = datetime.utcnow().isoformat()


async def resume_langgraph_workflow(thread_id: str, updated_state: ProblemSolvingState):
    """
    Background task to resume a paused LangGraph workflow
    
    Args:
        thread_id: Unique workflow thread identifier
        updated_state: Updated state with user input
    """
    workflow_data = active_workflows.get(thread_id)
    if not workflow_data:
        logger.error(f"Workflow data not found for thread {thread_id}")
        return
    
    try:
        # Get compiled workflow
        compiled_workflow = get_compiled_workflow()
        
        # Configure thread for checkpointing
        thread_config = {"configurable": {"thread_id": thread_id}}
        
        # Update state to show processing
        workflow_data["status"] = WorkflowStatus.PROCESSING
        workflow_data["message"] = "Processing user input..."
        
        # Resume workflow execution from where it left off
        async for output in compiled_workflow.astream(
            None,  # Continue from checkpoint
            config=thread_config,
            stream_mode="values"
        ):
            # Update workflow state and progress
            await update_workflow_progress(thread_id, output)
            
            # Check if workflow is interrupted again
            if output.get("requires_user_input"):
                workflow_data["requires_input"] = True
                workflow_data["status"] = WorkflowStatus.AWAITING_INPUT
                workflow_data["message"] = "Waiting for additional user input..."
                logger.info(f"Workflow {thread_id} paused again for user input")
                return
        
        # Workflow completed successfully
        workflow_data["status"] = WorkflowStatus.COMPLETED
        workflow_data["current_step"] = "completed"
        workflow_data["progress_percentage"] = 100
        workflow_data["message"] = "Analysis completed successfully"
        workflow_data["updated_at"] = datetime.utcnow().isoformat()
        
        logger.info(f"Workflow {thread_id} resumed and completed successfully")
        
    except Exception as e:
        logger.error(f"Failed to resume workflow {thread_id}: {e}")
        workflow_data["status"] = WorkflowStatus.ERROR
        workflow_data["message"] = f"Workflow failed: {str(e)}"
        workflow_data["updated_at"] = datetime.utcnow().isoformat()


async def update_workflow_progress(thread_id: str, state_output: Dict[str, Any]):
    """
    Update workflow progress based on state output
    
    Args:
        thread_id: Workflow thread identifier
        state_output: Current state output from LangGraph
    """
    workflow_data = active_workflows.get(thread_id)
    if not workflow_data:
        return
    
    # Map workflow steps to progress percentages
    step_progress = {
        "analyze_problem": 20,
        "collect_context": 40,
        "generate_requirements": 60,
        "design_solution": 80,
        "create_guide": 90
    }
    
    current_step = state_output.get("current_step", "unknown")
    progress = step_progress.get(current_step, workflow_data["progress_percentage"])
    
    # Update workflow metadata
    workflow_data["current_step"] = current_step
    workflow_data["progress_percentage"] = progress
    workflow_data["updated_at"] = datetime.utcnow().isoformat()
    workflow_data["state"] = state_output
    
    # Update status based on current step
    if current_step == "analyze_problem":
        workflow_data["status"] = WorkflowStatus.ANALYZING
        workflow_data["message"] = "Analyzing problem structure..."
    elif current_step == "collect_context":
        workflow_data["status"] = WorkflowStatus.COLLECTING_CONTEXT
        workflow_data["message"] = "Collecting additional context..."
    elif current_step == "generate_requirements":
        workflow_data["status"] = WorkflowStatus.GENERATING_REQUIREMENTS
        workflow_data["message"] = "Generating requirements document..."
    elif current_step == "design_solution":
        workflow_data["status"] = WorkflowStatus.DESIGNING_SOLUTION
        workflow_data["message"] = "Designing solution architecture..."
    elif current_step == "create_guide":
        workflow_data["status"] = WorkflowStatus.CREATING_GUIDE
        workflow_data["message"] = "Creating implementation guide..."
