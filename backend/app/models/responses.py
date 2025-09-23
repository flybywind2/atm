"""
Pydantic models for API response data structures

This module defines the response models used by the FastAPI endpoints
for the problem-solving workflow.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List
from enum import Enum


class WorkflowStatus(str, Enum):
    """Enumeration of possible workflow statuses"""
    STARTED = "started"
    ANALYZING = "analyzing"
    COLLECTING_CONTEXT = "collecting_context"
    AWAITING_INPUT = "awaiting_input"
    GENERATING_REQUIREMENTS = "generating_requirements"
    DESIGNING_SOLUTION = "designing_solution"
    CREATING_GUIDE = "creating_guide"
    COMPLETED = "completed"
    PAUSED = "paused"
    ERROR = "error"
    PROCESSING = "processing"


class StatusResponse(BaseModel):
    """
    Response model for workflow status queries
    
    Attributes:
        thread_id: Unique identifier for the workflow thread
        status: Current status of the workflow
        current_step: Current step being executed
        progress_percentage: Overall progress as percentage
        message: Status message or agent question
        requires_input: Whether workflow is paused waiting for user input
        questions: List of clarification questions from agents
        results: Partial results if available
    """
    thread_id: str = Field(..., description="Unique workflow thread identifier")
    status: WorkflowStatus = Field(..., description="Current workflow status")
    current_step: str = Field(..., description="Current step being executed")
    progress_percentage: int = Field(
        ..., 
        description="Progress as percentage (0-100)", 
        ge=0, 
        le=100
    )
    message: Optional[str] = Field(
        default=None,
        description="Status message or question from agent"
    )
    requires_input: bool = Field(
        default=False,
        description="Whether workflow is paused waiting for user input"
    )
    questions: Optional[List[str]] = Field(
        default=None,
        description="List of clarification questions from agents"
    )
    results: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Partial results or generated documents"
    )


class StartAnalysisResponse(BaseModel):
    """
    Response model for start analysis endpoint
    
    Attributes:
        thread_id: Unique identifier for the workflow thread
        status: Initial status of the workflow
        message: Initial status message
    """
    thread_id: str = Field(..., description="Unique workflow thread identifier")
    status: WorkflowStatus = Field(..., description="Initial workflow status")
    message: str = Field(..., description="Initial status message")


class AnalysisResponse(BaseModel):
    """
    Response model for completed analysis workflow results
    
    Attributes:
        thread_id: Unique identifier for the workflow thread
        status: Final status of the analysis
        documents: Generated documents (SRS, journey map, guide, etc.)
        solution_type: Classified solution type
        tech_stack: Recommended technology stack
    """
    thread_id: str = Field(..., description="Unique workflow thread identifier")
    status: WorkflowStatus = Field(..., description="Final workflow status")
    documents: Dict[str, str] = Field(
        default_factory=dict,
        description="Generated documents in markdown format"
    )
    solution_type: Optional[str] = Field(
        default=None,
        description="Classified solution type (SIMPLE_AUTOMATION, RAG, ML_CLASSIFICATION, etc.)"
    )
    tech_stack: Optional[List[str]] = Field(
        default=None,
        description="Recommended Python libraries and frameworks"
    )


class ErrorResponse(BaseModel):
    """
    Response model for error cases
    
    Attributes:
        error: Error type/code
        message: Human-readable error message
        details: Additional error details
    """
    error: str = Field(..., description="Error type or code")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional error details"
    )
