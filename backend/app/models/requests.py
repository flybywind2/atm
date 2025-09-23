"""
Pydantic models for API request data structures

This module defines the request models used by the FastAPI endpoints
for the problem-solving workflow.
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class AnalysisRequest(BaseModel):
    """
    Request model for starting a new problem analysis
    
    Attributes:
        problem_description: User's description of the problem to be analyzed
        user_context: Optional additional context about the user's environment
    """
    problem_description: str = Field(
        ..., 
        description="Detailed description of the problem to be analyzed",
        min_length=10,
        max_length=5000
    )
    user_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context about user's environment, constraints, etc."
    )


class ResumeRequest(BaseModel):
    """
    Request model for resuming a paused workflow with user input
    
    Attributes:
        user_input: User's response to agent questions or additional information
        context_data: Optional structured data from context collection
    """
    user_input: str = Field(
        ...,
        description="User's response to continue the workflow",
        min_length=1,
        max_length=2000
    )
    context_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured data collected during context collection phase"
    )
