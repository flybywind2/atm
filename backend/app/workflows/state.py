"""
LangGraph TypedDict state definitions

This module defines the state objects used by LangGraph workflows
for the problem-solving copilot system according to the LangGraph Workflow Agent specifications.
"""

from typing import TypedDict, Dict, List, Any, Optional
from typing_extensions import NotRequired


class WorkflowState(TypedDict):
    """
    Main state object for the LangGraph workflow as specified in the agent instructions.
    
    This TypedDict defines all the state variables that are passed
    between different nodes in the LangGraph workflow according to
    the LangGraph Workflow Agent specifications.
    """
    # Core problem information (from agent spec)
    problem_description: str
    conversation_history: List[Dict]
    context_data: Dict
    
    # Generated documents (from agent spec)
    requirements_doc: Optional[str]
    solution_type: Optional[str]
    implementation_guide: Optional[str]
    
    # Workflow control (from agent spec)
    current_step: str
    pending_questions: List[str]
    user_responses: List[str]
    
    # Analysis results (enhanced from SDD spec)
    problem_analysis: NotRequired[Dict[str, Any]]  # Structured problem breakdown
    missing_information: NotRequired[List[str]]    # Missing context areas
    
    # Generated journey map (from SDD spec)
    user_journey_map: NotRequired[str]  # User journey map in markdown
    
    # Requirements components (from SDD spec)
    requirements_definition: NotRequired[str]  # Complete SRS document
    
    # Solution design components (from SDD spec)
    recommended_solution_type: NotRequired[str]  # Solution type classification
    technology_stack: NotRequired[Dict]          # Recommended tech stack
    implementation_plan: NotRequired[str]        # Implementation plan
    
    # Workflow status (from SDD spec)
    current_status: str  # analyzing, awaiting_input, complete, etc.
    
    # Context collection state
    context_complete: bool
    requires_user_input: bool
    user_input: NotRequired[str]
    
    # Error handling
    error_message: NotRequired[str]
    retry_count: int


# Backward compatibility alias for existing code
ProblemSolvingState = WorkflowState


class AgentMessage(TypedDict):
    """
    Structure for agent-to-agent or agent-to-user messages
    """
    sender: str  # Agent name or "user"
    recipient: str  # Agent name or "user"
    message_type: str  # "question", "response", "data", "error"
    content: str
    timestamp: str
    metadata: NotRequired[Dict[str, Any]]


class ContextCollectionState(TypedDict):
    """
    Specialized state for context collection phase
    Enhanced to support HITL interaction
    """
    questions_asked: List[str]
    responses_received: List[str]
    missing_info: List[str]
    collection_complete: bool
    next_question: NotRequired[str]
    awaiting_user_response: bool


class RequirementsState(TypedDict):
    """
    Specialized state for requirements generation phase
    Based on SDD specifications for SRS generation
    """
    stakeholders: List[str]
    business_objectives: List[str]
    constraints: List[str]
    assumptions: List[str]
    risks: List[str]
    success_metrics: List[str]
    functional_requirements: List[str]
    non_functional_requirements: List[str]
    user_stories: List[Dict[str, str]]
    acceptance_criteria: List[str]


class SolutionDesignState(TypedDict):
    """
    Specialized state for solution design phase
    Enhanced for solution type routing and technology recommendations
    """
    design_patterns: List[str]
    data_flow: str
    integration_points: List[str]
    deployment_strategy: str
    testing_approach: str
    monitoring_requirements: List[str]
    solution_category: str  # SIMPLE_AUTOMATION, RAG, ML_CLASSIFICATION, etc.
    architecture_components: List[str]