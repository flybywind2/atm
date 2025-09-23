"""
Problem Analysis Agent

This agent handles the initial analysis and structuring of user problems.
It breaks down the problem into components and assesses complexity using LLM integration.
Implements the analyze_problem node for the LangGraph workflow.
"""

import json
import logging
from typing import Dict, Any
from datetime import datetime

from app.workflows.state import WorkflowState
from app.appendix.internal_llm import get_agent_service, LLMAgentService
from app.appendix.rag_retrieve import enhance_llm_context


logger = logging.getLogger(__name__)


async def analyze_problem(state: WorkflowState) -> WorkflowState:
    """
    Analyze and structure the user's problem description using LLM integration.
    
    This function implements the analyze_problem node in the LangGraph workflow,
    providing structured problem breakdown and categorization.
    
    Args:
        state: Current workflow state containing problem_description
        
    Returns:
        Updated state with structured problem analysis
    """
    try:
        logger.info("Starting problem analysis")
        
        problem_description = state["problem_description"]
        conversation_history = state.get("conversation_history", [])
        
        # Create structured problem analysis using LLM with RAG enhancement
        problem_analysis = await create_structured_analysis(problem_description, state.get("context_data", {}))
        
        # Extract key components from analysis
        problem_category = problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
        complexity_assessment = problem_analysis.get("complexity", "MEDIUM")
        missing_info = problem_analysis.get("missing_information", [])
        
        # Update conversation history
        conversation_history.append({
            "sender": "analyzer",
            "recipient": "system",
            "message_type": "analysis_complete",
            "content": f"Problem analyzed and categorized as: {problem_category} (Complexity: {complexity_assessment})",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "category": problem_category,
                "complexity": complexity_assessment,
                "missing_info_count": len(missing_info)
            }
        })
        
        # Determine if context collection is needed
        context_complete = len(missing_info) == 0
        
        # Update state with analysis results
        updated_state = state.copy()
        updated_state.update({
            "current_step": "problem_analyzed",
            "current_status": "analyzing_complete",
            "problem_analysis": problem_analysis,
            "missing_information": missing_info,
            "conversation_history": conversation_history,
            "context_data": {
                "category": problem_category,
                "complexity": complexity_assessment,
                "structured_analysis": problem_analysis
            },
            "context_complete": context_complete,
            "requires_user_input": not context_complete,
            "pending_questions": missing_info[:3],  # Limit initial questions
            "user_responses": [],
            "retry_count": 0
        })
        
        logger.info(f"Problem analysis completed. Category: {problem_category}, Complexity: {complexity_assessment}")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in problem analysis: {str(e)}")
        return handle_analysis_error(state, str(e))


async def create_structured_analysis(problem_description: str, context_data: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Create structured problem analysis using LLM integration with RAG enhancement.
    
    Args:
        problem_description: User's problem description
        context_data: Current context data from workflow state
        
    Returns:
        Structured analysis dictionary
    """
    try:
        # Enhance context with RAG-retrieved information
        enhanced_context = await enhance_llm_context(
            agent_type="problem_analyzer",
            query=problem_description,
            current_context=context_data or {},
            domain="business_automation"
        )
        
        # Get the LLM agent service
        agent_service = await get_agent_service()
        
        # Create enhanced prompt with RAG context
        enhanced_prompt = create_enhanced_analysis_prompt(problem_description, enhanced_context)
        
        # Use the specialized problem analysis method with enhanced context
        analysis = await agent_service.analyze_problem(enhanced_prompt)
        
        # Add RAG enhancement metadata
        if enhanced_context.get("rag_enhanced"):
            analysis["rag_enhanced"] = True
            analysis["context_sources"] = enhanced_context.get("context_sources", [])
            analysis["domain_patterns"] = enhanced_context.get("domain_patterns", [])
        
        # Validate and sanitize response
        analysis = validate_analysis_response(analysis)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Error in LLM analysis: {str(e)}")
        return create_fallback_analysis(problem_description)


def create_enhanced_analysis_prompt(problem_description: str, enhanced_context: Dict[str, Any]) -> str:
    """
    Create enhanced prompt for problem analysis using RAG context.
    
    Args:
        problem_description: Original problem description
        enhanced_context: RAG-enhanced context information
        
    Returns:
        Enhanced prompt string for LLM analysis
    """
    base_prompt = f"""
Problem to analyze: {problem_description}

Please provide a structured analysis of this problem with the following components:
- Problem category and classification
- Complexity assessment
- Key stakeholders and domain
- Current state vs desired state
- Pain points and challenges
- Success criteria and constraints
- Missing information needed for complete solution
"""

    # Add RAG-enhanced context if available
    if enhanced_context.get("rag_enhanced"):
        context_section = "\n\n--- RELEVANT CONTEXT FROM KNOWLEDGE BASE ---\n"
        
        # Add retrieved context summaries
        for ctx in enhanced_context.get("retrieved_context", []):
            context_section += f"\n• {ctx['title']} (Relevance: {ctx['relevance']:.2f})\n"
            context_section += f"  {ctx['content'][:200]}...\n"
        
        # Add domain patterns if available
        domain_patterns = enhanced_context.get("domain_patterns", [])
        if domain_patterns:
            context_section += f"\n--- DOMAIN-SPECIFIC PATTERNS ---\n"
            for pattern in domain_patterns[:3]:  # Limit to top 3
                context_section += f"• {pattern}\n"
        
        context_section += "\nPlease consider this context when analyzing the problem, especially for categorization and identifying common patterns.\n"
        
        return base_prompt + context_section
    
    return base_prompt


def validate_analysis_response(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize LLM analysis response.
    
    Args:
        analysis: Raw analysis from LLM
        
    Returns:
        Validated and sanitized analysis
    """
    # Valid categories
    valid_categories = {
        "AUTOMATION", "INFORMATION_RETRIEVAL", "MACHINE_LEARNING", 
        "DATA_VISUALIZATION", "INTEGRATION", "GENERAL_PROBLEM_SOLVING"
    }
    
    # Valid complexity levels
    valid_complexity = {"LOW", "MEDIUM", "HIGH"}
    
    # Ensure required fields exist with defaults
    validated = {
        "title": analysis.get("title", "Problem Analysis"),
        "category": analysis.get("category", "GENERAL_PROBLEM_SOLVING"),
        "complexity": analysis.get("complexity", "MEDIUM"),
        "domain": analysis.get("domain", "Business Process"),
        "stakeholders": analysis.get("stakeholders", ["End Users"]),
        "current_state": analysis.get("current_state", "Current manual process"),
        "desired_state": analysis.get("desired_state", "Improved automated process"),
        "pain_points": analysis.get("pain_points", ["Manual inefficiencies"]),
        "success_criteria": analysis.get("success_criteria", ["Improved efficiency"]),
        "constraints": analysis.get("constraints", []),
        "missing_information": analysis.get("missing_information", []),
        "urgency": analysis.get("urgency", "MEDIUM"),
        "scope": analysis.get("scope", "Process improvement"),
        "assumptions": analysis.get("assumptions", [])
    }
    
    # Validate category
    if validated["category"] not in valid_categories:
        validated["category"] = "GENERAL_PROBLEM_SOLVING"
    
    # Validate complexity
    if validated["complexity"] not in valid_complexity:
        validated["complexity"] = "MEDIUM"
    
    # Validate urgency
    if validated["urgency"] not in valid_complexity:
        validated["urgency"] = "MEDIUM"
    
    # Ensure lists are actually lists
    list_fields = ["stakeholders", "pain_points", "success_criteria", "constraints", "missing_information", "assumptions"]
    for field in list_fields:
        if not isinstance(validated[field], list):
            validated[field] = []
    
    return validated


def create_fallback_analysis(problem_description: str) -> Dict[str, Any]:
    """
    Create fallback analysis when LLM is unavailable.
    
    Args:
        problem_description: User's problem description
        
    Returns:
        Basic structured analysis
    """
    logger.info("Using fallback analysis due to LLM unavailability")
    
    # Basic keyword-based categorization
    category = categorize_problem_keywords(problem_description)
    complexity = assess_complexity_heuristic(problem_description)
    
    return {
        "title": "Automated Problem Analysis",
        "category": category,
        "complexity": complexity,
        "domain": "Business Process",
        "stakeholders": ["End Users", "Management"],
        "current_state": "Manual process requiring improvement",
        "desired_state": "Automated solution for improved efficiency",
        "pain_points": [
            "Time-consuming manual tasks",
            "Error-prone processes",
            "Lack of visibility"
        ],
        "success_criteria": [
            "Reduced processing time",
            "Improved accuracy",
            "Better reporting"
        ],
        "constraints": [
            "Must be implementable in Python",
            "Should be maintainable by beginners"
        ],
        "missing_information": [
            "Current process details",
            "Data sources and formats",
            "Expected volume and frequency",
            "Integration requirements",
            "Success metrics"
        ],
        "urgency": "MEDIUM",
        "scope": "Process automation and improvement",
        "assumptions": [
            "User has basic Python environment",
            "Required data is accessible",
            "Stakeholders are available for clarification"
        ]
    }


def categorize_problem_keywords(problem_description: str) -> str:
    """
    Categorize problem based on keywords (fallback method).
    
    Args:
        problem_description: User's problem description
        
    Returns:
        Problem category string
    """
    description_lower = problem_description.lower()
    
    # Keyword mapping for categories
    category_keywords = {
        "AUTOMATION": ["automate", "script", "repetitive", "manual", "schedule", "batch"],
        "INFORMATION_RETRIEVAL": ["search", "find", "document", "knowledge", "retrieve", "query"],
        "MACHINE_LEARNING": ["predict", "classify", "model", "pattern", "learn", "training"],
        "DATA_VISUALIZATION": ["chart", "graph", "dashboard", "report", "visualize", "plot"],
        "INTEGRATION": ["api", "connect", "integrate", "sync", "import", "export"]
    }
    
    # Count keyword matches
    category_scores = {}
    for category, keywords in category_keywords.items():
        score = sum(1 for keyword in keywords if keyword in description_lower)
        if score > 0:
            category_scores[category] = score
    
    # Return highest scoring category or default
    if category_scores:
        return max(category_scores, key=category_scores.get)
    else:
        return "GENERAL_PROBLEM_SOLVING"


def assess_complexity_heuristic(problem_description: str) -> str:
    """
    Assess complexity using heuristic methods (fallback).
    
    Args:
        problem_description: User's problem description
        
    Returns:
        Complexity level (LOW, MEDIUM, HIGH)
    """
    description_lower = problem_description.lower()
    word_count = len(problem_description.split())
    
    # Complexity indicators
    high_complexity_indicators = [
        "multiple systems", "integration", "real-time", "machine learning",
        "scalability", "high availability", "security", "compliance",
        "api", "database", "microservices", "distributed"
    ]
    
    medium_complexity_indicators = [
        "automate", "dashboard", "report", "schedule", "process",
        "workflow", "analysis", "transformation"
    ]
    
    # Count indicators
    high_count = sum(1 for indicator in high_complexity_indicators if indicator in description_lower)
    medium_count = sum(1 for indicator in medium_complexity_indicators if indicator in description_lower)
    
    # Determine complexity
    if word_count > 100 or high_count >= 2:
        return "HIGH"
    elif word_count > 50 or high_count >= 1 or medium_count >= 2:
        return "MEDIUM"
    else:
        return "LOW"


def handle_analysis_error(state: WorkflowState, error_message: str) -> WorkflowState:
    """
    Handle errors during problem analysis.
    
    Args:
        state: Current workflow state
        error_message: Error description
        
    Returns:
        Updated state with error information
    """
    logger.error(f"Problem analysis failed: {error_message}")
    
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({
        "sender": "analyzer",
        "recipient": "system",
        "message_type": "error",
        "content": f"Analysis failed: {error_message}",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"error_type": "analysis_error"}
    })
    
    # Create minimal fallback state
    updated_state = state.copy()
    updated_state.update({
        "current_step": "analysis_error",
        "current_status": "error",
        "error_message": error_message,
        "conversation_history": conversation_history,
        "retry_count": state.get("retry_count", 0) + 1,
        "requires_user_input": True,
        "context_complete": False,
        "pending_questions": ["Could you please provide more details about your problem?"],
        "context_data": {"error": True, "error_message": error_message}
    })
    
    return updated_state