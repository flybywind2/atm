"""
Context Collection Agent

This agent handles Human-in-the-Loop context collection by asking clarifying
questions and gathering additional information from users using LangGraph's interrupt() functionality.
Implements the collect_context node for the LangGraph workflow.
"""

import json
import logging
import os
from typing import List, Dict, Any
from datetime import datetime

# from langgraph.graph import interrupt  # Not available in current version

from app.workflows.state import WorkflowState
from app.appendix.internal_llm import get_agent_service, LLMAgentService
from app.appendix.rag_retrieve import enhance_llm_context


logger = logging.getLogger(__name__)


async def collect_context(state: WorkflowState) -> WorkflowState:
    """
    Collect additional context through Human-in-the-Loop interaction using interrupt().
    
    This function implements the collect_context node in the LangGraph workflow,
    managing context collection phase with intelligent question generation and
    user response processing.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with collected context or interrupt for user input
    """
    try:
        # Force debug output to file
        with open("debug_workflow.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== CONTEXT_COLLECTOR CALLED at {datetime.now()} ===\n")
            f.write(f"State keys: {list(state.keys())}\n")

        logger.info("Starting context collection")
        
        problem_analysis = state.get("problem_analysis", {})
        missing_information = state.get("missing_information", [])
        conversation_history = state.get("conversation_history", [])
        context_data = state.get("context_data", {})
        
        # Check if we have user input from a previous pause
        user_input = state.get("user_input")
        
        if user_input:
            # Process the user's response
            context_data = await process_user_response(
                user_input, 
                context_data, 
                problem_analysis, 
                conversation_history
            )
            
            # Update conversation history with user response
            conversation_history.append({
                "sender": "user",
                "recipient": "context_collector",
                "message_type": "response",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
                "metadata": {"context_collection": True}
            })
            
            # Update missing information based on response
            missing_information = await update_missing_information(
                problem_analysis, 
                context_data, 
                missing_information
            )
        
        # Check if we need more context (auto-complete if RAG is disabled)
        rag_enabled = os.getenv("RAG_SERVICE_ENABLED", "true").lower() == "true"
        
        # Auto-complete context collection if RAG is disabled
        if not rag_enabled:
            logger.info("RAG service is disabled, auto-completing context collection")
            # Set context as complete since we can't collect more without RAG
            conversation_history.append({
                "sender": "context_collector",
                "recipient": "system",
                "message_type": "completion",
                "content": "Context collection completed (RAG disabled). Proceeding to requirements generation.",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "context_complete": True,
                    "rag_disabled": True
                }
            })
            
            updated_state = state.copy()
            updated_state.update({
                "current_step": "context_collected",
                "current_status": "context_complete",
                "conversation_history": conversation_history,
                "context_data": context_data or {"auto_complete": True},
                "missing_information": [],
                "requires_user_input": False,
                "context_complete": True,
                "user_input": None,
                "pending_questions": []
            })
            
            logger.info("Context collection auto-completed (RAG disabled)")
            return updated_state
        
        if missing_information and not context_data.get("sufficient_context", False) and rag_enabled:
            # Generate the next intelligent question
            next_question = await generate_intelligent_question(
                missing_information[0], 
                problem_analysis, 
                context_data,
                conversation_history
            )
            
            # Add agent question to conversation history
            conversation_history.append({
                "sender": "context_collector",
                "recipient": "user",
                "message_type": "question",
                "content": next_question,
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "missing_info_type": missing_information[0],
                    "remaining_questions": len(missing_information)
                }
            })
            
            # Update state and trigger interrupt for user input
            updated_state = state.copy()
            updated_state.update({
                "current_step": "collecting_context",
                "current_status": "awaiting_input",
                "conversation_history": conversation_history,
                "context_data": context_data,
                "missing_information": missing_information,
                "pending_questions": [next_question],
                "requires_user_input": True,
                "context_complete": False,
                "user_input": None  # Clear previous input
            })
            
            logger.info(f"Generated question for missing info: {missing_information[0]}")
            
            # Note: interrupt() not available in current LangGraph version
            # The workflow will handle the pause via requires_user_input flag
            
            return updated_state
        
        else:
            # Context collection is complete
            conversation_history.append({
                "sender": "context_collector",
                "recipient": "system",
                "message_type": "completion",
                "content": "Context collection completed. Proceeding to requirements generation.",
                "timestamp": datetime.now().isoformat(),
                "metadata": {
                    "context_complete": True,
                    "collected_items": len(context_data)
                }
            })
            
            updated_state = state.copy()
            updated_state.update({
                "current_step": "context_collected",
                "current_status": "context_complete",
                "conversation_history": conversation_history,
                "context_data": context_data,
                "missing_information": [],
                "requires_user_input": False,
                "context_complete": True,
                "user_input": None,
                "pending_questions": []
            })
            
            logger.info("Context collection completed successfully")
            
            return updated_state
            
    except Exception as e:
        logger.error(f"Error in context collection: {str(e)}")
        return handle_context_error(state, str(e))


async def process_user_response(
    user_input: str, 
    context_data: Dict[str, Any], 
    problem_analysis: Dict[str, Any],
    conversation_history: List[Dict]
) -> Dict[str, Any]:
    """
    Process user's response using LLM to extract and categorize information.
    
    Args:
        user_input: User's response
        context_data: Current collected context
        problem_analysis: Problem analysis results
        conversation_history: Conversation history for context
        
    Returns:
        Updated context data
    """
    try:
        # Get the last question from conversation history
        last_question = None
        for msg in reversed(conversation_history):
            if msg.get("sender") == "context_collector" and msg.get("message_type") == "question":
                last_question = msg.get("content")
                break
        
        # Create prompt for LLM to process the response
        processing_prompt = f"""
        You are an expert at extracting structured information from user responses.
        
        Context:
        Problem Category: {problem_analysis.get('category', 'Unknown')}
        Problem Domain: {problem_analysis.get('domain', 'Unknown')}
        
        Last Question Asked: {last_question or 'General information request'}
        
        User Response: {user_input}
        
        Current Context Data: {json.dumps(context_data, indent=2)}
        
        Please extract and categorize information from the user's response and update the context data.
        Provide a JSON response with updated context data, including any new information discovered.
        
        Focus on extracting:
        - Process details and workflows
        - Data sources and formats
        - Technical constraints
        - Business requirements
        - Integration needs
        - Performance expectations
        - User requirements
        
        Return updated context in this format:
        {{
            "process_details": "Details about current process",
            "data_sources": ["List of data sources"],
            "data_formats": ["List of data formats"],
            "frequency": "How often this runs",
            "volume": "Data volume expectations",
            "users": ["Types of users"],
            "integrations": ["Required integrations"],
            "constraints": ["Technical or business constraints"],
            "success_metrics": ["How success will be measured"],
            "timeline": "Expected timeline",
            "budget_constraints": "Budget limitations if any",
            "security_requirements": ["Security needs"],
            "compliance_requirements": ["Compliance needs"],
            "sufficient_context": false or true
        }}
        
        Set "sufficient_context" to true only if you believe we have enough information to proceed with solution design.
        """
        
        # Get the LLM agent service
        agent_service = await get_agent_service()
        
        # Use LLM to process the response  
        llm_response = await agent_service.llm_manager.simple_completion(
            processing_prompt,
            temperature=0.3
        )
        
        # Parse JSON response
        processed_data = json.loads(llm_response)
        
        # Merge with existing context data
        updated_context = context_data.copy()
        for key, value in processed_data.items():
            if value and value != "Unknown" and value != []:
                updated_context[key] = value
        
        return updated_context
        
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"LLM processing failed, using fallback: {str(e)}")
        return fallback_response_processing(user_input, context_data, problem_analysis)


def fallback_response_processing(
    user_input: str, 
    context_data: Dict[str, Any], 
    problem_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Fallback method for processing user responses when LLM is unavailable.
    
    Args:
        user_input: User's response
        context_data: Current context data
        problem_analysis: Problem analysis
        
    Returns:
        Updated context data
    """
    logger.info("Using fallback response processing")
    
    updated_context = context_data.copy()
    input_lower = user_input.lower()
    
    # Simple keyword-based processing
    if any(word in input_lower for word in ["daily", "weekly", "monthly", "hourly", "real-time"]):
        updated_context["frequency"] = user_input
    
    if any(word in input_lower for word in ["csv", "json", "xml", "database", "api", "excel"]):
        if "data_sources" not in updated_context:
            updated_context["data_sources"] = []
        updated_context["data_sources"].append(user_input)
    
    if any(word in input_lower for word in ["report", "dashboard", "chart", "graph", "email"]):
        updated_context["output_format"] = user_input
    
    if any(word in input_lower for word in ["users", "people", "employees", "customers"]):
        if "users" not in updated_context:
            updated_context["users"] = []
        updated_context["users"].append(user_input)
    
    # Add generic response under current context
    if "responses" not in updated_context:
        updated_context["responses"] = []
    updated_context["responses"].append(user_input)
    
    # Check if we have minimum required context
    required_fields = ["frequency", "data_sources", "users"]
    sufficient_context = sum(1 for field in required_fields if field in updated_context) >= 2
    updated_context["sufficient_context"] = sufficient_context
    
    return updated_context


async def update_missing_information(
    problem_analysis: Dict[str, Any], 
    context_data: Dict[str, Any], 
    current_missing: List[str]
) -> List[str]:
    """
    Update missing information list based on current context.
    
    Args:
        problem_analysis: Problem analysis results
        context_data: Current context data
        current_missing: Current missing information list
        
    Returns:
        Updated missing information list
    """
    try:
        # Use LLM to intelligently determine what's still missing
        assessment_prompt = f"""
        You are an expert business analyst. Based on the problem analysis and collected context,
        determine what critical information is still missing for solution design.
        
        Problem Analysis:
        {json.dumps(problem_analysis, indent=2)}
        
        Collected Context:
        {json.dumps(context_data, indent=2)}
        
        Please provide a JSON list of missing information areas that are critical for solution design.
        Focus on the most important gaps that would prevent effective solution design.
        
        Example format:
        ["current_process_details", "data_volume_expectations", "integration_requirements"]
        
        Limit to maximum 3 most critical missing items.
        """
        
        # Get the LLM agent service
        agent_service = await get_agent_service()
        
        llm_response = await agent_service.llm_manager.simple_completion(
            assessment_prompt,
            temperature=0.3
        )
        missing_info = json.loads(llm_response)
        
        if isinstance(missing_info, list):
            return missing_info[:3]  # Limit to 3 items
        else:
            return fallback_missing_assessment(problem_analysis, context_data)
            
    except Exception as e:
        logger.warning(f"LLM assessment failed, using fallback: {str(e)}")
        return fallback_missing_assessment(problem_analysis, context_data)


def fallback_missing_assessment(
    problem_analysis: Dict[str, Any], 
    context_data: Dict[str, Any]
) -> List[str]:
    """
    Fallback method for determining missing information.
    
    Args:
        problem_analysis: Problem analysis
        context_data: Current context data
        
    Returns:
        List of missing information areas
    """
    category = problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
    
    # Category-specific required information
    required_info = {
        "AUTOMATION": ["process_details", "frequency", "data_sources"],
        "INFORMATION_RETRIEVAL": ["data_sources", "search_criteria", "volume"],
        "MACHINE_LEARNING": ["data_sources", "target_variable", "volume"],
        "DATA_VISUALIZATION": ["data_sources", "users", "update_frequency"],
        "INTEGRATION": ["systems_involved", "data_formats", "frequency"],
        "GENERAL_PROBLEM_SOLVING": ["process_details", "users", "success_metrics"]
    }
    
    required = required_info.get(category, required_info["GENERAL_PROBLEM_SOLVING"])
    
    # Check what's missing
    missing = []
    for item in required:
        if item not in context_data or not context_data[item]:
            missing.append(item)
    
    return missing[:3]  # Limit to 3 items


async def generate_intelligent_question(
    missing_info_type: str, 
    problem_analysis: Dict[str, Any], 
    context_data: Dict[str, Any],
    conversation_history: List[Dict]
) -> str:
    """
    Generate intelligent, contextual questions based on missing information.
    
    Args:
        missing_info_type: Type of missing information
        problem_analysis: Problem analysis results
        context_data: Current context data
        conversation_history: Conversation history
        
    Returns:
        Generated question string
    """
    try:
        # Count previous questions to adjust tone
        question_count = sum(1 for msg in conversation_history 
                           if msg.get("sender") == "context_collector" and msg.get("message_type") == "question")
        
        question_prompt = f"""
        You are an expert business analyst conducting a requirements gathering session.
        Generate a clear, specific question to gather missing information.
        
        Context:
        Problem: {problem_analysis.get('title', 'Unknown problem')}
        Category: {problem_analysis.get('category', 'Unknown')}
        Domain: {problem_analysis.get('domain', 'Unknown')}
        
        Missing Information Type: {missing_info_type}
        
        Current Context:
        {json.dumps(context_data, indent=2)}
        
        Question Number: {question_count + 1}
        
        Generate a single, clear question that:
        1. Is specific to the missing information type
        2. Considers the problem context
        3. Is easy for a non-technical user to understand
        4. Helps gather actionable information for solution design
        5. Uses a friendly, professional tone
        
        Examples of good questions:
        - "How often do you currently perform this process? (daily, weekly, monthly)"
        - "What file formats do you typically work with? (Excel, CSV, PDF, etc.)"
        - "Who are the main users of this system? (employees, customers, managers)"
        
        Generate only the question text, no additional formatting or explanation.
        """
        
        # Get the LLM agent service
        agent_service = await get_agent_service()
        
        llm_response = await agent_service.llm_manager.simple_completion(
            question_prompt,
            temperature=0.4
        )
        return llm_response.strip()
        
    except Exception as e:
        logger.warning(f"LLM question generation failed, using fallback: {str(e)}")
        return generate_fallback_question(missing_info_type, problem_analysis)


def generate_fallback_question(missing_info_type: str, problem_analysis: Dict[str, Any]) -> str:
    """
    Generate fallback questions when LLM is unavailable.
    
    Args:
        missing_info_type: Type of missing information
        problem_analysis: Problem analysis
        
    Returns:
        Generated question string
    """
    category = problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
    
    # Pre-defined questions based on missing information type
    question_templates = {
        "process_details": "Could you describe your current process in detail? What steps are involved and how long does it typically take?",
        "frequency": "How often does this process need to run? (daily, weekly, monthly, or on-demand)",
        "data_sources": "What data sources will you be working with? (files, databases, APIs, etc.)",
        "data_formats": "What file formats do you typically use? (Excel, CSV, JSON, PDF, etc.)",
        "volume": "What's the expected volume of data? (number of records, file sizes, etc.)",
        "users": "Who are the main users of this system? (employees, customers, managers, etc.)",
        "integrations": "Do you need to integrate with any existing systems? If so, which ones?",
        "success_metrics": "How will you measure the success of this solution?",
        "constraints": "Are there any technical constraints or limitations I should know about?",
        "timeline": "What's your expected timeline for implementing this solution?",
        "output_format": "What format do you need for the output? (dashboard, report, file, etc.)",
        "security_requirements": "Are there any security or privacy requirements I should consider?",
        "current_process_details": "Can you walk me through your current process step by step?",
        "data_volume_expectations": "How much data do you expect to process? (daily, weekly, total volume)",
        "integration_requirements": "What systems or tools need to be connected or integrated?"
    }
    
    question = question_templates.get(
        missing_info_type, 
        f"Could you provide more information about {missing_info_type.replace('_', ' ')}?"
    )
    
    return question


def handle_context_error(state: WorkflowState, error_message: str) -> WorkflowState:
    """
    Handle errors during context collection.
    
    Args:
        state: Current workflow state
        error_message: Error description
        
    Returns:
        Updated state with error information
    """
    logger.error(f"Context collection failed: {error_message}")
    
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({
        "sender": "context_collector",
        "recipient": "system",
        "message_type": "error",
        "content": f"Context collection error: {error_message}",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"error_type": "context_collection_error"}
    })
    
    # Create error state that requires user input
    updated_state = state.copy()
    updated_state.update({
        "current_step": "context_error",
        "current_status": "error",
        "error_message": error_message,
        "conversation_history": conversation_history,
        "retry_count": state.get("retry_count", 0) + 1,
        "requires_user_input": True,
        "context_complete": False,
        "pending_questions": ["I encountered an error. Could you please provide more details about your requirements?"]
    })
    
    return updated_state