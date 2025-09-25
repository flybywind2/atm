"""
컨텍스트 수집 에이전트(Context Collector)

비개발자 요약:
- 이 파일은 추가 질문을 통해 필요한 정보를 수집하는 역할을 합니다.
- 부족한 정보가 있으면 한국어 질문을 만들어 사용자에게 묻고,
  답변을 바탕으로 다음 질문 또는 다음 단계로 넘어갑니다.
- LangGraph 워크플로의 collect_context 단계에 해당합니다.
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
from app.config import settings


def _normalize_text(text: str) -> set:
    try:
        import re
        t = text.lower()
        t = re.sub(r"[\p{Punct}\s]+", " ", t)
    except Exception:
        t = text.lower()
    tokens = t.split()
    stop = {"은", "는", "이", "가", "을", "를", "에", "의", "와", "과", "또는", "그리고", "the", "a", "an", "to", "for", "of"}
    return set(tok for tok in tokens if tok not in stop)


def pick_diverse_question(candidates: list, asked: list, threshold: float = 0.5) -> str:
    asked_sets = [_normalize_text(q) for q in (asked or [])]
    for q in candidates or []:
        if not q:
            continue
        qset = _normalize_text(q)
        similar = False
        for aset in asked_sets:
            if not aset:
                continue
            inter = len(qset & aset)
            union = len(qset | aset) or 1
            if (inter / union) >= threshold:
                similar = True
                break
        if not similar:
            return q
    # Fallback: return first
    return (candidates or [None])[0]


logger = logging.getLogger(__name__)


async def collect_context(state: WorkflowState) -> WorkflowState:
    """
    Human-in-the-Loop(사람 개입) 방식으로 추가 컨텍스트를 수집합니다.

    매개변수:
        state: 현재 워크플로 상태(필요: problem_analysis, conversation_history 등)

    반환:
        다음 질문이 포함된 상태(입력 대기) 또는 컨텍스트 수집 완료 상태

    예시(입력 대기 상태 반환):
        {
          "current_step": "collecting_context",
          "current_status": "awaiting_input",
          "pending_questions": ["데이터의 출처/형식/예상 규모는?"],
          "requires_user_input": true
        }
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
        rag_enabled = settings.RAG_SERVICE_ENABLED

        # If RAG is disabled but HITL is enabled, ask LLM-generated questions
        if not rag_enabled and settings.ENABLE_HUMAN_LOOP:
            agent_service = await get_agent_service()
            # Count how many LLM questions have been asked already
            asked_count = 0
            for msg in conversation_history:
                if msg.get("sender") == "context_collector" and msg.get("message_type") == "question":
                    if msg.get("metadata", {}).get("llm"):
                        asked_count += 1

            # If we have user input, we just processed it above. Decide next action:
            if state.get("user_input"):
                # Stop if sufficient context or asked 3+ questions
                if context_data.get("sufficient_context") or asked_count >= 3:
                    conversation_history.append({
                        "sender": "context_collector",
                        "recipient": "system",
                        "message_type": "completion",
                        "content": "Context collection completed via HITL (LLM questions).",
                        "timestamp": datetime.now().isoformat(),
                        "metadata": {"context_complete": True, "llm": True}
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
                    return updated_state

            # Generate list of questions from LLM (3~5)
            # Build extended context with asked_questions and missing list
            asked_questions = [
                msg.get("content") for msg in conversation_history
                if msg.get("sender") == "context_collector" and msg.get("message_type") == "question"
            ]
            llm_context = context_data.copy() if isinstance(context_data, dict) else {}
            llm_context["asked_questions"] = asked_questions
            llm_context["missing_information"] = missing_information
            llm_context["missing_information"] = missing_information

            try:
                questions = await agent_service.collect_context_questions(problem_analysis, llm_context)
                if not isinstance(questions, list) or len(questions) == 0:
                    questions = []
                else:
                    # Ensure all items are strings
                    questions = [str(q).strip() for q in questions if q]
            except Exception as e:
                logger.warning(f"LLM context questions failed: {e}")
                questions = []

            # Fallback to single intelligent question if list empty
            if not questions:
                q = await generate_intelligent_question(
                    (missing_information[0] if missing_information else "additional_context"),
                    problem_analysis,
                    context_data,
                    conversation_history
                )
                questions = [q]

            # Pick a diverse next question
            next_q = pick_diverse_question(questions, asked_questions)

            # Determine next index for bookkeeping
            next_idx = len(asked_questions)

            conversation_history.append({
                "sender": "context_collector",
                "recipient": "user",
                "message_type": "question",
                "content": next_q,
                "timestamp": datetime.now().isoformat(),
                "metadata": {"llm": True, "index": next_idx}
            })

            updated_state = state.copy()
            updated_state.update({
                "current_step": "collecting_context",
                "current_status": "awaiting_input",
                "conversation_history": conversation_history,
                "context_data": context_data,
                "missing_information": missing_information,
                "pending_questions": [next_q],
                "requires_user_input": True,
                "context_complete": False,
                "user_input": None
            })
            return updated_state
        
        if not context_data.get("sufficient_context", False) and rag_enabled:
            # Prefer LLM-generated list of questions to avoid repetition
            asked_questions = [
                msg.get("content") for msg in conversation_history
                if msg.get("sender") == "context_collector" and msg.get("message_type") == "question"
            ]
            llm_context = context_data.copy() if isinstance(context_data, dict) else {}
            llm_context["asked_questions"] = asked_questions
            try:
                agent_service = await get_agent_service()
                questions = await agent_service.collect_context_questions(problem_analysis, llm_context)
                # Pick a diverse next question
                next_question = pick_diverse_question(questions, asked_questions)
                if not next_question:
                    # Fallback to single intelligent question
                    mi = missing_information[0] if missing_information else "additional_context"
                    next_question = await generate_intelligent_question(
                        mi, problem_analysis, context_data, conversation_history
                    )
            except Exception:
                mi = missing_information[0] if missing_information else "additional_context"
                next_question = await generate_intelligent_question(
                    mi, problem_analysis, context_data, conversation_history
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
        당신은 요구사항 수집을 진행하는 시니어 분석가입니다.
        아래 정보를 참고해 부족한 정보를 채우기 위한 구체적인 질문을 한국어(ko-KR)로 한 문장 생성하세요.

        맥락:
        - 문제: {problem_analysis.get('title', '알 수 없음')}
        - 분류: {problem_analysis.get('category', '알 수 없음')}
        - 도메인: {problem_analysis.get('domain', '알 수 없음')}
        - 부족한 정보 유형: {missing_info_type}
        - 현재 컨텍스트(JSON):
        {json.dumps(context_data, indent=2, ensure_ascii=False)}

        작성 원칙:
        1) {missing_info_type}에 정확히 해당하는 정보를 이끌어낼 수 있어야 합니다.
        2) 비전문가도 이해할 수 있게 간결하고 명확하게 묻습니다.
        3) 실행가능한 답을 유도합니다(예: 범주/단위/예시 포함).
        4) 친절하고 전문적인 톤으로 한 문장만 출력합니다.
        5) 기존에 물어본 질문(asked_questions)이 있다면 유사/중복 금지.

        출력: 질문 한 문장만 출력 (기타 설명/포맷 금지)
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
        "content": f"컨텍스트 수집 중 오류: {error_message}",
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
        "pending_questions": [{
            "text": "오류가 발생했습니다. 아래 항목을 포함하여 요구사항을 구체적으로 작성해 주세요.",
            "examples": [
                "현재 환경(서버/OS/네트워크)",
                "데이터(출처/형식/규모)",
                "목표(원하는 결과)",
                "제약(시간/예산/보안 등)"
            ]
        }]
    })
    
    return updated_state
