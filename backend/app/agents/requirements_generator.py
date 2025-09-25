"""
요구사항 생성 에이전트(Requirements Generator)

비개발자 요약:
- 분석/컨텍스트를 바탕으로 SRS(요구사항 명세서)를 생성합니다.
- 기능/비기능 요구사항, 사용자 스토리, 승인 기준 등을 포함합니다.
- LangGraph의 generate_requirements 단계에 해당합니다.
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from app.workflows.state import WorkflowState
from app.appendix.internal_llm import get_agent_service, LLMAgentService
from app.appendix.rag_retrieve import enhance_llm_context


logger = logging.getLogger(__name__)


async def generate_requirements(state: WorkflowState) -> WorkflowState:
    """
    분석/컨텍스트를 바탕으로 포괄적인 SRS(요구사항 명세서)를 생성합니다.

    매개변수:
        state: 현재 워크플로 상태(필요: problem_analysis, context_data 등)

    반환:
        SRS 문서와 여정 지도 등이 포함된 갱신 상태

    예시(반환 주요 키):
        {
          "current_step": "requirements_generated",
          "requirements_document": "# 소프트웨어 요구사항 명세서 ...",
          "user_journey_map": "# 사용자 여정 지도 ..."
        }
    """
    try:
        # Force debug output to file
        with open("debug_workflow.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== REQUIREMENTS_GENERATOR CALLED at {datetime.now()} ===\n")
            f.write(f"State keys: {list(state.keys())}\n")

        logger.info("=== STARTING REQUIREMENTS GENERATION ===")
        logger.info(f"Current state keys: {list(state.keys())}")

        problem_analysis = state.get("problem_analysis", {})
        context_data = state.get("context_data", {})
        conversation_history = state.get("conversation_history", [])

        logger.info(f"Problem analysis: {problem_analysis}")
        logger.info(f"Context data keys: {list(context_data.keys()) if context_data else 'No context data'}")
        
        # Enhance context with RAG for requirements generation
        logger.info("Enhancing context with RAG...")
        try:
            enhanced_context = await enhance_llm_context(
                agent_type="requirements_generator",
                query=f"requirements specification {problem_analysis.get('category', '')} {problem_analysis.get('domain', '')}",
                current_context=context_data,
                domain="business_automation"
            )
            logger.info(f"Enhanced context obtained: {type(enhanced_context)}")
        except Exception as e:
            logger.error(f"RAG enhancement failed: {str(e)}")
            enhanced_context = context_data

        # Generate structured requirements using LLM with RAG enhancement
        logger.info("Generating structured requirements...")
        try:
            requirements_structure = await generate_structured_requirements(problem_analysis, enhanced_context)
            logger.info(f"Requirements structure generated with keys: {list(requirements_structure.keys())}")
        except Exception as e:
            logger.error(f"Structured requirements generation failed: {str(e)}")
            requirements_structure = generate_fallback_requirements(problem_analysis, enhanced_context)

        # Generate user journey map with enhanced context
        logger.info("Generating user journey map...")
        try:
            user_journey_map = await generate_user_journey_map(problem_analysis, enhanced_context)
            logger.info(f"User journey map generated, length: {len(user_journey_map)}")
        except Exception as e:
            logger.error(f"User journey map generation failed: {str(e)}")
            user_journey_map = create_fallback_journey_map(problem_analysis, enhanced_context)

        # Create comprehensive SRS document with enhanced context
        logger.info("Creating SRS document...")
        try:
            srs_document = await create_srs_document(
                problem_analysis,
                enhanced_context,
                requirements_structure
            )
            logger.info(f"SRS document generated, length: {len(srs_document)}")
        except Exception as e:
            logger.error(f"SRS document generation failed: {str(e)}")
            srs_document = create_fallback_srs(problem_analysis, enhanced_context, requirements_structure)
        
        # Update conversation history
        conversation_history.append({
            "sender": "requirements_generator",
            "recipient": "system", 
            "message_type": "completion",
            "content": "Requirements specification document has been generated successfully.",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "functional_reqs": len(requirements_structure.get("functional_requirements", [])),
                "user_stories": len(requirements_structure.get("user_stories", [])),
                "srs_length": len(srs_document)
            }
        })
        
        # Update state with generated requirements
        updated_state = state.copy()
        updated_state.update({
            "current_step": "requirements_generated",
            "current_status": "requirements_complete",
            "conversation_history": conversation_history,
            "requirements_doc": srs_document,
            "requirements_definition": srs_document,
            "requirements_document": srs_document,  # Add frontend-expected key
            "user_journey_map": user_journey_map,
            "requires_user_input": False,
            # Store structured components for other agents
            **requirements_structure
        })

        logger.info("=== REQUIREMENTS GENERATION COMPLETED ===")
        logger.info(f"Final state keys: {list(updated_state.keys())}")
        logger.info(f"Requirements document length: {len(srs_document)}")
        logger.info(f"User journey map length: {len(user_journey_map)}")

        return updated_state
        
    except Exception as e:
        logger.error(f"Error in requirements generation: {str(e)}")
        return handle_requirements_error(state, str(e))


async def generate_structured_requirements(
    problem_analysis: Dict[str, Any], 
    context_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate structured requirements using LLM integration.
    
    Args:
        problem_analysis: Structured problem analysis
        context_data: Collected context information
        
    Returns:
        Structured requirements dictionary
    """
    try:
        requirements_prompt = f"""
        You are an expert business analyst and requirements engineer. Generate comprehensive 
        software requirements based on the problem analysis and collected context.
        
        Problem Analysis:
        {json.dumps(problem_analysis, indent=2)}
        
        Context Data:
        {json.dumps(context_data, indent=2)}
        
        Generate a comprehensive requirements structure with the following JSON format:
        {{
            "functional_requirements": [
                {{
                    "id": "FR-001",
                    "title": "Requirement title",
                    "description": "Detailed requirement description",
                    "priority": "HIGH/MEDIUM/LOW",
                    "category": "Core/Interface/Integration/Reporting"
                }}
            ],
            "non_functional_requirements": [
                {{
                    "id": "NFR-001", 
                    "title": "Non-functional requirement title",
                    "description": "Detailed description",
                    "category": "Performance/Security/Usability/Reliability",
                    "metric": "Measurable success criteria"
                }}
            ],
            "user_stories": [
                {{
                    "id": "US-001",
                    "role": "As a [user role]",
                    "goal": "I want to [goal]", 
                    "benefit": "So that [benefit]",
                    "acceptance_criteria": ["Given/When/Then criteria"],
                    "priority": "HIGH/MEDIUM/LOW"
                }}
            ],
            "business_rules": [
                {{
                    "id": "BR-001",
                    "rule": "Business rule description",
                    "rationale": "Why this rule exists"
                }}
            ],
            "constraints": [
                {{
                    "type": "Technical/Business/Legal",
                    "description": "Constraint description",
                    "impact": "Impact on solution"
                }}
            ],
            "assumptions": [
                {{
                    "description": "Assumption description",
                    "risk": "Risk if assumption is wrong"
                }}
            ],
            "success_criteria": [
                {{
                    "metric": "Success metric name",
                    "target": "Target value",
                    "measurement": "How to measure"
                }}
            ]
        }}
        
        Focus on:
        1. Specific, measurable, and testable requirements
        2. Complete coverage of the problem domain
        3. Clear acceptance criteria for user stories
        4. Realistic constraints and assumptions
        5. Quantifiable success metrics
        
        Ensure requirements are specific to the problem category: {problem_analysis.get('category', 'GENERAL')}
        """
        
        # Get the LLM agent service
        agent_service = await get_agent_service()
        
        # Use the specialized requirements generation method
        requirements_text = await agent_service.generate_requirements(problem_analysis, context_data)
        
        # Try to parse as JSON first, otherwise use as markdown
        try:
            requirements_structure = json.loads(requirements_text)
        except json.JSONDecodeError:
            # If not JSON, create structure from markdown content
            requirements_structure = {
                "functional_requirements": [],
                "non_functional_requirements": [],
                "user_stories": [],
                "business_rules": [],
                "constraints": [],
                "assumptions": [],
                "success_criteria": [],
                "requirements_document": requirements_text
            }
        
        # Validate and enhance the structure
        return validate_requirements_structure(requirements_structure)
        
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"LLM requirements generation failed, using fallback: {str(e)}")
        return generate_fallback_requirements(problem_analysis, context_data)


def validate_requirements_structure(requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and enhance requirements structure.
    
    Args:
        requirements: Raw requirements from LLM
        
    Returns:
        Validated requirements structure
    """
    # Ensure all required sections exist
    validated = {
        "functional_requirements": requirements.get("functional_requirements", []),
        "non_functional_requirements": requirements.get("non_functional_requirements", []),
        "user_stories": requirements.get("user_stories", []),
        "business_rules": requirements.get("business_rules", []),
        "constraints": requirements.get("constraints", []),
        "assumptions": requirements.get("assumptions", []),
        "success_criteria": requirements.get("success_criteria", [])
    }
    
    # Validate functional requirements structure
    for i, req in enumerate(validated["functional_requirements"]):
        if not isinstance(req, dict):
            continue
        req.setdefault("id", f"FR-{i+1:03d}")
        req.setdefault("priority", "MEDIUM")
        req.setdefault("category", "Core")
    
    # Validate non-functional requirements
    for i, req in enumerate(validated["non_functional_requirements"]):
        if not isinstance(req, dict):
            continue
        req.setdefault("id", f"NFR-{i+1:03d}")
        req.setdefault("category", "Performance")
    
    # Validate user stories
    for i, story in enumerate(validated["user_stories"]):
        if not isinstance(story, dict):
            continue
        story.setdefault("id", f"US-{i+1:03d}")
        story.setdefault("priority", "MEDIUM")
        story.setdefault("acceptance_criteria", [])
    
    return validated


def generate_fallback_requirements(
    problem_analysis: Dict[str, Any], 
    context_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate fallback requirements when LLM is unavailable.
    
    Args:
        problem_analysis: Problem analysis
        context_data: Context data
        
    Returns:
        Basic requirements structure
    """
    logger.info("Using fallback requirements generation")
    
    category = problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
    
    # Category-specific functional requirements
    functional_templates = {
        "AUTOMATION": [
            {"id": "FR-001", "title": "Process Automation", "description": "The system shall automate the specified manual process", "priority": "HIGH", "category": "Core"},
            {"id": "FR-002", "title": "Error Handling", "description": "The system shall handle errors gracefully and provide recovery options", "priority": "HIGH", "category": "Core"},
            {"id": "FR-003", "title": "Logging", "description": "The system shall log all processing activities for audit purposes", "priority": "MEDIUM", "category": "Reporting"}
        ],
        "INFORMATION_RETRIEVAL": [
            {"id": "FR-001", "title": "Search Functionality", "description": "The system shall provide comprehensive search capabilities across data sources", "priority": "HIGH", "category": "Core"},
            {"id": "FR-002", "title": "Result Ranking", "description": "The system shall rank search results by relevance", "priority": "MEDIUM", "category": "Core"},
            {"id": "FR-003", "title": "Export Results", "description": "The system shall allow users to export search results", "priority": "MEDIUM", "category": "Interface"}
        ],
        "MACHINE_LEARNING": [
            {"id": "FR-001", "title": "Model Training", "description": "The system shall train machine learning models on provided data", "priority": "HIGH", "category": "Core"},
            {"id": "FR-002", "title": "Prediction Service", "description": "The system shall provide prediction services with confidence scores", "priority": "HIGH", "category": "Core"},
            {"id": "FR-003", "title": "Model Validation", "description": "The system shall validate model performance using standard metrics", "priority": "MEDIUM", "category": "Core"}
        ]
    }
    
    # Base non-functional requirements
    non_functional_reqs = [
        {"id": "NFR-001", "title": "Performance", "description": "System response time shall not exceed 5 seconds", "category": "Performance", "metric": "< 5 seconds response time"},
        {"id": "NFR-002", "title": "Reliability", "description": "System shall have 99% uptime availability", "category": "Reliability", "metric": "99% uptime"},
        {"id": "NFR-003", "title": "Usability", "description": "System shall be usable by non-technical users", "category": "Usability", "metric": "User satisfaction > 80%"}
    ]
    
    # Base user stories
    user_stories = [
        {
            "id": "US-001",
            "role": "As an end user",
            "goal": "I want to solve my problem efficiently", 
            "benefit": "So that I can focus on higher-value work",
            "acceptance_criteria": ["Given a valid input, when I use the system, then I get expected results"],
            "priority": "HIGH"
        },
        {
            "id": "US-002", 
            "role": "As an administrator",
            "goal": "I want to monitor system health",
            "benefit": "So that I can ensure reliable operation",
            "acceptance_criteria": ["Given system is running, when I check status, then I see health metrics"],
            "priority": "MEDIUM"
        }
    ]
    
    functional_reqs = functional_templates.get(category, functional_templates["AUTOMATION"])
    
    return {
        "functional_requirements": functional_reqs,
        "non_functional_requirements": non_functional_reqs,
        "user_stories": user_stories,
        "business_rules": [
            {"id": "BR-001", "rule": "All data processing must be auditable", "rationale": "Compliance and debugging requirements"}
        ],
        "constraints": [
            {"type": "Technical", "description": "Must be implemented in Python", "impact": "Technology stack limitation"},
            {"type": "Business", "description": "Must be maintainable by beginners", "impact": "Code complexity constraints"}
        ],
        "assumptions": [
            {"description": "Users have basic computer literacy", "risk": "May need additional training"},
            {"description": "Required data sources are accessible", "risk": "Integration complexity may increase"}
        ],
        "success_criteria": [
            {"metric": "Processing Time", "target": "50% reduction", "measurement": "Before/after comparison"},
            {"metric": "Error Rate", "target": "< 1%", "measurement": "Error logs analysis"}
        ]
    }


async def generate_user_journey_map(
    problem_analysis: Dict[str, Any], 
    context_data: Dict[str, Any]
) -> str:
    """
    Generate user journey map using LLM integration.
    
    Args:
        problem_analysis: Problem analysis
        context_data: Context data
        
    Returns:
        User journey map in markdown format
    """
    try:
        journey_prompt = f"""
        당신은 UX 디자이너이자 비즈니스 분석가입니다. 아래 정보를 바탕으로
        한국어(ko-KR)로 현재(As-Is)와 개선 후(To-Be) 사용자 여정을 마크다운으로 작성하세요.

        Problem Analysis(JSON):
        {json.dumps(problem_analysis, indent=2, ensure_ascii=False)}

        Context Data(JSON):
        {json.dumps(context_data, indent=2, ensure_ascii=False)}

        포함할 내용:
        1) 현재 상태(As-Is): 단계별 활동, 소요 시간, 주요 불편/리스크, 사용 도구
        2) 목표 상태(To-Be): 자동화로 인한 개선점, 시간/비용 절감, UX 향상, 신규 역량
        3) 접점/인터랙션: UI 터치포인트, 시스템 연동, 데이터 흐름
        4) 성공 지표: 정량/정성 KPI

        표/목록을 적절히 사용하고, 도메인({problem_analysis.get('domain', 'Business Process')}) 특화 용어를 사용하세요.
        """
        
        # Get the LLM agent service
        agent_service = await get_agent_service()
        
        llm_response = await agent_service.llm_manager.simple_completion(
            journey_prompt,
            temperature=0.5
        )
        return llm_response
        
    except Exception as e:
        logger.warning(f"LLM journey map generation failed, using fallback: {str(e)}")
        return create_fallback_journey_map(problem_analysis, context_data)


def create_fallback_journey_map(
    problem_analysis: Dict[str, Any], 
    context_data: Dict[str, Any]
) -> str:
    """
    Create fallback user journey map when LLM is unavailable.
    
    Args:
        problem_analysis: Problem analysis
        context_data: Context data
        
    Returns:
        User journey map in markdown
    """
    title = problem_analysis.get("title", "Process Improvement")
    pain_points = problem_analysis.get("pain_points", ["Manual inefficiencies"])
    
    journey_map = f"""# User Journey Map: {title}

## Current State (As-Is)

### Current Process Flow
1. **Manual Problem Identification** (30 minutes)
   - User recognizes process inefficiency
   - Searches for solutions manually
   - *Pain Point: Time-consuming research*

2. **Ad-hoc Solution Attempts** (2-4 hours)
   - Trial and error approach
   - Limited technical knowledge
   - *Pain Point: Inconsistent results*

3. **Manual Execution** (Variable time)
   - Repetitive manual tasks
   - Error-prone processes
   - *Pain Point: High effort, low reliability*

### Current Pain Points
{chr(10).join(f"- {pain_point}" for pain_point in pain_points)}

### User Emotions (Current)
- **Frustrated** with repetitive tasks
- **Overwhelmed** by technical complexity
- **Uncertain** about best practices

## Future State (To-Be)

### Improved Process Flow
1. **Problem Input** (5 minutes)
   - Simple problem description
   - Guided information gathering
   - *Improvement: Quick and intuitive*

2. **Automated Analysis** (2-3 minutes)
   - AI-powered problem breakdown
   - Intelligent recommendations
   - *Improvement: Expert-level analysis*

3. **Solution Implementation** (1-2 hours)
   - Step-by-step guidance
   - Pre-built templates and code
   - *Improvement: Reliable, repeatable results*

### Expected Benefits
- **70% time reduction** in problem-solving
- **90% reduction** in manual errors
- **Improved consistency** in solution quality
- **Enhanced learning** through guided process

### User Emotions (Future)
- **Confident** in solution approach
- **Efficient** in problem resolution
- **Satisfied** with comprehensive guidance

## Touchpoints

### Primary Interfaces
- **Web Interface**: Problem input and progress tracking
- **Generated Documents**: Requirements, guides, recommendations
- **Implementation Tools**: Code examples and templates

### System Integrations
- **LLM Services**: For intelligent analysis and generation
- **Knowledge Base**: For best practices and examples
- **Workflow Engine**: For process orchestration

## Success Metrics

| Metric | Current State | Target State | Measurement |
|--------|---------------|--------------|-------------|
| Time to Solution | 4-8 hours | 1-2 hours | Process timing |
| Success Rate | 60% | 90% | Solution effectiveness |
| User Satisfaction | 3/5 | 4.5/5 | User feedback |
| Consistency | Variable | High | Output quality |

## Implementation Phases

### Phase 1: Core Automation
- Basic problem analysis
- Simple solution generation
- Initial user interface

### Phase 2: Enhanced Intelligence  
- Advanced LLM integration
- Comprehensive documentation
- Improved user experience

### Phase 3: Advanced Features
- Integration capabilities
- Advanced analytics
- Community features

---
*Generated by AI Problem Solving Copilot*
"""
    
    return journey_map


async def create_srs_document(
    problem_analysis: Dict[str, Any], 
    context_data: Dict[str, Any], 
    requirements_structure: Dict[str, Any]
) -> str:
    """
    Create comprehensive SRS document using LLM integration.
    
    Args:
        problem_analysis: Problem analysis
        context_data: Context data  
        requirements_structure: Structured requirements
        
    Returns:
        Complete SRS document in markdown
    """
    try:
        title = problem_analysis.get("title", "소프트웨어 솔루션")
        srs_prompt = f"""
        당신은 뛰어난 기술 문서 작성자이자 비즈니스 분석가입니다. 아래 정보를 바탕으로
        한국어(ko-KR)로 완전한 소프트웨어 요구사항 명세서(SRS)를 마크다운 형식으로 작성하세요.

        - H1 제목은 반드시 다음을 그대로 사용하세요: "소프트웨어 요구사항 명세서 (SRS) - {title}"
        - 문제 도메인과 맥락에 맞는 한국어 용어를 사용하세요.
        - 표와 목록을 적절히 활용하고, 요구사항은 테스트 가능하도록 구체적으로 작성하세요.

        Problem Analysis(JSON):
        {json.dumps(problem_analysis, indent=2, ensure_ascii=False)}

        Context Data(JSON):
        {json.dumps(context_data, indent=2, ensure_ascii=False)}

        Requirements Structure(JSON):
        {json.dumps(requirements_structure, indent=2, ensure_ascii=False)}

        문서 구성 가이드(참고):
        # 소프트웨어 요구사항 명세서 (SRS) - {title}
        ## 1. 개요(Introduction)
        ### 1.1 목적 및 범위
        ### 1.2 시스템 개요
        ### 1.3 용어 정의

        ## 2. 전반적 설명(Overall Description)
        ### 2.1 시스템 관점
        ### 2.2 주요 기능
        ### 2.3 사용자 분류 및 특성
        ### 2.4 운영 환경
        ### 2.5 제약사항

        ## 3. 요구사항(System Features and Requirements)
        ### 3.1 기능 요구사항(식별자, 설명, 수용 기준 포함)
        ### 3.2 비기능 요구사항(성능/보안/신뢰성/사용성 등)
        ### 3.3 사용자 스토리 및 수용 기준

        ## 4. 외부 인터페이스 요구사항
        ## 5. 품질 속성
        ## 6. 기타 요구사항(업무 규칙, 가정/제약, 성공 기준)

        전문적이고 도메인에 특화된 내용으로 작성하세요.
        """
        
        # Get the LLM agent service
        agent_service = await get_agent_service()
        
        llm_response = await agent_service.llm_manager.simple_completion(
            srs_prompt,
            temperature=0.4
        )
        return llm_response
        
    except Exception as e:
        logger.warning(f"LLM SRS generation failed, using fallback: {str(e)}")
        return create_fallback_srs(problem_analysis, context_data, requirements_structure)


def create_fallback_srs(
    problem_analysis: Dict[str, Any], 
    context_data: Dict[str, Any], 
    requirements_structure: Dict[str, Any]
) -> str:
    """
    Create fallback SRS document when LLM is unavailable.
    
    Args:
        problem_analysis: Problem analysis
        context_data: Context data
        requirements_structure: Requirements structure
        
    Returns:
        SRS document in markdown
    """
    title = problem_analysis.get("title", "소프트웨어 솔루션")
    domain = problem_analysis.get("domain", "업무 프로세스")
    
    # Build functional requirements section
    functional_reqs = requirements_structure.get("functional_requirements", [])
    functional_section = ""
    for req in functional_reqs:
        functional_section += f"- **{req.get('id', 'FR-XXX')}**: {req.get('description', 'Requirement description')}\n"
    
    # Build user stories section
    user_stories = requirements_structure.get("user_stories", [])
    stories_section = ""
    for story in user_stories:
        stories_section += f"- **{story.get('id', 'US-XXX')}**: {story.get('role', 'As a user')}, {story.get('goal', 'I want functionality')}, {story.get('benefit', 'so that I get value')}\n"
    
    srs_document = f"""# 소프트웨어 요구사항 명세서 (SRS) - {title}

## 1. 개요(Introduction)

### 1.1 목적
본 문서는 {domain} 도메인에서 {title} 시스템의 요구사항을 정의합니다. 설계/개발/테스트의 기준 문서로 사용됩니다.

### 1.2 범위
- 현재 상태: {problem_analysis.get('current_state', '수작업 중심의 현행 프로세스')}
- 목표 상태: {problem_analysis.get('desired_state', '자동화를 통한 효율화')}

### 1.3 이해관계자
{chr(10).join(f"- {stakeholder}" for stakeholder in problem_analysis.get('stakeholders', ['최종 사용자']))}

## 2. 전반적 설명(Overall Description)

### 2.1 시스템 관점
이 시스템은 {problem_analysis.get('category', 'GENERAL')} 유형의 솔루션으로, {domain} 프로세스 개선을 목표로 합니다.

### 2.2 주요 기능
- 사용자 요청 자동 처리
- 지능형 분석 및 추천
- 보고서/문서화 제공
- 오류 처리 및 복구 기능

### 2.3 사용자 분류
주요 사용자:
{chr(10).join(f"- {stakeholder}" for stakeholder in problem_analysis.get('stakeholders', ['최종 사용자']))}

### 2.4 운영 환경
- **플랫폼**: Python 기반
- **의존성**: 표준 라이브러리 및 명시된 패키지
- **연동**: 기존 업무 시스템과의 연계 지원

### 2.5 제약사항
{chr(10).join(f"- {constraint.get('description', '제약사항')}" for constraint in requirements_structure.get('constraints', []))}

## 3. 기능 요구사항(Functional Requirements)

{functional_section}

## 4. Non-Functional Requirements

### 4.1 Performance
- System response time shall not exceed 5 seconds for typical operations
- The system shall handle concurrent users efficiently

### 4.2 Reliability
- System uptime shall be 99% or higher
- Data integrity shall be maintained at all times

### 4.3 Usability
- The system shall be usable by non-technical users
- Clear error messages and guidance shall be provided

### 4.4 Security
- User data shall be protected according to industry standards
- Access controls shall be implemented where required

## 5. User Stories

{stories_section}

## 6. Business Rules

{chr(10).join(f"- **{rule.get('id', 'BR-XXX')}**: {rule.get('rule', 'Business rule')}" for rule in requirements_structure.get('business_rules', []))}

## 7. Success Criteria

{chr(10).join(f"- **{criteria.get('metric', 'Metric')}**: {criteria.get('target', 'Target')} ({criteria.get('measurement', 'Measurement method')})" for criteria in requirements_structure.get('success_criteria', []))}

## 8. Assumptions

{chr(10).join(f"- {assumption.get('description', 'Assumption')}" for assumption in requirements_structure.get('assumptions', []))}

---
*Generated by AI Problem Solving Copilot*
*Document Version: 1.0*
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return srs_document


def handle_requirements_error(state: WorkflowState, error_message: str) -> WorkflowState:
    """
    Handle errors during requirements generation.
    
    Args:
        state: Current workflow state
        error_message: Error description
        
    Returns:
        Updated state with error information
    """
    logger.error(f"Requirements generation failed: {error_message}")
    
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({
        "sender": "requirements_generator", 
        "recipient": "system",
        "message_type": "error",
        "content": f"Requirements generation error: {error_message}",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"error_type": "requirements_generation_error"}
    })
    
    # Create error state
    updated_state = state.copy()
    updated_state.update({
        "current_step": "requirements_error",
        "current_status": "error", 
        "error_message": error_message,
        "conversation_history": conversation_history,
        "retry_count": state.get("retry_count", 0) + 1,
        "requires_user_input": False
    })
    
    return updated_state
