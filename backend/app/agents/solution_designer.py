"""
Solution Design Agent

This agent analyzes requirements and designs appropriate solution architecture,
selecting suitable solution types and technology stacks using LLM integration.
Implements the design_solution node for the LangGraph workflow with routing capabilities.
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from app.workflows.state import WorkflowState
from app.appendix.internal_llm import get_agent_service, LLMAgentService
from app.appendix.rag_retrieve import enhance_llm_context, retrieve_solution_examples, get_technology_recommendations


logger = logging.getLogger(__name__)


async def design_solution(state: WorkflowState) -> WorkflowState:
    """
    Design solution architecture and select appropriate technology stack using LLM integration.
    
    Analyzes requirements and problem characteristics to determine the best
    solution approach and recommend suitable technologies for routing decisions.
    
    Args:
        state: Current workflow state
        
    Returns:
        Updated state with solution design and routing information
    """
    try:
        logger.info("Starting solution design")
        
        problem_analysis = state.get("problem_analysis", {})
        context_data = state.get("context_data", {})
        functional_requirements = state.get("functional_requirements", [])
        conversation_history = state.get("conversation_history", [])
        
        # Get enhanced context for solution design
        enhanced_context = await enhance_llm_context(
            agent_type="solution_designer",
            query=f"solution design {problem_analysis.get('category', '')} architecture",
            current_context=context_data,
            domain="web_development"
        )
        
        # Get technology recommendations based on requirements
        tech_requirements = {
            "domain": problem_analysis.get("domain", "general"),
            "complexity": problem_analysis.get("complexity", "medium"),
            "constraints": problem_analysis.get("constraints", []),
            "features": [req.get("title", "") for req in functional_requirements[:5]]
        }
        tech_recommendations = await get_technology_recommendations(tech_requirements)
        
        # Get solution examples for reference
        problem_type = problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
        tech_stack = [rec["technology"] for rec in tech_recommendations[:3] if "technology" in rec]
        solution_examples = await retrieve_solution_examples(problem_type, tech_stack, problem_analysis.get("domain"))
        
        # Determine solution type using LLM with RAG enhancement
        solution_design = await create_comprehensive_solution_design(
            problem_analysis, 
            enhanced_context, 
            functional_requirements,
            tech_recommendations,
            solution_examples
        )
        
        # Generate technology recommendations
        tech_recommendations = await generate_technology_recommendations(
            solution_design.get("solution_type", "SIMPLE_AUTOMATION"),
            problem_analysis,
            context_data,
            rag_context
        )
        
        # Create architecture overview
        architecture_overview = await create_architecture_overview(
            solution_design, 
            tech_recommendations,
            problem_analysis
        )
        
        # Update conversation history
        conversation_history.append({
            "sender": "solution_designer",
            "recipient": "system",
            "message_type": "completion",
            "content": f"Solution designed with type: {solution_design.get('solution_type')} and comprehensive technology stack.",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "solution_type": solution_design.get("solution_type"),
                "complexity": solution_design.get("complexity_assessment"),
                "tech_count": len(tech_recommendations.get("primary_technologies", []))
            }
        })
        
        # Update state with solution design
        updated_state = state.copy()
        updated_state.update({
            "current_step": "solution_designed",
            "current_status": "solution_complete",
            "conversation_history": conversation_history,
            "solution_type": solution_design.get("solution_type"),
            "recommended_solution_type": solution_design.get("solution_type"),
            "technology_stack": tech_recommendations,
            "implementation_plan": solution_design.get("implementation_plan", ""),
            "requires_user_input": False,
            # Store detailed design components
            **solution_design
        })
        
        logger.info(f"Solution design completed. Type: {solution_design.get('solution_type')}")
        
        return updated_state
        
    except Exception as e:
        logger.error(f"Error in solution design: {str(e)}")
        return handle_solution_error(state, str(e))


async def get_enhanced_rag_context(
    problem_analysis: Dict[str, Any], 
    context_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Get enhanced context from RAG system for better solution design.
    
    Args:
        problem_analysis: Problem analysis results
        context_data: Collected context data
        
    Returns:
        Enhanced context from RAG system
    """
    try:
        # Construct search query for RAG system
        category = problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
        domain = problem_analysis.get("domain", "Business Process")
        
        search_query = f"best practices {category.lower()} {domain.lower()} python implementation"
        
        # Get RAG context (fallback to empty dict if service unavailable)
        rag_results = await get_rag_context(search_query, max_results=5)
        
        return {
            "search_query": search_query,
            "results": rag_results,
            "has_context": len(rag_results) > 0
        }
        
    except Exception as e:
        logger.warning(f"RAG context retrieval failed: {str(e)}")
        return {"search_query": "", "results": [], "has_context": False}


async def create_comprehensive_solution_design(
    problem_analysis: Dict[str, Any], 
    context_data: Dict[str, Any], 
    functional_requirements: List[Dict[str, Any]],
    rag_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create comprehensive solution design using LLM with RAG enhancement.
    
    Args:
        problem_analysis: Problem analysis results
        context_data: Collected context data
        functional_requirements: Functional requirements list
        rag_context: Enhanced context from RAG
        
    Returns:
        Comprehensive solution design dictionary
    """
    try:
        # Prepare requirements text for LLM
        requirements_text = ""
        for req in functional_requirements:
            if isinstance(req, dict):
                requirements_text += f"- {req.get('description', str(req))}\n"
            else:
                requirements_text += f"- {str(req)}\n"
        
        # Prepare RAG context for LLM
        rag_context_text = ""
        if rag_context.get("has_context"):
            rag_context_text = f"\nRelevant best practices and examples:\n"
            for result in rag_context.get("results", [])[:3]:  # Limit to top 3
                rag_context_text += f"- {result}\n"
        
        design_prompt = f"""
        You are an expert solution architect and technology consultant. Design a comprehensive 
        solution based on the problem analysis, context, and requirements.
        
        Problem Analysis:
        {json.dumps(problem_analysis, indent=2)}
        
        Context Data:
        {json.dumps(context_data, indent=2)}
        
        Functional Requirements:
        {requirements_text}
        
        {rag_context_text}
        
        Provide a comprehensive solution design with the following JSON structure:
        {{
            "solution_type": "One of: SIMPLE_AUTOMATION, COMPLEX_AUTOMATION, RAG, ADVANCED_RAG, ML_CLASSIFICATION, ML_ADVANCED, DASHBOARD, API_INTEGRATION, HYBRID_SOLUTION",
            "solution_category": "Primary category for routing",
            "complexity_assessment": "LOW, MEDIUM, HIGH",
            "recommended_approach": "Detailed approach description",
            "architecture_pattern": "Architectural pattern (e.g., MVC, Microservices, Monolithic)",
            "data_flow": "Description of data flow through the system",
            "core_components": [
                {{
                    "name": "Component name",
                    "description": "Component description",
                    "technology": "Recommended technology",
                    "purpose": "Core/Interface/Integration/Storage"
                }}
            ],
            "integration_points": [
                {{
                    "system": "External system name",
                    "method": "Integration method",
                    "data_format": "Data format",
                    "frequency": "Integration frequency"
                }}
            ],
            "scalability_considerations": [
                "Scalability factor descriptions"
            ],
            "security_requirements": [
                "Security consideration descriptions"
            ],
            "deployment_strategy": {{
                "type": "Local/Cloud/Hybrid",
                "platform": "Deployment platform",
                "requirements": ["Platform requirements"]
            }},
            "testing_approach": {{
                "unit_testing": "Unit testing strategy",
                "integration_testing": "Integration testing approach",
                "performance_testing": "Performance testing plan"
            }},
            "implementation_phases": [
                {{
                    "phase": "Phase name",
                    "duration": "Estimated duration",
                    "deliverables": ["Phase deliverables"],
                    "dependencies": ["Dependencies"]
                }}
            ],
            "risk_assessment": [
                {{
                    "risk": "Risk description",
                    "probability": "LOW/MEDIUM/HIGH",
                    "impact": "LOW/MEDIUM/HIGH",
                    "mitigation": "Mitigation strategy"
                }}
            ],
            "success_metrics": [
                {{
                    "metric": "Metric name",
                    "target": "Target value",
                    "measurement": "How to measure"
                }}
            ]
        }}
        
        Solution Type Guidelines:
        - SIMPLE_AUTOMATION: Basic scripting, file processing, simple workflows
        - COMPLEX_AUTOMATION: Advanced workflows, multiple integrations, orchestration
        - RAG: Basic retrieval augmented generation systems
        - ADVANCED_RAG: Complex RAG with multiple data sources, advanced embeddings
        - ML_CLASSIFICATION: Basic machine learning classification tasks
        - ML_ADVANCED: Complex ML with multiple models, deep learning
        - DASHBOARD: Data visualization and reporting systems
        - API_INTEGRATION: System integration and API orchestration
        - HYBRID_SOLUTION: Combination of multiple solution types
        
        Focus on:
        1. Selecting the most appropriate solution type for routing
        2. Comprehensive technical architecture
        3. Realistic implementation planning
        4. Risk mitigation strategies
        5. Measurable success criteria
        """
        
        # Get the LLM agent service
        agent_service = await get_agent_service()
        
        # Use the specialized solution design method
        requirements_text = context_data.get("requirements_document", json.dumps(functional_requirements))
        solution_design = await agent_service.design_solution(requirements_text, problem_analysis)
        
        # Validate and enhance the design
        return validate_solution_design(solution_design, problem_analysis)
        
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"LLM solution design failed, using fallback: {str(e)}")
        return create_fallback_solution_design(problem_analysis, context_data, functional_requirements)


def validate_solution_design(design: Dict[str, Any], problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and enhance solution design structure.
    
    Args:
        design: Raw design from LLM
        problem_analysis: Problem analysis for validation
        
    Returns:
        Validated solution design
    """
    # Valid solution types for routing
    valid_solution_types = {
        "SIMPLE_AUTOMATION", "COMPLEX_AUTOMATION", "RAG", "ADVANCED_RAG",
        "ML_CLASSIFICATION", "ML_ADVANCED", "DASHBOARD", "API_INTEGRATION", "HYBRID_SOLUTION"
    }
    
    # Ensure solution type is valid
    solution_type = design.get("solution_type", "SIMPLE_AUTOMATION")
    if solution_type not in valid_solution_types:
        # Map to valid solution type based on problem category
        category = problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
        solution_type = map_category_to_solution_type(category)
    
    # Set solution category for routing
    solution_category = get_solution_category(solution_type)
    
    # Ensure all required fields exist
    validated = {
        "solution_type": solution_type,
        "solution_category": solution_category,
        "complexity_assessment": design.get("complexity_assessment", "MEDIUM"),
        "recommended_approach": design.get("recommended_approach", "Iterative development approach"),
        "architecture_pattern": design.get("architecture_pattern", "Monolithic"),
        "data_flow": design.get("data_flow", "Input → Processing → Output"),
        "core_components": design.get("core_components", []),
        "integration_points": design.get("integration_points", []),
        "scalability_considerations": design.get("scalability_considerations", []),
        "security_requirements": design.get("security_requirements", []),
        "deployment_strategy": design.get("deployment_strategy", {}),
        "testing_approach": design.get("testing_approach", {}),
        "implementation_phases": design.get("implementation_phases", []),
        "risk_assessment": design.get("risk_assessment", []),
        "success_metrics": design.get("success_metrics", [])
    }
    
    # Validate complexity
    if validated["complexity_assessment"] not in ["LOW", "MEDIUM", "HIGH"]:
        validated["complexity_assessment"] = "MEDIUM"
    
    return validated


def map_category_to_solution_type(category: str) -> str:
    """
    Map problem category to appropriate solution type.
    
    Args:
        category: Problem category
        
    Returns:
        Mapped solution type
    """
    mapping = {
        "AUTOMATION": "SIMPLE_AUTOMATION",
        "INFORMATION_RETRIEVAL": "RAG",
        "MACHINE_LEARNING": "ML_CLASSIFICATION",
        "DATA_VISUALIZATION": "DASHBOARD",
        "INTEGRATION": "API_INTEGRATION",
        "GENERAL_PROBLEM_SOLVING": "SIMPLE_AUTOMATION"
    }
    
    return mapping.get(category, "SIMPLE_AUTOMATION")


def get_solution_category(solution_type: str) -> str:
    """
    Get solution category for routing purposes.
    
    Args:
        solution_type: Solution type
        
    Returns:
        Solution category
    """
    category_mapping = {
        "SIMPLE_AUTOMATION": "AUTOMATION",
        "COMPLEX_AUTOMATION": "AUTOMATION",
        "RAG": "RETRIEVAL",
        "ADVANCED_RAG": "RETRIEVAL",
        "ML_CLASSIFICATION": "MACHINE_LEARNING",
        "ML_ADVANCED": "MACHINE_LEARNING",
        "DASHBOARD": "VISUALIZATION",
        "API_INTEGRATION": "INTEGRATION",
        "HYBRID_SOLUTION": "HYBRID"
    }
    
    return category_mapping.get(solution_type, "AUTOMATION")


def create_fallback_solution_design(
    problem_analysis: Dict[str, Any], 
    context_data: Dict[str, Any], 
    functional_requirements: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Create fallback solution design when LLM is unavailable.
    
    Args:
        problem_analysis: Problem analysis
        context_data: Context data
        functional_requirements: Functional requirements
        
    Returns:
        Basic solution design
    """
    logger.info("Using fallback solution design")
    
    category = problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
    complexity = problem_analysis.get("complexity", "MEDIUM")
    
    # Map to solution type
    solution_type = map_category_to_solution_type(category)
    solution_category = get_solution_category(solution_type)
    
    return {
        "solution_type": solution_type,
        "solution_category": solution_category,
        "complexity_assessment": complexity,
        "recommended_approach": "Iterative development with Python-based implementation",
        "architecture_pattern": "Monolithic",
        "data_flow": "Input → Processing → Output → Storage/Report",
        "core_components": [
            {
                "name": "Main Processor",
                "description": "Core processing logic",
                "technology": "Python",
                "purpose": "Core"
            },
            {
                "name": "Configuration Manager",
                "description": "Settings and configuration",
                "technology": "JSON/YAML",
                "purpose": "Core"
            },
            {
                "name": "Logger",
                "description": "Activity logging and monitoring",
                "technology": "Python logging",
                "purpose": "Core"
            }
        ],
        "integration_points": [],
        "scalability_considerations": [
            "Design for single-user operation initially",
            "Consider batch processing for large datasets",
            "Plan for future multi-user support"
        ],
        "security_requirements": [
            "Protect sensitive data",
            "Implement input validation",
            "Use secure file operations"
        ],
        "deployment_strategy": {
            "type": "Local",
            "platform": "Local Python environment",
            "requirements": ["Python 3.8+", "Required packages"]
        },
        "testing_approach": {
            "unit_testing": "pytest for individual components",
            "integration_testing": "End-to-end workflow testing",
            "performance_testing": "Load testing with sample data"
        },
        "implementation_phases": [
            {
                "phase": "Foundation",
                "duration": "1-2 weeks",
                "deliverables": ["Basic structure", "Core components"],
                "dependencies": ["Environment setup"]
            },
            {
                "phase": "Core Development",
                "duration": "2-3 weeks",
                "deliverables": ["Main functionality", "Testing"],
                "dependencies": ["Foundation phase"]
            },
            {
                "phase": "Integration & Polish",
                "duration": "1 week",
                "deliverables": ["Final integration", "Documentation"],
                "dependencies": ["Core development"]
            }
        ],
        "risk_assessment": [
            {
                "risk": "Technical complexity underestimated",
                "probability": "MEDIUM",
                "impact": "MEDIUM",
                "mitigation": "Prototype early, iterative development"
            },
            {
                "risk": "Data quality issues",
                "probability": "MEDIUM",
                "impact": "HIGH",
                "mitigation": "Implement robust data validation"
            }
        ],
        "success_metrics": [
            {
                "metric": "Implementation Time",
                "target": "4-6 weeks",
                "measurement": "Project timeline tracking"
            },
            {
                "metric": "Solution Effectiveness",
                "target": "Meets functional requirements",
                "measurement": "Requirements traceability"
            }
        ]
    }


async def generate_technology_recommendations(
    solution_type: str,
    problem_analysis: Dict[str, Any],
    context_data: Dict[str, Any],
    rag_context: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate comprehensive technology recommendations using LLM.
    
    Args:
        solution_type: Determined solution type
        problem_analysis: Problem analysis
        context_data: Context data
        rag_context: RAG context
        
    Returns:
        Technology recommendations dictionary
    """
    try:
        tech_prompt = f"""
        You are an expert technology consultant. Recommend a comprehensive technology stack 
        for the following solution.
        
        Solution Type: {solution_type}
        Problem Category: {problem_analysis.get('category', 'GENERAL')}
        Complexity: {problem_analysis.get('complexity', 'MEDIUM')}
        
        Context Requirements:
        {json.dumps(context_data, indent=2)}
        
        Provide technology recommendations in the following JSON format:
        {{
            "primary_technologies": [
                {{
                    "name": "Technology name",
                    "version": "Recommended version", 
                    "purpose": "What it's used for",
                    "priority": "CRITICAL/HIGH/MEDIUM/LOW",
                    "installation": "pip install command",
                    "documentation": "Documentation URL"
                }}
            ],
            "alternative_technologies": [
                {{
                    "name": "Alternative name",
                    "use_case": "When to use this instead",
                    "pros": ["Advantages"],
                    "cons": ["Disadvantages"]
                }}
            ],
            "development_tools": [
                {{
                    "tool": "Tool name",
                    "purpose": "Development purpose",
                    "installation": "How to install"
                }}
            ],
            "deployment_technologies": [
                {{
                    "technology": "Deployment technology",
                    "scenario": "When to use",
                    "complexity": "Setup complexity"
                }}
            ],
            "architecture_decisions": [
                {{
                    "decision": "Technical decision",
                    "rationale": "Why this choice",
                    "alternatives": ["Other options considered"]
                }}
            ]
        }}
        
        Focus on:
        1. Python ecosystem libraries and frameworks
        2. Appropriate complexity level for the solution
        3. Maintainability by Python beginners
        4. Industry best practices
        5. Cost-effective solutions
        """
        
        # Get the LLM agent service
        agent_service = await get_agent_service()
        
        llm_response = await agent_service.llm_manager.simple_completion(
            tech_prompt,
            temperature=0.4
        )
        
        try:
            tech_recommendations = json.loads(llm_response)
        except json.JSONDecodeError:
            # If not JSON, create a basic structure
            tech_recommendations = {
                "primary_technologies": [],
                "supporting_libraries": [],
                "deployment_platform": "Local Python environment",
                "raw_response": llm_response
            }
        
        return validate_tech_recommendations(tech_recommendations, solution_type)
        
    except (json.JSONDecodeError, Exception) as e:
        logger.warning(f"LLM tech recommendations failed, using fallback: {str(e)}")
        return create_fallback_tech_recommendations(solution_type, problem_analysis)


def validate_tech_recommendations(recommendations: Dict[str, Any], solution_type: str) -> Dict[str, Any]:
    """
    Validate and enhance technology recommendations.
    
    Args:
        recommendations: Raw recommendations from LLM
        solution_type: Solution type
        
    Returns:
        Validated recommendations
    """
    # Ensure all sections exist
    validated = {
        "primary_technologies": recommendations.get("primary_technologies", []),
        "alternative_technologies": recommendations.get("alternative_technologies", []),
        "development_tools": recommendations.get("development_tools", []),
        "deployment_technologies": recommendations.get("deployment_technologies", []),
        "architecture_decisions": recommendations.get("architecture_decisions", [])
    }
    
    # Validate primary technologies structure
    for tech in validated["primary_technologies"]:
        if isinstance(tech, dict):
            tech.setdefault("priority", "MEDIUM")
            tech.setdefault("installation", f"pip install {tech.get('name', 'package')}")
    
    return validated


def create_fallback_tech_recommendations(
    solution_type: str, 
    problem_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Create fallback technology recommendations when LLM is unavailable.
    
    Args:
        solution_type: Solution type
        problem_analysis: Problem analysis
        
    Returns:
        Basic technology recommendations
    """
    # Base technologies for all solutions
    base_tech = [
        {
            "name": "python",
            "version": "3.8+",
            "purpose": "Core programming language",
            "priority": "CRITICAL",
            "installation": "Built-in",
            "documentation": "https://docs.python.org/"
        },
        {
            "name": "pytest",
            "version": "latest",
            "purpose": "Testing framework",
            "priority": "HIGH",
            "installation": "pip install pytest",
            "documentation": "https://docs.pytest.org/"
        }
    ]
    
    # Solution-specific technologies
    solution_tech = {
        "SIMPLE_AUTOMATION": [
            {"name": "schedule", "version": "latest", "purpose": "Task scheduling", "priority": "MEDIUM"},
            {"name": "pathlib", "version": "built-in", "purpose": "File operations", "priority": "HIGH"}
        ],
        "RAG": [
            {"name": "langchain", "version": "latest", "purpose": "LLM framework", "priority": "CRITICAL"},
            {"name": "chromadb", "version": "latest", "purpose": "Vector database", "priority": "HIGH"}
        ],
        "ML_CLASSIFICATION": [
            {"name": "scikit-learn", "version": "latest", "purpose": "Machine learning", "priority": "CRITICAL"},
            {"name": "pandas", "version": "latest", "purpose": "Data manipulation", "priority": "HIGH"}
        ],
        "DASHBOARD": [
            {"name": "streamlit", "version": "latest", "purpose": "Web dashboard", "priority": "CRITICAL"},
            {"name": "plotly", "version": "latest", "purpose": "Interactive charts", "priority": "HIGH"}
        ]
    }
    
    primary_tech = base_tech + solution_tech.get(solution_type, [])
    
    return {
        "primary_technologies": primary_tech,
        "alternative_technologies": [
            {
                "name": "Alternative frameworks",
                "use_case": "When specific requirements differ",
                "pros": ["Specialized features"],
                "cons": ["Learning curve"]
            }
        ],
        "development_tools": [
            {"tool": "VS Code", "purpose": "Code editing", "installation": "Download from website"},
            {"tool": "Git", "purpose": "Version control", "installation": "Download from git-scm.com"}
        ],
        "deployment_technologies": [
            {"technology": "Local Python", "scenario": "Development and testing", "complexity": "Low"},
            {"technology": "Docker", "scenario": "Containerized deployment", "complexity": "Medium"}
        ],
        "architecture_decisions": [
            {
                "decision": "Python-based solution",
                "rationale": "Matches team skills and requirements",
                "alternatives": ["Other languages considered"]
            }
        ]
    }


async def create_architecture_overview(
    solution_design: Dict[str, Any],
    tech_recommendations: Dict[str, Any],
    problem_analysis: Dict[str, Any]
) -> str:
    """
    Create comprehensive architecture overview documentation.
    
    Args:
        solution_design: Solution design details
        tech_recommendations: Technology recommendations
        problem_analysis: Problem analysis
        
    Returns:
        Architecture overview in markdown format
    """
    solution_type = solution_design.get("solution_type", "SIMPLE_AUTOMATION")
    components = solution_design.get("core_components", [])
    primary_tech = tech_recommendations.get("primary_technologies", [])
    
    overview = f"""# Architecture Overview: {problem_analysis.get('title', 'Solution')}

## Solution Type: {solution_type}

### High-Level Architecture
**Pattern**: {solution_design.get('architecture_pattern', 'Monolithic')}
**Complexity**: {solution_design.get('complexity_assessment', 'MEDIUM')}

### Data Flow
{solution_design.get('data_flow', 'Input → Processing → Output')}

### Core Components
"""
    
    for component in components:
        if isinstance(component, dict):
            overview += f"- **{component.get('name', 'Component')}**: {component.get('description', 'Description')}\n"
    
    overview += f"\n### Technology Stack\n"
    for tech in primary_tech:
        if isinstance(tech, dict):
            overview += f"- **{tech.get('name', 'Technology')}**: {tech.get('purpose', 'Purpose')}\n"
    
    overview += f"\n### Implementation Approach\n{solution_design.get('recommended_approach', 'Standard implementation approach')}\n"
    
    phases = solution_design.get("implementation_phases", [])
    if phases:
        overview += f"\n### Implementation Phases\n"
        for phase in phases:
            if isinstance(phase, dict):
                overview += f"**{phase.get('phase', 'Phase')}** ({phase.get('duration', 'TBD')}): {', '.join(phase.get('deliverables', []))}\n"
    
    overview += f"\n---\n*Generated by AI Problem Solving Copilot*"
    
    return overview


def handle_solution_error(state: WorkflowState, error_message: str) -> WorkflowState:
    """
    Handle errors during solution design.
    
    Args:
        state: Current workflow state
        error_message: Error description
        
    Returns:
        Updated state with error information
    """
    logger.error(f"Solution design failed: {error_message}")
    
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({
        "sender": "solution_designer",
        "recipient": "system",
        "message_type": "error",
        "content": f"Solution design error: {error_message}",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"error_type": "solution_design_error"}
    })
    
    # Create error state with fallback solution type
    problem_analysis = state.get("problem_analysis", {})
    fallback_solution_type = map_category_to_solution_type(
        problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
    )
    
    updated_state = state.copy()
    updated_state.update({
        "current_step": "solution_error",
        "current_status": "error",
        "error_message": error_message,
        "conversation_history": conversation_history,
        "retry_count": state.get("retry_count", 0) + 1,
        "solution_type": fallback_solution_type,  # Provide fallback for routing
        "recommended_solution_type": fallback_solution_type,
        "requires_user_input": False
    })
    
    return updated_state