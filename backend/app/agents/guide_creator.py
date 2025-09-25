"""
Implementation Guide Creator Agent

This agent creates comprehensive implementation guides with step-by-step
instructions, code examples, and testing strategies using LLM integration.
Implements the create_guide node for the LangGraph workflow.
"""

import json
import logging
from typing import List, Dict, Any
from datetime import datetime

from app.workflows.state import WorkflowState
from app.appendix.internal_llm import get_agent_service, LLMAgentService
from app.appendix.rag_retrieve import enhance_llm_context, retrieve_solution_examples

logger = logging.getLogger(__name__)


async def create_guide(state: WorkflowState) -> WorkflowState:
    """
    Create comprehensive implementation guide using LLM integration.

    Generates detailed build plans with code examples, testing strategies,
    and deployment instructions based on the designed solution.

    Args:
        state: Current workflow state

    Returns:
        Updated state with implementation guide
    """
    try:
        # Force debug output to file
        with open("debug_workflow.txt", "a", encoding="utf-8") as f:
            f.write(f"\n=== GUIDE_CREATOR CALLED at {datetime.now()} ===\n")
            f.write(f"State keys: {list(state.keys())}\n")

        logger.info("=== STARTING GUIDE CREATION ===")
        logger.info(f"Current state keys: {list(state.keys())}")

        problem_analysis = state.get("problem_analysis", {})
        context_data = state.get("context_data", {})
        requirements_doc = state.get("requirements_definition", "")
        solution_design = state.get("solution_design", {})
        technology_stack = state.get("technology_stack", {})
        conversation_history = state.get("conversation_history", [])

        logger.info(f"Requirements doc length: {len(requirements_doc)}")
        logger.info(f"Solution design keys: {list(solution_design.keys()) if solution_design else 'No solution design'}")
        logger.info(f"Technology stack keys: {list(technology_stack.keys()) if technology_stack else 'No technology stack'}")
        
        # Enhance context with RAG for implementation guide creation
        enhanced_context = await enhance_llm_context(
            agent_type="guide_creator",
            query=f"implementation guide {solution_design.get('solution_type', '')} {technology_stack.get('primary_framework', '')}",
            current_context=context_data,
            domain="api_development"
        )
        
        # Get solution examples for code reference
        solution_type = solution_design.get("solution_type", "SIMPLE_AUTOMATION")
        tech_stack_list = [
            technology_stack.get("primary_framework", ""),
            technology_stack.get("database", ""),
            technology_stack.get("web_framework", "")
        ]
        tech_stack_list = [tech for tech in tech_stack_list if tech]  # Remove empty strings
        
        solution_examples = await retrieve_solution_examples(
            solution_type, 
            tech_stack_list, 
            problem_analysis.get("domain")
        )
        
        # Create comprehensive implementation guide using LLM with RAG enhancement
        implementation_guide = await create_comprehensive_guide(
            problem_analysis,
            enhanced_context,
            requirements_doc,
            solution_design,
            technology_stack,
            solution_examples
        )
        
        # Generate additional technical documentation
        technical_specs = await generate_technical_specifications(
            solution_design,
            technology_stack,
            problem_analysis
        )
        
        # Create deployment checklist
        deployment_checklist = await create_deployment_checklist(
            technology_stack,
            problem_analysis.get("complexity", "MEDIUM")
        )
        
        # Update conversation history
        conversation_history.append({
            "sender": "guide_creator",
            "recipient": "system",
            "message_type": "completion",
            "content": "Implementation guide with technical specifications and deployment checklist has been generated successfully.",
            "timestamp": datetime.now().isoformat(),
            "metadata": {
                "guide_length": len(implementation_guide),
                "tech_stack_items": len(technology_stack.get("primary_technologies", [])),
                "deployment_steps": len(deployment_checklist.get("steps", []))
            }
        })
        
        # Update state with generated guide and documentation while preserving previous results
        updated_state = state.copy()
        updated_state.update({
            "current_step": "guide_created",
            "current_status": "complete",
            "conversation_history": conversation_history,
            "implementation_plan": implementation_guide,
            "implementation_guide": implementation_guide,  # Add frontend-expected key
            "technical_specifications": technical_specs,
            "deployment_checklist": deployment_checklist,
            "requires_user_input": False,
            "workflow_complete": True,
            # Preserve all previous stage results
            "requirements_document": state.get("requirements_document"),
            "user_journey_map": state.get("user_journey_map"),
            "requirements_definition": state.get("requirements_definition"),
            "solution_type": state.get("solution_type"),
            "recommended_solution_type": state.get("recommended_solution_type"),
            "technology_stack": state.get("technology_stack"),
            "tech_recommendations": state.get("tech_recommendations"),
            "tech_stack": state.get("tech_stack")
        })

        logger.info("=== GUIDE CREATION COMPLETED ===")
        logger.info(f"Final state keys: {list(updated_state.keys())}")
        logger.info(f"Implementation guide length: {len(implementation_guide)}")
        logger.info(f"Preserved requirements_document: {bool(updated_state.get('requirements_document'))}")
        logger.info(f"Preserved user_journey_map: {bool(updated_state.get('user_journey_map'))}")

        return updated_state
        
    except Exception as e:
        logger.error(f"Error in guide creation: {str(e)}")
        return handle_guide_error(state, str(e))


async def create_comprehensive_guide(
    problem_analysis: Dict[str, Any],
    context_data: Dict[str, Any],
    requirements_doc: str,
    solution_design: Dict[str, Any],
    technology_stack: Dict[str, Any],
    solution_examples: Dict[str, Any] = None
) -> str:
    """
    Create comprehensive implementation guide using LLM integration.
    
    Args:
        problem_analysis: Problem analysis results
        context_data: Collected context
        requirements_doc: Requirements document
        solution_design: Solution design information
        technology_stack: Technology recommendations
        solution_examples: Solution examples from RAG (optional)
        
    Returns:
        Comprehensive implementation guide in markdown
    """
    try:
        # Get the LLM agent service
        agent_service = await get_agent_service()
        
        # Use the specialized implementation guide creation method
        implementation_guide = await agent_service.create_implementation_guide(
            solution_design,
            requirements_doc
        )
        
        return implementation_guide
        
    except Exception as e:
        logger.warning(f"LLM guide generation failed, using fallback: {str(e)}")
        return create_fallback_guide(problem_analysis, context_data, solution_design, technology_stack)


def create_fallback_guide(
    problem_analysis: Dict[str, Any],
    context_data: Dict[str, Any],
    solution_design: Dict[str, Any],
    technology_stack: Dict[str, Any]
) -> str:
    """
    Create fallback implementation guide when LLM is unavailable.
    
    Args:
        problem_analysis: Problem analysis
        context_data: Context data
        solution_design: Solution design
        technology_stack: Technology stack
        
    Returns:
        Basic implementation guide in markdown
    """
    logger.info("Using fallback guide generation")
    
    title = problem_analysis.get("title", "Solution Implementation")
    solution_type = solution_design.get("solution_type", "SIMPLE_AUTOMATION")
    complexity = problem_analysis.get("complexity", "MEDIUM")
    
    # Get primary technologies
    primary_tech = technology_stack.get("primary_technologies", ["Python", "Standard Libraries"])
    if isinstance(primary_tech, str):
        primary_tech = [primary_tech]
    
    guide = f"""# Implementation Guide: {title}

## Overview

This guide provides step-by-step instructions for implementing the {solution_type} solution.

**Solution Type**: {solution_type}
**Complexity Level**: {complexity}
**Estimated Duration**: {get_complexity_duration(complexity)}

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Virtual environment capability
- Text editor or IDE
- Command line access

### Technical Skills
- Basic Python programming
- Understanding of file operations
- Command line usage

## Technology Stack

### Primary Technologies
{chr(10).join(f"- **{tech}**: {get_tech_description(tech)}" for tech in primary_tech)}

### Supporting Libraries
{chr(10).join(f"- {lib}" for lib in technology_stack.get("supporting_libraries", ["os", "json", "logging"]))}

## Implementation Steps

### Phase 1: Environment Setup (30 minutes)

#### Step 1.1: Create Project Directory
```bash
mkdir {title.lower().replace(' ', '_')}_solution
cd {title.lower().replace(' ', '_')}_solution
```

#### Step 1.2: Set Up Virtual Environment
```bash
python -m venv venv
# On Windows
venv\\Scripts\\activate
# On macOS/Linux
source venv/bin/activate
```

#### Step 1.3: Install Dependencies
```bash
pip install {' '.join(technology_stack.get("supporting_libraries", ["requests"]))}
```

### Phase 2: Core Implementation (2-4 hours)

#### Step 2.1: Create Main Application Structure
```python
# main.py
import os
import json
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('application.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class SolutionImplementation:
    def __init__(self):
        self.config = self.load_config()
        
    def load_config(self):
        # Load configuration settings
        return {{
            "input_path": "input/",
            "output_path": "output/",
            "log_level": "INFO"
        }}
    
    def run(self):
        try:
            logger.info("Starting solution execution")
            # Main solution logic here
            self.process_data()
            logger.info("Solution execution completed successfully")
        except Exception as e:
            logger.error(f"Solution execution failed: {{e}}")
            raise
    
    def process_data(self):
        # Implement main processing logic
        pass

if __name__ == "__main__":
    solution = SolutionImplementation()
    solution.run()
```

#### Step 2.2: Implement Core Functionality
Based on your specific requirements, implement the core functionality in the `process_data` method.

#### Step 2.3: Add Error Handling and Validation
```python
def validate_input(self, data):
    if not data:
        raise ValueError("Input data cannot be empty")
    # Add specific validation logic
    return True

def handle_errors(self, error):
    logger.error(f"Error occurred: {{error}}")
    # Implement recovery logic
    return False
```

### Phase 3: Testing and Validation (1-2 hours)

#### Step 3.1: Create Test Cases
```python
# test_solution.py
import unittest
from main import SolutionImplementation

class TestSolution(unittest.TestCase):
    def setUp(self):
        self.solution = SolutionImplementation()
    
    def test_basic_functionality(self):
        # Test basic functionality
        result = self.solution.process_data()
        self.assertIsNotNone(result)
    
    def test_error_handling(self):
        # Test error handling
        with self.assertRaises(ValueError):
            self.solution.validate_input(None)

if __name__ == '__main__':
    unittest.main()
```

#### Step 3.2: Run Tests
```bash
python -m pytest test_solution.py -v
```

#### Step 3.3: Performance Testing
- Test with sample data
- Monitor memory usage
- Measure execution time
- Validate output quality

### Phase 4: Documentation and Deployment (1 hour)

#### Step 4.1: Create User Documentation
```markdown
# User Guide

## How to Use
1. Place input files in the `input/` directory
2. Run the solution: `python main.py`
3. Check results in the `output/` directory
4. Review logs in `application.log`

## Configuration
Edit the configuration in `main.py` to customize:
- Input/output paths
- Processing parameters
- Logging levels
```

#### Step 4.2: Deployment Checklist
- [ ] All dependencies installed
- [ ] Tests passing
- [ ] Configuration validated
- [ ] Input/output directories created
- [ ] Permissions set correctly
- [ ] Documentation complete

## Monitoring and Maintenance

### Performance Monitoring
- Check log files regularly
- Monitor processing times
- Watch for error patterns
- Track success rates

### Maintenance Tasks
- Update dependencies monthly
- Review and rotate logs
- Backup configuration files
- Test with new data samples

## Troubleshooting

### Common Issues
1. **Import Errors**: Ensure virtual environment is activated
2. **Permission Denied**: Check file/directory permissions
3. **Memory Issues**: Process data in smaller batches
4. **Slow Performance**: Optimize algorithms or increase resources

### Error Codes
- `ERR001`: Configuration file not found
- `ERR002`: Invalid input data format
- `ERR003`: Output directory not accessible
- `ERR004`: Processing timeout

## Success Metrics

### Key Performance Indicators
- **Processing Time**: Target < {get_performance_target(complexity)}
- **Success Rate**: Target > 95%
- **Error Rate**: Target < 5%
- **User Satisfaction**: Target > 4/5

### Measurement Methods
- Automated logging and metrics collection
- User feedback surveys
- Performance benchmarking
- Error rate monitoring

## Next Steps

### Immediate Actions
1. Set up the development environment
2. Implement core functionality
3. Test with sample data
4. Deploy in test environment

### Future Enhancements
- Add web interface
- Implement batch processing
- Add advanced error recovery
- Integrate with existing systems

## Support and Resources

### Technical Support
- Check application logs first
- Review this documentation
- Search for similar issues online
- Contact development team if needed

### Additional Resources
- Python documentation: https://docs.python.org/
- Project repository: [Your repository URL]
- Issue tracker: [Your issue tracker URL]

---
*Generated by AI Problem Solving Copilot*
*Document Version: 1.0*
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
    
    return guide


async def generate_technical_specifications(
    solution_design: Dict[str, Any],
    technology_stack: Dict[str, Any],
    problem_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate technical specifications document.
    
    Args:
        solution_design: Solution design information
        technology_stack: Technology recommendations
        problem_analysis: Problem analysis results
        
    Returns:
        Technical specifications dictionary
    """
    try:
        # Get the LLM agent service
        agent_service = await get_agent_service()
        
        tech_spec_prompt = f"""
        Create technical specifications for the solution implementation.
        
        Solution Design: {json.dumps(solution_design, indent=2)}
        Technology Stack: {json.dumps(technology_stack, indent=2)}
        Problem Analysis: {json.dumps(problem_analysis, indent=2)}
        
        Generate technical specifications including:
        1. System architecture details
        2. API specifications
        3. Data models and schemas
        4. Security considerations
        5. Performance requirements
        6. Integration specifications
        
        Return as a structured JSON document.
        """
        
        response = await agent_service.llm_manager.simple_completion(
            tech_spec_prompt,
            temperature=0.3
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"technical_specifications": response}
            
    except Exception as e:
        logger.warning(f"Technical specifications generation failed: {str(e)}")
        return create_fallback_tech_specs(solution_design, technology_stack)


def create_fallback_tech_specs(
    solution_design: Dict[str, Any],
    technology_stack: Dict[str, Any]
) -> Dict[str, Any]:
    """Create fallback technical specifications."""
    return {
        "architecture": {
            "type": solution_design.get("solution_type", "SIMPLE_AUTOMATION"),
            "components": ["Main Application", "Data Processing", "Error Handling"],
            "data_flow": "Input -> Processing -> Output"
        },
        "performance": {
            "response_time": "< 5 seconds",
            "throughput": "Depends on data volume",
            "memory_usage": "< 1GB"
        },
        "security": {
            "data_protection": "Local file system",
            "access_control": "File permissions",
            "logging": "Application logs"
        },
        "integration": {
            "input_formats": ["CSV", "JSON", "TXT"],
            "output_formats": ["CSV", "JSON", "Report"],
            "apis": "None required"
        }
    }


async def create_deployment_checklist(
    technology_stack: Dict[str, Any],
    complexity: str
) -> Dict[str, Any]:
    """
    Create deployment checklist based on technology stack and complexity.
    
    Args:
        technology_stack: Technology recommendations
        complexity: Solution complexity level
        
    Returns:
        Deployment checklist dictionary
    """
    base_steps = [
        "Verify Python installation (3.8+)",
        "Create virtual environment",
        "Install required dependencies",
        "Set up project directory structure",
        "Configure input/output directories",
        "Test basic functionality",
        "Validate configuration settings",
        "Run unit tests",
        "Check error handling",
        "Verify logging functionality",
        "Test with sample data",
        "Document usage instructions"
    ]
    
    if complexity == "HIGH":
        base_steps.extend([
            "Set up database connections",
            "Configure API endpoints",
            "Test integration points",
            "Set up monitoring",
            "Configure backup procedures",
            "Test disaster recovery",
            "Performance testing",
            "Security validation"
        ])
    elif complexity == "MEDIUM":
        base_steps.extend([
            "Test data validation",
            "Configure batch processing",
            "Set up basic monitoring",
            "Test error recovery"
        ])
    
    return {
        "steps": base_steps,
        "estimated_time": get_deployment_time(complexity),
        "prerequisites": list(technology_stack.get("primary_technologies", ["Python"])),
        "validation_criteria": [
            "All tests pass",
            "No critical errors in logs",
            "Sample data processes correctly",
            "Documentation is complete"
        ]
    }


def get_complexity_duration(complexity: str) -> str:
    """Get estimated duration based on complexity."""
    durations = {
        "LOW": "4-6 hours",
        "MEDIUM": "1-2 days", 
        "HIGH": "3-5 days"
    }
    return durations.get(complexity, "1-2 days")


def get_performance_target(complexity: str) -> str:
    """Get performance target based on complexity."""
    targets = {
        "LOW": "5 seconds",
        "MEDIUM": "30 seconds",
        "HIGH": "2 minutes"
    }
    return targets.get(complexity, "30 seconds")


def get_deployment_time(complexity: str) -> str:
    """Get deployment time based on complexity."""
    times = {
        "LOW": "2-3 hours",
        "MEDIUM": "4-6 hours",
        "HIGH": "1-2 days"
    }
    return times.get(complexity, "4-6 hours")


def get_tech_description(tech: str) -> str:
    """Get description for technology."""
    descriptions = {
        "Python": "Core programming language",
        "FastAPI": "Modern web framework for APIs",
        "SQLite": "Lightweight database",
        "Pandas": "Data manipulation and analysis",
        "Requests": "HTTP library for API calls",
        "Flask": "Lightweight web framework",
        "Django": "Full-featured web framework",
        "Streamlit": "Web app framework for data apps",
        "Jupyter": "Interactive development environment"
    }
    return descriptions.get(tech, "Supporting technology")


def handle_guide_error(state: WorkflowState, error_message: str) -> WorkflowState:
    """
    Handle errors during guide creation.
    
    Args:
        state: Current workflow state
        error_message: Error description
        
    Returns:
        Updated state with error information
    """
    logger.error(f"Guide creation failed: {error_message}")
    
    conversation_history = state.get("conversation_history", [])
    conversation_history.append({
        "sender": "guide_creator",
        "recipient": "system",
        "message_type": "error",
        "content": f"Implementation guide creation error: {error_message}",
        "timestamp": datetime.now().isoformat(),
        "metadata": {"error_type": "guide_creation_error"}
    })
    
    # Create basic fallback guide
    fallback_guide = """# Implementation Guide

An error occurred during guide generation. Please follow these basic steps:

1. Set up Python environment
2. Install required libraries
3. Implement core functionality
4. Test thoroughly
5. Deploy and monitor

Contact support for detailed assistance.
"""
    
    # Create error state with minimal guide
    updated_state = state.copy()
    updated_state.update({
        "current_step": "guide_error",
        "current_status": "error_with_fallback",
        "error_message": error_message,
        "conversation_history": conversation_history,
        "implementation_plan": fallback_guide,
        "retry_count": state.get("retry_count", 0) + 1,
        "requires_user_input": False,
        "workflow_complete": True  # Complete with errors
    })
    
    return updated_state