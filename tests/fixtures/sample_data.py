"""
Sample data fixtures for testing
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List


# Sample problem descriptions for testing
SAMPLE_PROBLEMS = {
    "simple_automation": {
        "description": "I need to automate daily Excel report generation. Currently I manually open files, validate data, and create summaries which takes 3 hours daily.",
        "user_context": {
            "technical_level": "beginner",
            "environment": "Windows 10",
            "tools": ["Excel", "Outlook"],
            "daily_volume": "20 files",
            "time_spent": "3 hours"
        },
        "expected_category": "SIMPLE_AUTOMATION"
    },
    
    "rag_system": {
        "description": "Build a document question-answering system for our company knowledge base with 10,000+ technical documents",
        "user_context": {
            "technical_level": "advanced",
            "environment": "Linux",
            "document_types": ["PDF", "Word", "Text"],
            "query_volume": "500 per day",
            "performance_requirements": "sub-3-second response"
        },
        "expected_category": "RAG"
    },
    
    "ml_classification": {
        "description": "Classify customer feedback emails into positive, negative, and neutral categories automatically",
        "user_context": {
            "technical_level": "intermediate",
            "data_volume": "1000 emails per day",
            "categories": ["positive", "negative", "neutral"],
            "accuracy_requirement": "90%+",
            "training_data_available": True
        },
        "expected_category": "ML_CLASSIFICATION"
    },
    
    "complex_automation": {
        "description": "Automate multi-step data pipeline: extract from APIs, transform data, validate business rules, generate reports, and send notifications",
        "user_context": {
            "technical_level": "advanced",
            "environment": "Cloud (AWS)",
            "data_sources": ["REST APIs", "databases", "file uploads"],
            "business_rules": "complex validation logic",
            "schedule": "hourly processing"
        },
        "expected_category": "COMPLEX_AUTOMATION"
    }
}

# Sample workflow states at different stages
SAMPLE_WORKFLOW_STATES = {
    "initial": {
        "problem_description": "Automate daily report generation",
        "conversation_history": [],
        "context_data": {},
        "current_step": "start",
        "current_status": "initialized",
        "context_complete": False,
        "requires_user_input": False,
        "retry_count": 0,
        "progress_percentage": 0,
        "problem_analysis": None,
        "requirements_doc": None,
        "solution_type": None,
        "technology_stack": None,
        "implementation_plan": None,
        "implementation_guide": None,
        "user_journey": None
    },
    
    "after_analysis": {
        "problem_description": "Automate daily report generation",
        "conversation_history": [
            {"role": "user", "content": "I need help automating daily reports"},
            {"role": "assistant", "content": "I'll analyze your automation needs"}
        ],
        "context_data": {"technical_level": "beginner"},
        "current_step": "analyze_problem",
        "current_status": "analyzing",
        "context_complete": False,
        "requires_user_input": False,
        "retry_count": 0,
        "progress_percentage": 20,
        "problem_analysis": {
            "type": "automation",
            "complexity": "low",
            "solution_category": "SIMPLE_AUTOMATION",
            "key_components": ["data_processing", "file_handling"],
            "estimated_effort": "2-4 hours"
        },
        "requirements_doc": None,
        "solution_type": None
    },
    
    "awaiting_context": {
        "problem_description": "Complex automation with multiple requirements",
        "conversation_history": [
            {"role": "user", "content": "I need complex automation help"},
            {"role": "assistant", "content": "I need more details about your requirements"}
        ],
        "context_data": {"technical_level": "intermediate"},
        "current_step": "collect_context",
        "current_status": "awaiting_input",
        "context_complete": False,
        "requires_user_input": True,
        "questions": [
            "What file formats do you work with?",
            "How many files do you process daily?",
            "What validation rules are required?"
        ],
        "retry_count": 0,
        "progress_percentage": 40,
        "problem_analysis": {
            "type": "automation",
            "complexity": "medium",
            "solution_category": "SIMPLE_AUTOMATION"
        }
    },
    
    "complete": {
        "problem_description": "Automate invoice processing workflow",
        "conversation_history": [
            {"role": "user", "content": "I need to automate invoice processing"},
            {"role": "assistant", "content": "I'll help you build an automation solution"}
        ],
        "context_data": {
            "technical_level": "intermediate",
            "file_format": "PDF",
            "daily_volume": "50 invoices",
            "validation_rules": "business logic validation",
            "output_format": "Excel reports"
        },
        "current_step": "create_guide",
        "current_status": "complete",
        "context_complete": True,
        "requires_user_input": False,
        "retry_count": 0,
        "progress_percentage": 100,
        "problem_analysis": {
            "type": "automation",
            "complexity": "medium",
            "solution_category": "SIMPLE_AUTOMATION"
        },
        "requirements_doc": """# Software Requirements Specification

## Functional Requirements
- FR-001: Process PDF invoices automatically
- FR-002: Validate business logic rules
- FR-003: Generate Excel reports
- FR-004: Handle error cases gracefully

## Non-Functional Requirements
- NFR-001: Process 50 invoices within 30 minutes
- NFR-002: 99% accuracy in data extraction
- NFR-003: Secure handling of sensitive data

## Acceptance Criteria
- Successfully processes invoices without manual intervention
- Generates accurate reports with validation results
- Handles various PDF formats and layouts""",
        "solution_type": "SIMPLE_AUTOMATION",
        "technology_stack": {
            "language": "python",
            "framework": "fastapi",
            "libraries": ["pandas", "PyPDF2", "openpyxl"],
            "deployment": "docker"
        },
        "implementation_plan": """## Implementation Plan
1. PDF data extraction module
2. Business rules validation engine
3. Excel report generation
4. Error handling and logging
5. Testing and deployment""",
        "implementation_guide": """# Invoice Processing Automation Guide

## Prerequisites
- Python 3.8+
- PDF processing libraries
- Excel manipulation tools

## Step-by-Step Implementation

### Step 1: Environment Setup
```bash
pip install pandas PyPDF2 openpyxl fastapi
```

### Step 2: PDF Processing
```python
import PyPDF2
import pandas as pd

def extract_invoice_data(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return parse_invoice_text(text)
```

### Step 3: Validation Engine
```python
def validate_invoice_data(data):
    errors = []
    if not data.get('amount') or data['amount'] <= 0:
        errors.append("Invalid amount")
    if not data.get('date'):
        errors.append("Missing date")
    return errors
```

## Testing Strategy
- Unit tests for each component
- Integration tests with sample PDFs
- Performance tests with batch processing""",
        "user_journey": """# User Journey: Invoice Processing Automation

## Current State (As-Is)
### Daily Workflow
1. **8:00 AM** - Receive invoice PDFs via email
2. **8:30 AM** - Download and organize files
3. **9:00 AM** - Manual data entry into Excel
4. **11:00 AM** - Validate business rules manually
5. **12:00 PM** - Generate reports
6. **1:00 PM** - Send reports to stakeholders

### Pain Points
- Manual data entry errors
- Time-consuming validation
- Inconsistent formatting
- Delayed reporting

## Future State (To-Be)
### Automated Workflow
1. **8:00 AM** - System monitors email for invoices
2. **8:05 AM** - Automatic PDF processing
3. **8:10 AM** - Data validation and error flagging
4. **8:15 AM** - Report generation
5. **8:20 AM** - Automatic distribution

### Benefits
- 95% time reduction
- Eliminated manual errors
- Consistent processing
- Real-time reporting"""
    }
}

# Sample LLM responses for different scenarios
SAMPLE_LLM_RESPONSES = {
    "analysis_simple": {
        "content": """{
            "problem_type": "automation",
            "complexity": "low",
            "solution_category": "SIMPLE_AUTOMATION",
            "key_components": ["data_processing", "file_handling"],
            "estimated_effort": "2-4 hours",
            "confidence": 0.9
        }""",
        "usage": {"tokens": 120}
    },
    
    "analysis_rag": {
        "content": """{
            "problem_type": "knowledge_management",
            "complexity": "high",
            "solution_category": "RAG",
            "key_components": ["document_embedding", "vector_search", "llm_generation"],
            "estimated_effort": "2-3 weeks",
            "confidence": 0.8
        }""",
        "usage": {"tokens": 150}
    },
    
    "context_questions": {
        "content": """{
            "questions": [
                "What file formats do you work with?",
                "How many files do you process daily?",
                "What is your current manual process?",
                "What tools do you currently use?"
            ],
            "context_required": true,
            "priority": "high"
        }""",
        "usage": {"tokens": 80}
    },
    
    "context_sufficient": {
        "content": """{
            "context_sufficient": true,
            "analysis": "User provided comprehensive context about Excel automation needs",
            "next_step": "requirements_generation"
        }""",
        "usage": {"tokens": 60}
    },
    
    "requirements_document": {
        "content": """# Software Requirements Specification

## Project Overview
Automated Excel report generation system for daily business operations.

## Functional Requirements

### FR-001: Data Processing
- System shall automatically process Excel files from specified directory
- System shall validate data integrity and format
- System shall handle multiple Excel file formats (.xlsx, .xls)

### FR-002: Report Generation
- System shall generate summary reports with key metrics
- System shall create charts and visualizations
- System shall export reports in PDF and Excel formats

### FR-003: Error Handling
- System shall log all processing errors
- System shall generate error reports for failed files
- System shall continue processing other files when errors occur

## Non-Functional Requirements

### NFR-001: Performance
- System shall process files within 30 seconds each
- System shall handle up to 100 files per batch
- System shall maintain 99% uptime

### NFR-002: Usability
- System shall provide simple configuration interface
- System shall display clear progress indicators
- System shall generate user-friendly error messages

## Acceptance Criteria
- Successfully processes Excel files without manual intervention
- Generates accurate reports matching manual calculations
- Completes daily processing within allocated time window""",
        "usage": {"tokens": 300}
    },
    
    "solution_design": {
        "content": """{
            "solution_type": "SIMPLE_AUTOMATION",
            "technology_stack": {
                "language": "python",
                "framework": "fastapi",
                "libraries": ["pandas", "openpyxl", "matplotlib"],
                "deployment": "docker",
                "database": "sqlite"
            },
            "architecture_pattern": "microservice",
            "estimated_complexity": "low",
            "development_phases": [
                "data_processing_module",
                "report_generation_engine",
                "api_endpoints",
                "error_handling",
                "testing_deployment"
            ]
        }""",
        "usage": {"tokens": 200}
    }
}

# Sample RAG responses for different queries
SAMPLE_RAG_RESPONSES = {
    "automation_best_practices": {
        "documents": [
            "Best practices for automation include proper error handling, logging, and monitoring",
            "Automation systems should be designed with scalability and maintainability in mind",
            "Testing is crucial for automation reliability - include unit, integration, and end-to-end tests"
        ],
        "relevance_scores": [0.9, 0.8, 0.85]
    },
    
    "python_libraries": {
        "documents": [
            "pandas is excellent for data manipulation and analysis in Python",
            "openpyxl provides comprehensive Excel file handling capabilities",
            "fastapi offers modern, fast web API development with automatic documentation"
        ],
        "relevance_scores": [0.88, 0.92, 0.86]
    },
    
    "deployment_strategies": {
        "documents": [
            "Docker containerization ensures consistent deployment across environments",
            "CI/CD pipelines automate testing and deployment processes",
            "Cloud platforms provide scalable hosting with managed services"
        ],
        "relevance_scores": [0.91, 0.87, 0.83]
    }
}

# Sample error scenarios for testing
ERROR_SCENARIOS = {
    "llm_timeout": {
        "error_type": "TimeoutError",
        "message": "LLM service request timed out after 30 seconds",
        "retry_after": 5,
        "recoverable": True
    },
    
    "llm_rate_limit": {
        "error_type": "RateLimitError",
        "message": "API rate limit exceeded. Please try again later",
        "retry_after": 60,
        "recoverable": True
    },
    
    "llm_service_down": {
        "error_type": "ServiceUnavailableError",
        "message": "LLM service is temporarily unavailable",
        "retry_after": 300,
        "recoverable": True
    },
    
    "database_connection": {
        "error_type": "DatabaseError",
        "message": "Unable to connect to database",
        "retry_after": 10,
        "recoverable": True
    },
    
    "invalid_input": {
        "error_type": "ValidationError",
        "message": "Invalid problem description provided",
        "retry_after": 0,
        "recoverable": False
    },
    
    "corrupted_state": {
        "error_type": "StateCorruptionError",
        "message": "Workflow state appears to be corrupted",
        "retry_after": 0,
        "recoverable": False
    }
}

# Performance test configurations
PERFORMANCE_CONFIGS = {
    "light_load": {
        "concurrent_users": 5,
        "test_duration": 30,
        "request_timeout": 5,
        "success_rate_threshold": 0.98,
        "max_response_time": 2.0
    },
    
    "medium_load": {
        "concurrent_users": 20,
        "test_duration": 60,
        "request_timeout": 10,
        "success_rate_threshold": 0.95,
        "max_response_time": 5.0
    },
    
    "heavy_load": {
        "concurrent_users": 50,
        "test_duration": 120,
        "request_timeout": 15,
        "success_rate_threshold": 0.90,
        "max_response_time": 10.0
    },
    
    "stress_test": {
        "concurrent_users": 100,
        "test_duration": 300,
        "request_timeout": 20,
        "success_rate_threshold": 0.80,
        "max_response_time": 15.0
    }
}

# Sample API responses for different endpoints
SAMPLE_API_RESPONSES = {
    "health_check": {
        "status": "healthy",
        "timestamp": "2024-01-01T10:00:00Z",
        "version": "1.0.0",
        "database": "connected",
        "services": {
            "llm": "available",
            "rag": "available"
        }
    },
    
    "analysis_started": {
        "status": "started",
        "thread_id": "thread_12345",
        "message": "Analysis started successfully",
        "estimated_duration": "2-5 minutes"
    },
    
    "status_running": {
        "thread_id": "thread_12345",
        "status": "running",
        "current_step": "analyze_problem",
        "progress_percentage": 25,
        "message": "Analyzing your problem requirements...",
        "requires_input": False,
        "created_at": "2024-01-01T10:00:00Z",
        "updated_at": "2024-01-01T10:02:00Z"
    },
    
    "status_awaiting_input": {
        "thread_id": "thread_12345",
        "status": "awaiting_input",
        "current_step": "collect_context",
        "progress_percentage": 40,
        "message": "Additional information needed",
        "requires_input": True,
        "questions": [
            "What file formats do you work with?",
            "How many files do you process daily?"
        ],
        "created_at": "2024-01-01T10:00:00Z",
        "updated_at": "2024-01-01T10:03:00Z"
    },
    
    "status_completed": {
        "thread_id": "thread_12345",
        "status": "completed",
        "current_step": "create_guide",
        "progress_percentage": 100,
        "message": "Analysis completed successfully",
        "requires_input": False,
        "results": {
            "requirements_doc": "# Requirements Document\n...",
            "implementation_guide": "# Implementation Guide\n...",
            "user_journey": "# User Journey\n...",
            "solution_type": "SIMPLE_AUTOMATION"
        },
        "created_at": "2024-01-01T10:00:00Z",
        "updated_at": "2024-01-01T10:15:00Z"
    }
}

# Test data generators
def generate_sample_problem(category: str = "simple_automation") -> Dict[str, Any]:
    """Generate a sample problem for testing."""
    if category in SAMPLE_PROBLEMS:
        return SAMPLE_PROBLEMS[category].copy()
    
    return {
        "description": f"Sample {category} problem for testing",
        "user_context": {"technical_level": "intermediate"},
        "expected_category": category.upper()
    }

def generate_workflow_state(stage: str = "initial") -> Dict[str, Any]:
    """Generate a workflow state for testing."""
    if stage in SAMPLE_WORKFLOW_STATES:
        return SAMPLE_WORKFLOW_STATES[stage].copy()
    
    return SAMPLE_WORKFLOW_STATES["initial"].copy()

def generate_llm_response(response_type: str) -> Dict[str, Any]:
    """Generate an LLM response for testing."""
    if response_type in SAMPLE_LLM_RESPONSES:
        return SAMPLE_LLM_RESPONSES[response_type].copy()
    
    return {
        "content": f"Mock {response_type} response",
        "usage": {"tokens": 100}
    }

def generate_error_scenario(error_type: str) -> Dict[str, Any]:
    """Generate an error scenario for testing."""
    if error_type in ERROR_SCENARIOS:
        return ERROR_SCENARIOS[error_type].copy()
    
    return {
        "error_type": "GenericError",
        "message": f"Mock {error_type} error",
        "retry_after": 5,
        "recoverable": True
    }

def generate_large_workflow_state(size_mb: float = 1.0) -> Dict[str, Any]:
    """Generate a large workflow state for memory/performance testing."""
    base_state = generate_workflow_state("complete")
    
    # Add large data to simulate memory usage
    large_data_size = int(size_mb * 1024 * 1024 / 4)  # Rough estimate for string size
    base_state["large_test_data"] = "x" * large_data_size
    
    return base_state

def generate_conversation_history(length: int = 10) -> List[Dict[str, Any]]:
    """Generate conversation history for testing."""
    history = []
    for i in range(length):
        role = "user" if i % 2 == 0 else "assistant"
        content = f"Message {i+1} from {role}"
        timestamp = (datetime.now() - timedelta(minutes=length-i)).isoformat()
        
        history.append({
            "role": role,
            "content": content,
            "timestamp": timestamp
        })
    
    return history