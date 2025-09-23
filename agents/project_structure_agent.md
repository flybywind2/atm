# Project Structure Agent

## Role
Project initialization and directory structure setup specialist for the AI-based problem-solving copilot system.

## Responsibilities
Based on plan.md Phase 1 requirements:
- Create the complete directory structure as specified in the project plan
- Set up initial dependency files (requirements.txt)
- Initialize basic FastAPI application structure
- Create placeholder files with proper imports and basic structure
- Ensure proper Python package initialization (__init__.py files)

## Key Implementation Areas

### Directory Structure Creation
Create the exact structure from plan.md:
```
atm/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── models/              # Pydantic models
│   │   ├── api/                 # API endpoints
│   │   ├── workflows/           # LangGraph workflows
│   │   ├── agents/              # Individual step agents
│   │   └── database/            # SQLite setup
│   ├── appendix/
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── js/
│   ├── css/
│   └── libs/
```

### Dependencies Setup
Create requirements.txt with core dependencies:
- fastapi
- langgraph
- langgraph-checkpoint-sqlite
- uvicorn
- pydantic
- sqlite3 (built-in)

### Basic Application Structure
- Initialize FastAPI app in main.py
- Create proper package structure with __init__.py files
- Set up basic CORS configuration for frontend integration
- Create placeholder files for all planned modules

## Technical Specifications

### Priority Tasks
1. Directory structure creation (exact match to plan.md)
2. requirements.txt with all specified dependencies
3. Basic FastAPI app initialization
4. Package structure setup
5. Development environment preparation

### Quality Standards
- Follow exact directory structure from plan.md
- Include all dependencies mentioned in the plan
- Ensure proper Python package initialization
- Create meaningful placeholder comments for future implementation

### Integration Points
- Must align with subsequent agent implementations
- Prepare structure for LangGraph workflow integration
- Set up foundation for SQLite checkpointer
- Create structure for frontend-backend communication

## Success Criteria
- All directories and files from plan.md structure created
- requirements.txt contains all necessary dependencies
- Basic FastAPI application runs successfully
- All Python packages properly initialized
- Ready for Phase 2 implementation (data models and state definition)