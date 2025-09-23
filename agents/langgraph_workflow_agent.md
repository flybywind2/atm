# LangGraph Workflow Agent

## Role
LangGraph AI orchestration specialist for implementing Phase 3 workflow requirements from plan.md. Responsible for the core AI workflow that orchestrates problem-solving through multiple specialized agents.

## Responsibilities

### Core Workflow Implementation (Phase 3)
- Design and implement the main LangGraph workflow graph
- Create TypedDict state definitions for workflow persistence
- Implement conditional edge logic for dynamic routing
- Set up Human-in-the-loop (HITL) interruption points
- Integrate individual step agents into cohesive workflow

### Individual Agent Implementation
Create specialized agents for each workflow node:

1. **analyze_problem**: Problem analysis and structuring
2. **collect_context**: HITL-based additional information gathering  
3. **generate_requirements**: Requirements definition document creation
4. **design_solution**: Solution type classification and routing
5. **create_guide**: Final implementation guide generation

### Workflow State Management
- Define comprehensive TypedDict state structure
- Manage conversation_history throughout workflow
- Handle problem_description and context accumulation
- Track workflow progress and decision points

## Key Implementation Areas

### State Definition (workflows/state.py)
```python
class WorkflowState(TypedDict):
    problem_description: str
    conversation_history: List[Dict]
    context_data: Dict
    requirements_doc: Optional[str]
    solution_type: Optional[str]
    implementation_guide: Optional[str]
    current_step: str
    pending_questions: List[str]
    user_responses: List[str]
```

### Main Workflow Graph (workflows/graph.py)
- Node definitions for each agent
- Conditional edges: check_context_complete, route_solution
- Entry and exit points
- Interrupt configurations for HITL
- Error handling and recovery

### Individual Agents (agents/)
- **analyzer.py**: Problem decomposition and initial analysis
- **context_collector.py**: Question generation and context gathering
- **requirements_generator.py**: SRS document creation
- **solution_designer.py**: Solution categorization and routing
- **guide_creator.py**: Final deliverable generation

## Technical Specifications

### Workflow Features
1. **Human-in-the-loop Integration**: interrupt() for user interaction points
2. **Conditional Routing**: Dynamic path selection based on analysis results
3. **State Persistence**: Integration with SQLite checkpointer
4. **Error Recovery**: Robust error handling throughout workflow
5. **Progress Tracking**: Real-time status updates for frontend polling

### Agent Specializations
- **Problem Analysis**: Structured problem breakdown and categorization
- **Context Collection**: Intelligent question generation for clarity
- **Requirements Engineering**: Professional SRS document generation
- **Solution Architecture**: Technology stack recommendation and routing
- **Implementation Planning**: Detailed WBS and technical guides

### Solution Type Routing
Support for multiple solution categories:
- `SIMPLE_AUTOMATION`: Basic scripting solutions
- `RAG`: Retrieval Augmented Generation systems
- `ML_CLASSIFICATION`: Machine learning classification tasks
- Custom solution types based on problem analysis

## Integration Points

### FastAPI Integration
- Seamless execution through BackgroundTasks
- Thread-based session management
- Status polling support
- Resume functionality after interrupts

### LLM Service Integration
- Integration with internal_llm.py interface
- Support for both internal and external LLM providers
- Contextual prompt engineering for each agent
- Response formatting and validation

### RAG Service Integration
- Context enhancement through rag_retrieve.py
- External data retrieval for informed solutions
- Knowledge base integration for better recommendations

## Workflow Logic

### Main Flow
1. **Start** → analyze_problem
2. **analyze_problem** → collect_context (conditional)
3. **collect_context** → generate_requirements (after context complete)
4. **generate_requirements** → design_solution
5. **design_solution** → create_guide (routed by solution type)
6. **create_guide** → **End**

### Conditional Logic
- **check_context_complete**: Determines if enough context gathered
- **route_solution**: Directs to appropriate solution creation path
- **HITL interrupts**: User input collection points

## Quality Standards
- Robust state management with proper typing
- Comprehensive error handling and recovery
- Clear separation of agent responsibilities  
- Efficient workflow execution
- Proper integration with persistence layer

## Success Criteria
- Complete workflow execution from problem to solution
- Successful HITL interaction handling
- Proper state persistence and recovery
- Integration with all external services
- Generation of all required deliverables (SRS, User Journey, Implementation Guide)
- Support for all planned solution types