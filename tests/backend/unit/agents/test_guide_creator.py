"""
Unit tests for the guide creator agent
"""

import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any


class TestGuideCreatorAgent:
    """Test cases for the guide creator agent."""

    @pytest.mark.asyncio
    async def test_create_implementation_guide_automation(self, sample_workflow_state, mock_llm_service):
        """Test implementation guide creation for automation project."""
        with patch('backend.app.agents.guide_creator.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                # Implementation Guide: Excel Report Automation
                
                ## Project Overview
                This guide will help you build an automated Excel report generation system.
                
                ## Prerequisites
                - Python 3.8+
                - Basic understanding of Excel files
                - Command line familiarity
                
                ## Technology Stack
                - **Language**: Python
                - **Framework**: FastAPI
                - **Libraries**: pandas, openpyxl, fastapi
                
                ## Step-by-Step Implementation
                
                ### Step 1: Environment Setup
                ```bash
                # Create virtual environment
                python -m venv excel_automation
                source excel_automation/bin/activate  # Linux/Mac
                excel_automation\\Scripts\\activate  # Windows
                
                # Install dependencies
                pip install pandas openpyxl fastapi uvicorn
                ```
                
                ### Step 2: Data Processing Module
                ```python
                # data_processor.py
                import pandas as pd
                
                class ExcelProcessor:
                    def __init__(self, file_path):
                        self.file_path = file_path
                        
                    def process_data(self):
                        df = pd.read_excel(self.file_path)
                        # Add your processing logic here
                        return df.describe()
                ```
                
                ### Step 3: API Endpoints
                ```python
                # main.py
                from fastapi import FastAPI, UploadFile
                from data_processor import ExcelProcessor
                
                app = FastAPI()
                
                @app.post("/process-excel")
                async def process_excel(file: UploadFile):
                    processor = ExcelProcessor(file.filename)
                    result = processor.process_data()
                    return {"summary": result.to_dict()}
                ```
                
                ## Testing Strategy
                - Unit tests for data processing functions
                - Integration tests for API endpoints
                - Manual testing with sample Excel files
                
                ## Deployment Options
                1. **Local Development**: Run with `uvicorn main:app --reload`
                2. **Docker**: Containerize for consistent deployment
                3. **Cloud**: Deploy to AWS/Azure/GCP
                
                ## Next Steps
                1. Implement error handling
                2. Add data validation
                3. Create user interface
                4. Set up monitoring
                """,
                "usage": {"tokens": 400}
            }
            
            from backend.app.agents.guide_creator import create_guide
            
            state = {
                **sample_workflow_state,
                "solution_type": "SIMPLE_AUTOMATION",
                "technology_stack": {
                    "language": "python",
                    "framework": "fastapi",
                    "libraries": ["pandas", "openpyxl"]
                },
                "requirements_doc": "## Requirements\n- Process Excel files\n- Generate reports",
                "implementation_plan": "## Plan\n1. Data processing\n2. API creation"
            }
            
            result = await create_guide(state)
            
            # Verify guide content
            assert result["current_step"] == "create_guide"
            assert result["current_status"] == "complete"
            assert "implementation_guide" in result
            
            guide = result["implementation_guide"]
            assert "Step-by-Step Implementation" in guide
            assert "Prerequisites" in guide
            assert "Technology Stack" in guide
            assert "python" in guide.lower()
            assert "fastapi" in guide.lower()

    @pytest.mark.asyncio
    async def test_create_rag_implementation_guide(self, sample_workflow_state, mock_llm_service):
        """Test implementation guide creation for RAG system."""
        with patch('backend.app.agents.guide_creator.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                # Implementation Guide: Document Q&A System (RAG)
                
                ## Project Overview
                Build a Retrieval-Augmented Generation system for document question answering.
                
                ## Technology Stack
                - **Framework**: LangChain
                - **Vector Database**: ChromaDB
                - **LLM Provider**: OpenAI
                - **Web Interface**: Streamlit
                
                ## Step-by-Step Implementation
                
                ### Step 1: Environment Setup
                ```bash
                pip install langchain chromadb openai streamlit pypdf2
                ```
                
                ### Step 2: Document Ingestion
                ```python
                # document_loader.py
                from langchain.document_loaders import PyPDFLoader
                from langchain.text_splitter import RecursiveCharacterTextSplitter
                from langchain.embeddings import OpenAIEmbeddings
                from langchain.vectorstores import Chroma
                
                def load_documents(pdf_path):
                    loader = PyPDFLoader(pdf_path)
                    documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    splits = text_splitter.split_documents(documents)
                    
                    embeddings = OpenAIEmbeddings()
                    vectorstore = Chroma.from_documents(splits, embeddings)
                    
                    return vectorstore
                ```
                
                ### Step 3: RAG Pipeline
                ```python
                # rag_system.py
                from langchain.chains import RetrievalQA
                from langchain.llms import OpenAI
                
                def create_rag_chain(vectorstore):
                    llm = OpenAI(temperature=0)
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        chain_type="stuff",
                        retriever=vectorstore.as_retriever()
                    )
                    
                    return qa_chain
                ```
                
                ## Performance Optimization
                - Implement caching for embeddings
                - Use batch processing for document ingestion
                - Optimize chunk size and overlap
                
                ## Deployment Architecture
                1. Document storage layer
                2. Vector database (ChromaDB)
                3. RAG processing service
                4. Web interface (Streamlit)
                """,
                "usage": {"tokens": 450}
            }
            
            from backend.app.agents.guide_creator import create_guide
            
            state = {
                **sample_workflow_state,
                "solution_type": "RAG",
                "technology_stack": {
                    "language": "python",
                    "libraries": ["langchain", "chromadb", "openai"],
                    "vector_db": "chromadb"
                }
            }
            
            result = await create_guide(state)
            
            guide = result["implementation_guide"]
            assert "Document Ingestion" in guide
            assert "RAG Pipeline" in guide
            assert "langchain" in guide.lower()
            assert "chromadb" in guide.lower()

    @pytest.mark.asyncio
    async def test_create_user_journey_map(self, sample_workflow_state, mock_llm_service):
        """Test user journey map creation."""
        with patch('backend.app.agents.guide_creator.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                # User Journey Map: Excel Report Automation
                
                ## Current State (As-Is)
                
                ### User: Business Analyst
                
                #### Daily Workflow
                1. **8:00 AM** - Receive data files via email
                2. **8:30 AM** - Download and organize Excel files
                3. **9:00 AM** - Manual data validation and cleaning
                4. **10:30 AM** - Create pivot tables and charts
                5. **12:00 PM** - Format reports for presentation
                6. **2:00 PM** - Email reports to stakeholders
                
                #### Pain Points
                - ❌ **Time Consuming**: 4+ hours daily on manual tasks
                - ❌ **Error Prone**: Manual data entry leads to mistakes
                - ❌ **Repetitive**: Same process every day
                - ❌ **Late Delivery**: Reports often delayed
                - ❌ **No Scalability**: Cannot handle increased volume
                
                ## Future State (To-Be)
                
                #### Automated Workflow
                1. **8:00 AM** - System automatically fetches data files
                2. **8:05 AM** - Automated data validation and cleaning
                3. **8:15 AM** - Reports generated automatically
                4. **8:20 AM** - Stakeholders receive reports via email
                
                #### Benefits
                - ✅ **Time Savings**: 95% reduction in manual work
                - ✅ **Accuracy**: Eliminates human errors
                - ✅ **Consistency**: Same process every time
                - ✅ **Scalability**: Can handle 10x more data
                - ✅ **Real-time**: Near-instant report generation
                
                ## Impact Analysis
                
                ### Quantitative Benefits
                - **Time Saved**: 3.5 hours per day
                - **Cost Reduction**: $15,000 annually
                - **Error Reduction**: 99% fewer mistakes
                - **Productivity Increase**: 400% improvement
                
                ### Qualitative Benefits
                - Higher job satisfaction
                - Focus on strategic analysis
                - Improved stakeholder relationships
                - Better work-life balance
                """,
                "usage": {"tokens": 350}
            }
            
            from backend.app.agents.guide_creator import create_user_journey_map
            
            context_data = {
                "user_role": "business_analyst",
                "current_process": "manual_excel_processing",
                "pain_points": ["time_consuming", "error_prone"]
            }
            
            result = await create_user_journey_map(context_data, "SIMPLE_AUTOMATION")
            
            assert "Current State" in result
            assert "Future State" in result
            assert "Pain Points" in result
            assert "Benefits" in result
            assert "Time Savings" in result

    def test_generate_code_examples(self):
        """Test code example generation."""
        from backend.app.agents.guide_creator import generate_code_examples
        
        tech_stack = {
            "language": "python",
            "framework": "fastapi",
            "libraries": ["pandas", "openpyxl"]
        }
        
        solution_type = "SIMPLE_AUTOMATION"
        
        examples = generate_code_examples(tech_stack, solution_type)
        
        assert len(examples) > 0
        assert any("import pandas" in example for example in examples)
        assert any("FastAPI" in example for example in examples)

    def test_create_testing_strategy(self):
        """Test testing strategy creation."""
        from backend.app.agents.guide_creator import create_testing_strategy
        
        solution_data = {
            "solution_type": "RAG",
            "technology_stack": {
                "libraries": ["langchain", "chromadb"]
            },
            "complexity": "high"
        }
        
        strategy = create_testing_strategy(solution_data)
        
        assert "Unit Tests" in strategy
        assert "Integration Tests" in strategy
        assert "Performance Tests" in strategy

    def test_generate_deployment_guide(self):
        """Test deployment guide generation."""
        from backend.app.agents.guide_creator import generate_deployment_guide
        
        tech_stack = {
            "framework": "fastapi",
            "deployment": "docker",
            "language": "python"
        }
        
        deployment_guide = generate_deployment_guide(tech_stack)
        
        assert "Docker" in deployment_guide
        assert "Environment Variables" in deployment_guide
        assert "uvicorn" in deployment_guide.lower()

    @pytest.mark.asyncio
    async def test_create_guide_with_error_handling(self, sample_workflow_state, mock_llm_service):
        """Test guide creation with LLM error."""
        with patch('backend.app.agents.guide_creator.llm_service', mock_llm_service):
            mock_llm_service.generate_response.side_effect = Exception("LLM service error")
            
            from backend.app.agents.guide_creator import create_guide
            
            result = await create_guide(sample_workflow_state)
            
            # Should handle error gracefully
            assert result["current_status"] == "error"
            assert "error" in result
            assert result["retry_count"] == 1

    def test_validate_guide_completeness(self):
        """Test guide completeness validation."""
        from backend.app.agents.guide_creator import validate_guide_completeness
        
        complete_guide = """
        # Implementation Guide
        
        ## Prerequisites
        - Python installed
        
        ## Step-by-Step Implementation
        ### Step 1: Setup
        Instructions here
        
        ## Testing Strategy
        Test instructions
        
        ## Deployment
        Deployment instructions
        """
        
        incomplete_guide = "Just some basic text"
        
        assert validate_guide_completeness(complete_guide) == True
        assert validate_guide_completeness(incomplete_guide) == False

    def test_calculate_project_timeline(self):
        """Test project timeline calculation."""
        from backend.app.agents.guide_creator import calculate_project_timeline
        
        simple_project = {
            "solution_type": "SIMPLE_AUTOMATION",
            "complexity": "low",
            "technical_level": "beginner"
        }
        
        complex_project = {
            "solution_type": "RAG",
            "complexity": "high",
            "technical_level": "intermediate"
        }
        
        simple_timeline = calculate_project_timeline(simple_project)
        complex_timeline = calculate_project_timeline(complex_project)
        
        assert simple_timeline["duration_days"] < complex_timeline["duration_days"]
        assert "phases" in simple_timeline
        assert "phases" in complex_timeline

    @pytest.mark.asyncio
    async def test_guide_creation_progress_tracking(self, sample_workflow_state, mock_llm_service):
        """Test progress tracking during guide creation."""
        with patch('backend.app.agents.guide_creator.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": "# Implementation Guide\n## Step 1\nInstructions here",
                "usage": {"tokens": 100}
            }
            
            from backend.app.agents.guide_creator import create_guide
            
            result = await create_guide(sample_workflow_state)
            
            # Should complete the workflow
            assert result["progress_percentage"] == 100
            assert result["current_status"] == "complete"