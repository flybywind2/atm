"""
Unit tests for the requirements generator agent
"""

import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any


class TestRequirementsGeneratorAgent:
    """Test cases for the requirements generator agent."""

    @pytest.mark.asyncio
    async def test_generate_requirements_simple_automation(self, sample_workflow_state, mock_llm_service):
        """Test requirements generation for simple automation."""
        with patch('backend.app.agents.requirements_generator.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                # Software Requirements Specification (SRS)
                
                ## Functional Requirements
                
                ### FR-001: File Processing
                - System shall automatically process Excel files
                - System shall validate data integrity
                - System shall generate summary reports
                
                ### FR-002: User Interface
                - System shall provide simple configuration interface
                - System shall display processing status
                
                ## Non-Functional Requirements
                
                ### NFR-001: Performance
                - System shall process files within 30 seconds
                - System shall handle up to 100 files per batch
                
                ### NFR-002: Usability
                - System shall be operable by non-technical users
                - System shall provide clear error messages
                
                ## Acceptance Criteria
                - Successfully processes Excel files without errors
                - Generates accurate reports
                - Completes processing within time limits
                """,
                "usage": {"tokens": 250}
            }
            
            from backend.app.agents.requirements_generator import generate_requirements
            
            state = {
                **sample_workflow_state,
                "problem_analysis": {
                    "solution_category": "SIMPLE_AUTOMATION",
                    "complexity": "low"
                },
                "context_data": {
                    "file_format": "Excel",
                    "daily_volume": "20 files",
                    "technical_level": "beginner"
                }
            }
            
            result = await generate_requirements(state)
            
            # Verify requirements document structure
            assert result["current_step"] == "generate_requirements"
            assert result["current_status"] == "requirements_generated"
            assert "requirements_doc" in result
            
            requirements = result["requirements_doc"]
            assert "Functional Requirements" in requirements
            assert "Non-Functional Requirements" in requirements
            assert "Acceptance Criteria" in requirements

    @pytest.mark.asyncio
    async def test_generate_requirements_rag_system(self, sample_workflow_state, mock_llm_service):
        """Test requirements generation for RAG system."""
        with patch('backend.app.agents.requirements_generator.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                # Software Requirements Specification (SRS) - Document Q&A System
                
                ## Functional Requirements
                
                ### FR-001: Document Ingestion
                - System shall index PDF, Word, and text documents
                - System shall extract and chunk document content
                - System shall generate embeddings for semantic search
                
                ### FR-002: Query Processing
                - System shall accept natural language queries
                - System shall retrieve relevant document chunks
                - System shall generate contextual answers
                
                ### FR-003: Knowledge Management
                - System shall support document updates
                - System shall track document versions
                - System shall handle document deletion
                
                ## Non-Functional Requirements
                
                ### NFR-001: Performance
                - Query response time shall be under 3 seconds
                - System shall support 100 concurrent users
                - Document indexing shall complete within 1 hour
                
                ### NFR-002: Accuracy
                - Retrieval precision shall exceed 80%
                - Answer relevance shall be above 85%
                
                ## Technical Requirements
                
                ### TR-001: Infrastructure
                - Vector database for embeddings storage
                - LLM integration for answer generation
                - Web interface for user interaction
                
                ## Acceptance Criteria
                - Successfully indexes document collection
                - Provides accurate answers to domain questions
                - Maintains performance under load
                """,
                "usage": {"tokens": 320}
            }
            
            from backend.app.agents.requirements_generator import generate_requirements
            
            state = {
                **sample_workflow_state,
                "problem_analysis": {
                    "solution_category": "RAG",
                    "complexity": "high"
                },
                "context_data": {
                    "document_types": ["PDF", "Word", "Text"],
                    "query_volume": "100 per day",
                    "technical_level": "intermediate"
                }
            }
            
            result = await generate_requirements(state)
            
            requirements = result["requirements_doc"]
            assert "Document Ingestion" in requirements
            assert "Query Processing" in requirements
            assert "embeddings" in requirements.lower()
            assert "vector database" in requirements.lower()

    @pytest.mark.asyncio
    async def test_generate_requirements_ml_classification(self, sample_workflow_state, mock_llm_service):
        """Test requirements generation for ML classification."""
        with patch('backend.app.agents.requirements_generator.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                # Software Requirements Specification (SRS) - Text Classification System
                
                ## Functional Requirements
                
                ### FR-001: Data Processing
                - System shall preprocess text data
                - System shall handle multiple text formats
                - System shall clean and normalize input data
                
                ### FR-002: Model Training
                - System shall train classification models
                - System shall validate model performance
                - System shall support model retraining
                
                ### FR-003: Prediction API
                - System shall provide REST API for predictions
                - System shall return confidence scores
                - System shall handle batch predictions
                
                ## Non-Functional Requirements
                
                ### NFR-001: Accuracy
                - Classification accuracy shall exceed 90%
                - Model shall generalize to new data
                
                ### NFR-002: Performance
                - Prediction latency shall be under 100ms
                - System shall handle 1000 requests per minute
                
                ## Data Requirements
                
                ### DR-001: Training Data
                - Minimum 1000 labeled examples per category
                - Balanced distribution across categories
                - Regular data quality validation
                
                ## Acceptance Criteria
                - Model achieves target accuracy on test set
                - API responds within latency requirements
                - System handles production load
                """,
                "usage": {"tokens": 280}
            }
            
            from backend.app.agents.requirements_generator import generate_requirements
            
            state = {
                **sample_workflow_state,
                "problem_analysis": {
                    "solution_category": "ML_CLASSIFICATION",
                    "complexity": "medium"
                },
                "context_data": {
                    "data_type": "customer_feedback",
                    "categories": ["positive", "negative", "neutral"],
                    "training_data_available": True
                }
            }
            
            result = await generate_requirements(state)
            
            requirements = result["requirements_doc"]
            assert "Classification" in requirements
            assert "Model Training" in requirements
            assert "accuracy" in requirements.lower()
            assert "training data" in requirements.lower()

    def test_validate_requirements_structure(self):
        """Test requirements document structure validation."""
        from backend.app.agents.requirements_generator import validate_requirements_structure
        
        valid_requirements = """
        # Software Requirements Specification
        
        ## Functional Requirements
        ### FR-001: Feature One
        - Requirement 1
        - Requirement 2
        
        ## Non-Functional Requirements
        ### NFR-001: Performance
        - Performance requirement
        
        ## Acceptance Criteria
        - Criterion 1
        - Criterion 2
        """
        
        assert validate_requirements_structure(valid_requirements) == True
        
        invalid_requirements = "Just some text without proper structure"
        assert validate_requirements_structure(invalid_requirements) == False

    def test_extract_functional_requirements(self):
        """Test extraction of functional requirements."""
        from backend.app.agents.requirements_generator import extract_functional_requirements
        
        requirements_doc = """
        ## Functional Requirements
        
        ### FR-001: Data Processing
        - Process CSV files
        - Validate data formats
        
        ### FR-002: Report Generation
        - Generate summary reports
        - Export to PDF
        """
        
        functional_reqs = extract_functional_requirements(requirements_doc)
        
        assert len(functional_reqs) >= 2
        assert "Data Processing" in str(functional_reqs)
        assert "Report Generation" in str(functional_reqs)

    def test_extract_non_functional_requirements(self):
        """Test extraction of non-functional requirements."""
        from backend.app.agents.requirements_generator import extract_non_functional_requirements
        
        requirements_doc = """
        ## Non-Functional Requirements
        
        ### NFR-001: Performance
        - Response time under 2 seconds
        - Handle 100 concurrent users
        
        ### NFR-002: Security
        - Encrypt data at rest
        - Use HTTPS for all communications
        """
        
        nf_reqs = extract_non_functional_requirements(requirements_doc)
        
        assert len(nf_reqs) >= 2
        assert "Performance" in str(nf_reqs)
        assert "Security" in str(nf_reqs)

    @pytest.mark.asyncio
    async def test_generate_requirements_with_rag_context(self, sample_workflow_state, mock_llm_service, mock_rag_service):
        """Test requirements generation with RAG context."""
        with patch('backend.app.agents.requirements_generator.llm_service', mock_llm_service), \
             patch('backend.app.agents.requirements_generator.rag_service', mock_rag_service):
            
            # Mock RAG context
            mock_rag_service.retrieve_context.return_value = {
                "documents": [
                    "Best practices for automation systems include error handling and logging",
                    "Requirements should specify performance metrics and acceptance criteria"
                ],
                "relevance_scores": [0.9, 0.8]
            }
            
            mock_llm_service.generate_response.return_value = {
                "content": """
                # Enhanced SRS with Best Practices
                
                ## Functional Requirements
                ### FR-001: Core Processing
                - Include comprehensive error handling
                - Implement detailed logging
                
                ## Non-Functional Requirements
                ### NFR-001: Performance Metrics
                - Define specific response time targets
                - Include throughput requirements
                """,
                "usage": {"tokens": 200}
            }
            
            from backend.app.agents.requirements_generator import generate_requirements
            
            result = await generate_requirements(sample_workflow_state)
            
            # Should incorporate RAG context
            requirements = result["requirements_doc"]
            assert "error handling" in requirements.lower()
            assert "logging" in requirements.lower()
            assert "performance metrics" in requirements.lower()

    @pytest.mark.asyncio
    async def test_generate_requirements_with_error(self, sample_workflow_state, mock_llm_service):
        """Test requirements generation with LLM error."""
        with patch('backend.app.agents.requirements_generator.llm_service', mock_llm_service):
            mock_llm_service.generate_response.side_effect = Exception("LLM service error")
            
            from backend.app.agents.requirements_generator import generate_requirements
            
            result = await generate_requirements(sample_workflow_state)
            
            # Should handle error gracefully
            assert result["current_status"] == "error"
            assert "error" in result
            assert result["retry_count"] == 1

    def test_calculate_requirements_complexity(self):
        """Test requirements complexity calculation."""
        from backend.app.agents.requirements_generator import calculate_requirements_complexity
        
        simple_context = {
            "technical_level": "beginner",
            "solution_category": "SIMPLE_AUTOMATION"
        }
        
        complex_context = {
            "technical_level": "advanced",
            "solution_category": "RAG",
            "document_types": ["PDF", "Word", "Excel"],
            "query_volume": "high"
        }
        
        simple_complexity = calculate_requirements_complexity(simple_context)
        complex_complexity = calculate_requirements_complexity(complex_context)
        
        assert simple_complexity < complex_complexity
        assert simple_complexity in ["low", "medium"]
        assert complex_complexity in ["medium", "high"]

    @pytest.mark.asyncio
    async def test_generate_user_stories(self, sample_workflow_state):
        """Test user story generation."""
        from backend.app.agents.requirements_generator import generate_user_stories
        
        context_data = {
            "user_role": "business_analyst",
            "daily_tasks": ["report_generation", "data_analysis"],
            "pain_points": ["manual_process", "time_consuming"]
        }
        
        user_stories = generate_user_stories(context_data, "SIMPLE_AUTOMATION")
        
        assert len(user_stories) > 0
        assert all("As a" in story for story in user_stories)
        assert any("business analyst" in story.lower() for story in user_stories)

    @pytest.mark.asyncio
    async def test_requirements_progress_tracking(self, sample_workflow_state, mock_llm_service):
        """Test progress tracking during requirements generation."""
        with patch('backend.app.agents.requirements_generator.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": "# Requirements Document\n## Functional Requirements\n- Feature 1",
                "usage": {"tokens": 100}
            }
            
            from backend.app.agents.requirements_generator import generate_requirements
            
            result = await generate_requirements(sample_workflow_state)
            
            # Should update progress
            assert result["progress_percentage"] > 40  # Higher than previous steps
            assert result["progress_percentage"] <= 60