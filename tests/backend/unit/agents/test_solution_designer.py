"""
Unit tests for the solution designer agent
"""

import pytest
from unittest.mock import AsyncMock, patch
from typing import Dict, Any


class TestSolutionDesignerAgent:
    """Test cases for the solution designer agent."""

    @pytest.mark.asyncio
    async def test_design_simple_automation_solution(self, sample_workflow_state, mock_llm_service):
        """Test solution design for simple automation."""
        with patch('backend.app.agents.solution_designer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "solution_type": "SIMPLE_AUTOMATION",
                    "technology_stack": {
                        "language": "python",
                        "framework": "fastapi",
                        "libraries": ["pandas", "openpyxl", "requests"],
                        "deployment": "docker"
                    },
                    "architecture_pattern": "microservice",
                    "implementation_approach": "api_based",
                    "estimated_complexity": "low",
                    "development_phases": [
                        "data_processing_module",
                        "api_endpoints",
                        "error_handling",
                        "testing_deployment"
                    ]
                }
                """,
                "usage": {"tokens": 200}
            }
            
            from backend.app.agents.solution_designer import design_solution
            
            state = {
                **sample_workflow_state,
                "problem_analysis": {
                    "solution_category": "SIMPLE_AUTOMATION",
                    "complexity": "low"
                },
                "context_data": {
                    "file_format": "Excel",
                    "technical_level": "beginner",
                    "environment": "Windows"
                },
                "requirements_doc": "## Functional Requirements\n- Process Excel files\n- Generate reports"
            }
            
            result = await design_solution(state)
            
            # Verify solution design
            assert result["current_step"] == "design_solution"
            assert result["current_status"] == "solution_designed"
            assert result["solution_type"] == "SIMPLE_AUTOMATION"
            assert "technology_stack" in result
            assert result["technology_stack"]["language"] == "python"
            assert "pandas" in result["technology_stack"]["libraries"]

    @pytest.mark.asyncio
    async def test_design_rag_solution(self, sample_workflow_state, mock_llm_service):
        """Test solution design for RAG system."""
        with patch('backend.app.agents.solution_designer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "solution_type": "RAG",
                    "technology_stack": {
                        "language": "python",
                        "framework": "fastapi",
                        "libraries": ["langchain", "chromadb", "openai", "streamlit"],
                        "vector_db": "chromadb",
                        "llm_provider": "openai",
                        "deployment": "docker_compose"
                    },
                    "architecture_pattern": "rag_pipeline",
                    "implementation_approach": "vector_search",
                    "estimated_complexity": "high",
                    "development_phases": [
                        "document_ingestion",
                        "embedding_generation",
                        "vector_storage",
                        "retrieval_system",
                        "generation_pipeline",
                        "web_interface"
                    ]
                }
                """,
                "usage": {"tokens": 280}
            }
            
            from backend.app.agents.solution_designer import design_solution
            
            state = {
                **sample_workflow_state,
                "problem_analysis": {
                    "solution_category": "RAG",
                    "complexity": "high"
                },
                "context_data": {
                    "document_types": ["PDF", "Word"],
                    "query_volume": "high",
                    "technical_level": "intermediate"
                }
            }
            
            result = await design_solution(state)
            
            assert result["solution_type"] == "RAG"
            assert "vector_db" in result["technology_stack"]
            assert "langchain" in result["technology_stack"]["libraries"]
            assert "document_ingestion" in result["development_phases"]

    @pytest.mark.asyncio
    async def test_design_ml_classification_solution(self, sample_workflow_state, mock_llm_service):
        """Test solution design for ML classification."""
        with patch('backend.app.agents.solution_designer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "solution_type": "ML_CLASSIFICATION",
                    "technology_stack": {
                        "language": "python",
                        "framework": "fastapi",
                        "libraries": ["scikit-learn", "pandas", "numpy", "joblib"],
                        "ml_framework": "scikit_learn",
                        "deployment": "docker"
                    },
                    "architecture_pattern": "ml_pipeline",
                    "implementation_approach": "supervised_learning",
                    "estimated_complexity": "medium",
                    "development_phases": [
                        "data_preprocessing",
                        "feature_engineering",
                        "model_training",
                        "model_evaluation",
                        "prediction_api",
                        "monitoring"
                    ]
                }
                """,
                "usage": {"tokens": 250}
            }
            
            from backend.app.agents.solution_designer import design_solution
            
            state = {
                **sample_workflow_state,
                "problem_analysis": {
                    "solution_category": "ML_CLASSIFICATION",
                    "complexity": "medium"
                },
                "context_data": {
                    "data_type": "text",
                    "categories": ["positive", "negative", "neutral"],
                    "training_data_available": True
                }
            }
            
            result = await design_solution(state)
            
            assert result["solution_type"] == "ML_CLASSIFICATION"
            assert "scikit-learn" in result["technology_stack"]["libraries"]
            assert "model_training" in result["development_phases"]

    def test_select_technology_stack_simple(self):
        """Test technology stack selection for simple automation."""
        from backend.app.agents.solution_designer import select_technology_stack
        
        context = {
            "technical_level": "beginner",
            "environment": "Windows",
            "solution_category": "SIMPLE_AUTOMATION"
        }
        
        stack = select_technology_stack(context)
        
        assert stack["language"] == "python"
        assert "pandas" in stack["libraries"]
        assert stack["complexity"] == "low"

    def test_select_technology_stack_advanced(self):
        """Test technology stack selection for advanced solutions."""
        from backend.app.agents.solution_designer import select_technology_stack
        
        context = {
            "technical_level": "advanced",
            "environment": "Linux",
            "solution_category": "RAG",
            "performance_requirements": "high"
        }
        
        stack = select_technology_stack(context)
        
        assert "langchain" in stack["libraries"] or "llamaindex" in stack["libraries"]
        assert stack["complexity"] in ["medium", "high"]

    def test_estimate_development_effort(self):
        """Test development effort estimation."""
        from backend.app.agents.solution_designer import estimate_development_effort
        
        simple_solution = {
            "solution_type": "SIMPLE_AUTOMATION",
            "complexity": "low",
            "phases": ["data_processing", "api_endpoints"]
        }
        
        complex_solution = {
            "solution_type": "RAG",
            "complexity": "high",
            "phases": ["ingestion", "embedding", "retrieval", "generation", "interface"]
        }
        
        simple_effort = estimate_development_effort(simple_solution)
        complex_effort = estimate_development_effort(complex_solution)
        
        assert simple_effort["hours"] < complex_effort["hours"]
        assert simple_effort["complexity"] == "low"
        assert complex_effort["complexity"] == "high"

    def test_validate_solution_feasibility(self):
        """Test solution feasibility validation."""
        from backend.app.agents.solution_designer import validate_solution_feasibility
        
        feasible_solution = {
            "technology_stack": {
                "language": "python",
                "libraries": ["pandas", "fastapi"]
            },
            "context": {
                "technical_level": "intermediate",
                "timeline": "2 weeks"
            }
        }
        
        infeasible_solution = {
            "technology_stack": {
                "language": "assembly",
                "libraries": ["complex_ai_framework"]
            },
            "context": {
                "technical_level": "beginner",
                "timeline": "1 day"
            }
        }
        
        assert validate_solution_feasibility(feasible_solution) == True
        assert validate_solution_feasibility(infeasible_solution) == False

    @pytest.mark.asyncio
    async def test_design_with_custom_requirements(self, sample_workflow_state, mock_llm_service):
        """Test solution design with custom requirements."""
        with patch('backend.app.agents.solution_designer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """
                {
                    "solution_type": "CUSTOM_AUTOMATION",
                    "technology_stack": {
                        "language": "python",
                        "framework": "fastapi",
                        "libraries": ["custom_lib", "pandas"],
                        "special_requirements": ["real_time_processing", "high_availability"]
                    },
                    "custom_components": ["monitoring_dashboard", "alert_system"]
                }
                """,
                "usage": {"tokens": 180}
            }
            
            from backend.app.agents.solution_designer import design_solution
            
            state = {
                **sample_workflow_state,
                "problem_analysis": {
                    "solution_category": "CUSTOM",
                    "complexity": "medium"
                },
                "requirements_doc": "## Special Requirements\n- Real-time processing\n- High availability"
            }
            
            result = await design_solution(state)
            
            assert "custom_components" in result
            assert "real_time_processing" in str(result["technology_stack"])

    @pytest.mark.asyncio
    async def test_design_with_error_handling(self, sample_workflow_state, mock_llm_service):
        """Test solution design with LLM error."""
        with patch('backend.app.agents.solution_designer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.side_effect = Exception("LLM service error")
            
            from backend.app.agents.solution_designer import design_solution
            
            result = await design_solution(sample_workflow_state)
            
            # Should handle error gracefully
            assert result["current_status"] == "error"
            assert "error" in result
            assert result["retry_count"] == 1

    def test_generate_architecture_diagram_data(self):
        """Test architecture diagram data generation."""
        from backend.app.agents.solution_designer import generate_architecture_diagram_data
        
        solution = {
            "solution_type": "RAG",
            "technology_stack": {
                "framework": "fastapi",
                "vector_db": "chromadb",
                "llm_provider": "openai"
            },
            "development_phases": ["ingestion", "retrieval", "generation"]
        }
        
        diagram_data = generate_architecture_diagram_data(solution)
        
        assert "components" in diagram_data
        assert "connections" in diagram_data
        assert len(diagram_data["components"]) > 0

    def test_calculate_resource_requirements(self):
        """Test resource requirements calculation."""
        from backend.app.agents.solution_designer import calculate_resource_requirements
        
        simple_solution = {"solution_type": "SIMPLE_AUTOMATION", "complexity": "low"}
        complex_solution = {"solution_type": "RAG", "complexity": "high"}
        
        simple_resources = calculate_resource_requirements(simple_solution)
        complex_resources = calculate_resource_requirements(complex_solution)
        
        assert simple_resources["memory_gb"] < complex_resources["memory_gb"]
        assert simple_resources["cpu_cores"] <= complex_resources["cpu_cores"]

    @pytest.mark.asyncio
    async def test_solution_design_progress_tracking(self, sample_workflow_state, mock_llm_service):
        """Test progress tracking during solution design."""
        with patch('backend.app.agents.solution_designer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": """{"solution_type": "SIMPLE_AUTOMATION", "technology_stack": {"language": "python"}}""",
                "usage": {"tokens": 100}
            }
            
            from backend.app.agents.solution_designer import design_solution
            
            result = await design_solution(sample_workflow_state)
            
            # Should update progress
            assert result["progress_percentage"] > 60  # Higher than previous steps
            assert result["progress_percentage"] <= 80

    def test_solution_type_routing(self):
        """Test solution type routing logic."""
        from backend.app.agents.solution_designer import route_solution_type
        
        automation_analysis = {"solution_category": "SIMPLE_AUTOMATION"}
        rag_analysis = {"solution_category": "RAG"}
        ml_analysis = {"solution_category": "ML_CLASSIFICATION"}
        
        assert route_solution_type(automation_analysis) == "automation_design"
        assert route_solution_type(rag_analysis) == "rag_design"
        assert route_solution_type(ml_analysis) == "ml_design"