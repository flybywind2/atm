"""
RAG Service Interface for RAGaaS Portal Integration (Backend App Version)

This module provides the RAG service interface for the backend application,
importing from the main appendix implementation while providing backward compatibility.
"""

# Import the main RAG implementation
import sys
import os
import logging
from typing import Dict, List, Any

# Configure logger
logger = logging.getLogger(__name__)

# Add the root appendix directory to the path
root_appendix_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'appendix')
sys.path.insert(0, root_appendix_path)

try:
    from rag_retrieve import (
        RAGService,
        SearchStrategy,
        ContentType,
        RetrievalContext,
        RetrievedDocument,
        get_rag_service,
        enhance_llm_context,
        retrieve_solution_examples,
        get_technology_recommendations
    )
except ImportError:
    # Fallback implementations for testing when main module is not available
    
    async def enhance_llm_context(agent_type: str, query: str, current_context: Dict[str, Any],
                                domain: str = None) -> Dict[str, Any]:
        """Fallback context enhancement for testing"""
        logger.info(f"Fallback: enhancing context for {agent_type} with query: {query[:50]}...")
        enhanced_context = current_context.copy()
        enhanced_context["rag_enhanced"] = False
        enhanced_context["fallback_mode"] = True
        enhanced_context["retrieved_context"] = []
        enhanced_context["context_sources"] = []
        return enhanced_context
    
    async def retrieve_solution_examples(problem_type: str, tech_stack: List[str],
                                       domain: str = None) -> List[Dict[str, Any]]:
        """Fallback solution examples for testing"""
        logger.info(f"Fallback: retrieving solutions for {problem_type} with stack: {tech_stack}")
        return [
            {
                "title": f"Sample {problem_type} Solution",
                "content": f"Basic implementation example using {', '.join(tech_stack[:2])}...",
                "source": "fallback_templates",
                "relevance": 0.5,
                "tech_match": 0.3,
                "domain": domain or "general"
            }
        ]
    
    async def get_technology_recommendations(requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback technology recommendations for testing"""
        logger.info("Fallback: generating technology recommendations")
        complexity = requirements.get("complexity", "medium").lower()
        domain = requirements.get("domain", "general").lower()
        
        recommendations = []
        
        if "web" in domain or "api" in domain:
            recommendations.append({
                "category": "web_framework",
                "technology": "fastapi" if complexity in ["high", "medium"] else "flask",
                "confidence": 0.7,
                "rationale": "Popular choice for Python web development",
                "alternatives": ["django", "tornado"]
            })
        
        recommendations.append({
            "category": "database",
            "technology": "sqlite",
            "confidence": 0.8,
            "rationale": "Simple, embedded database suitable for prototypes",
            "alternatives": ["postgresql", "mysql"]
        })
        
        return recommendations
    
    # Mock classes for compatibility
    class SearchStrategy:
        SEMANTIC = "semantic"
        KEYWORD = "keyword"  
        HYBRID = "hybrid"
        DOMAIN_SPECIFIC = "domain_specific"
    
    class ContentType:
        DOCUMENTATION = "documentation"
        CODE_EXAMPLES = "code_examples"
        TUTORIALS = "tutorials"
        BEST_PRACTICES = "best_practices"
        PATTERNS = "patterns"
        FRAMEWORKS = "frameworks"

# Backward compatibility functions
def retrieve_context(query: str) -> str:
    """Mock RAG retrieval function for backward compatibility"""
    return f"Mock retrieved context for query: {query[:50]}..."

def get_rag_context(query: str) -> str:
    """Mock RAG context retrieval function for backward compatibility"""
    return f"Mock RAG context for query: {query[:50]}..."

async def get_enhanced_rag_context(problem_analysis: Dict[str, Any], context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Enhanced RAG context for solution design"""
    try:
        domain = problem_analysis.get("domain", "business_automation")
        category = problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
        
        enhanced_context = await enhance_llm_context(
            agent_type="solution_designer",
            query=f"solution architecture {category} {domain}",
            current_context=context_data,
            domain=domain
        )
        
        return enhanced_context
        
    except Exception as e:
        logger.error(f"Error getting enhanced RAG context: {str(e)}")
        return context_data