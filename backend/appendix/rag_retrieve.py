"""
RAG Retrieval Service Interface

This module provides an interface to RAGaaS Portal for retrieving
relevant context and documentation to enhance problem-solving capabilities.
"""

import os
import httpx
from typing import Dict, Any, List, Optional
import logging
import asyncio
from dataclasses import dataclass

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """
    Configuration for RAG service connection
    """
    base_url: str
    api_key: str
    timeout: int = 30
    max_retries: int = 3
    default_collection: str = "general"


class RAGRetrievalClient:
    """
    Client for RAGaaS Portal data retrieval
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG retrieval client
        
        Args:
            config: RAG configuration, if None will load from environment
        """
        self.config = config or self._load_config_from_env()
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json"
            }
        )
    
    def _load_config_from_env(self) -> RAGConfig:
        """
        Load configuration from environment variables
        
        Returns:
            RAGConfig instance
            
        Raises:
            ValueError: If required environment variables are missing
        """
        base_url = os.getenv("RAG_SERVICE_BASE_URL")
        api_key = os.getenv("RAG_SERVICE_API_KEY")
        
        if not base_url:
            # Use dummy URL for development/testing
            base_url = "http://localhost:8001"  # Placeholder RAG service
            logger.warning("Using placeholder RAG service URL")
        
        if not api_key:
            api_key = "dummy-rag-key"  # Placeholder for development
            logger.warning("Using placeholder RAG API key")
        
        default_collection = os.getenv("RAG_DEFAULT_COLLECTION", "general")
        
        return RAGConfig(
            base_url=base_url,
            api_key=api_key,
            default_collection=default_collection
        )
    
    async def search_documents(
        self,
        query: str,
        collection: Optional[str] = None,
        limit: int = 5,
        similarity_threshold: float = 0.7,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant documents in the RAG system
        
        Args:
            query: Search query
            collection: Document collection to search (optional)
            limit: Maximum number of results to return
            similarity_threshold: Minimum similarity score for results
            **kwargs: Additional search parameters
            
        Returns:
            List of relevant documents with metadata
            
        Raises:
            httpx.HTTPError: If API request fails
        """
        collection = collection or self.config.default_collection
        
        payload = {
            "query": query,
            "collection": collection,
            "limit": limit,
            "similarity_threshold": similarity_threshold,
            **kwargs
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = await self.client.post(
                    "/api/v1/search",
                    json=payload
                )
                response.raise_for_status()
                
                result = response.json()
                logger.debug(f"RAG search successful (attempt {attempt + 1})")
                return result.get("documents", [])
                
            except httpx.HTTPError as e:
                logger.warning(f"RAG search failed (attempt {attempt + 1}): {e}")
                if attempt == self.config.max_retries - 1:
                    # Return empty results on final failure
                    logger.error("RAG service unavailable, returning empty results")
                    return []
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
    
    async def get_context_for_problem(
        self,
        problem_category: str,
        problem_description: str,
        tech_stack: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get relevant context for a specific problem
        
        Args:
            problem_category: Category of the problem
            problem_description: Detailed problem description
            tech_stack: Optional technology stack information
            
        Returns:
            Relevant context and documentation
        """
        # Construct search queries based on problem characteristics
        queries = [
            f"{problem_category} best practices",
            f"{problem_description}",
            "implementation patterns"
        ]
        
        if tech_stack:
            for tech in tech_stack[:3]:  # Limit to top 3 technologies
                queries.append(f"{tech} documentation examples")
        
        all_results = []
        
        for query in queries:
            try:
                results = await self.search_documents(
                    query,
                    limit=3,  # Fewer results per query
                    similarity_threshold=0.6
                )
                all_results.extend(results)
            except Exception as e:
                logger.warning(f"Failed to search for '{query}': {e}")
        
        return {
            "problem_category": problem_category,
            "relevant_documents": all_results[:10],  # Limit total results
            "context_summary": self._summarize_context(all_results[:5])
        }
    
    def _summarize_context(self, documents: List[Dict[str, Any]]) -> str:
        """
        Create a summary of retrieved context documents
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Context summary string
        """
        if not documents:
            return "No relevant context found in knowledge base."
        
        summary_parts = []
        
        for doc in documents[:3]:  # Summarize top 3 documents
            title = doc.get("title", "Untitled")
            content = doc.get("content", "")
            
            # Extract first sentence or first 100 characters
            if content:
                if "." in content:
                    first_sentence = content.split(".")[0] + "."
                else:
                    first_sentence = content[:100] + "..." if len(content) > 100 else content
                
                summary_parts.append(f"**{title}**: {first_sentence}")
        
        return "\n\n".join(summary_parts)
    
    async def get_implementation_examples(
        self,
        solution_type: str,
        technologies: List[str]
    ) -> Dict[str, Any]:
        """
        Get implementation examples for specific solution types
        
        Args:
            solution_type: Type of solution (AUTOMATION, RAG, etc.)
            technologies: List of technologies to search for
            
        Returns:
            Implementation examples and patterns
        """
        examples = {}
        
        for tech in technologies[:5]:  # Limit to 5 technologies
            try:
                query = f"{solution_type} {tech} implementation example"
                results = await self.search_documents(
                    query,
                    collection="code_examples",  # Specific collection for code
                    limit=2
                )
                
                if results:
                    examples[tech] = results
                    
            except Exception as e:
                logger.warning(f"Failed to get examples for {tech}: {e}")
        
        return {
            "solution_type": solution_type,
            "implementation_examples": examples,
            "example_count": sum(len(examples[tech]) for tech in examples)
        }
    
    async def get_best_practices(
        self,
        domain: str,
        complexity_level: str = "MEDIUM"
    ) -> List[str]:
        """
        Get best practices for a specific domain
        
        Args:
            domain: Problem domain
            complexity_level: Complexity level (LOW, MEDIUM, HIGH)
            
        Returns:
            List of best practices
        """
        try:
            query = f"{domain} best practices {complexity_level} complexity"
            results = await self.search_documents(
                query,
                collection="best_practices",
                limit=5
            )
            
            practices = []
            for doc in results:
                content = doc.get("content", "")
                if content:
                    # Extract bullet points or numbered lists
                    lines = content.split("\n")
                    for line in lines:
                        line = line.strip()
                        if line.startswith("-") or line.startswith("*") or any(line.startswith(f"{i}.") for i in range(1, 10)):
                            practices.append(line)
            
            return practices[:10]  # Return top 10 practices
            
        except Exception as e:
            logger.warning(f"Failed to get best practices: {e}")
            return [
                "Follow Python PEP 8 style guidelines",
                "Implement comprehensive error handling",
                "Include thorough documentation",
                "Write unit tests for critical functionality",
                "Use version control for code management"
            ]
    
    async def close(self):
        """
        Close the HTTP client
        """
        await self.client.aclose()


# Global RAG client instance
_rag_client: Optional[RAGRetrievalClient] = None


async def get_rag_client() -> RAGRetrievalClient:
    """
    Get the global RAG client instance
    
    Returns:
        RAGRetrievalClient instance
    """
    global _rag_client
    
    if _rag_client is None:
        _rag_client = RAGRetrievalClient()
    
    return _rag_client


async def cleanup_rag_client():
    """
    Clean up the global RAG client
    """
    global _rag_client
    
    if _rag_client:
        await _rag_client.close()
        _rag_client = None


async def search_knowledge_base(
    query: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Convenience function to search the knowledge base
    
    Args:
        query: Search query
        limit: Maximum results
        
    Returns:
        Search results
    """
    client = await get_rag_client()
    return await client.search_documents(query, limit=limit)


async def get_contextual_help(
    problem_type: str,
    technologies: List[str]
) -> str:
    """
    Get contextual help for a problem and technology combination
    
    Args:
        problem_type: Type of problem
        technologies: List of technologies
        
    Returns:
        Contextual help text
    """
    client = await get_rag_client()
    
    try:
        context = await client.get_context_for_problem(
            problem_type,
            f"How to implement {problem_type} solution",
            technologies
        )
        
        return context.get("context_summary", "No contextual help available.")
        
    except Exception as e:
        logger.error(f"Failed to get contextual help: {e}")
        return "Contextual help service is currently unavailable."


# Example usage
if __name__ == "__main__":
    async def test_rag():
        client = await get_rag_client()
        
        try:
            # Test document search
            results = await client.search_documents(
                "Python automation best practices"
            )
            print(f"Search results: {len(results)} documents found")
            
            # Test context retrieval
            context = await client.get_context_for_problem(
                "AUTOMATION",
                "Automate data processing workflow",
                ["python", "pandas"]
            )
            print(f"Context: {context['context_summary']}")
            
            # Test best practices
            practices = await client.get_best_practices("Data Processing")
            print(f"Best practices: {practices[:3]}")
            
        except Exception as e:
            print(f"Error: {e}")
        
        finally:
            await cleanup_rag_client()
    
    # Run test
    asyncio.run(test_rag())
