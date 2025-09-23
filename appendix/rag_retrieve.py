"""
RAG Service Interface for RAGaaS Portal Integration

This module implements the complete RAGaaS Portal interface for enhanced
context retrieval and knowledge base integration as specified in the
RAG Service Agent instructions.
"""

import aiohttp
import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum
import hashlib
from datetime import datetime, timedelta


logger = logging.getLogger(__name__)


class SearchStrategy(Enum):
    """Supported search strategies for context retrieval"""
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"
    DOMAIN_SPECIFIC = "domain_specific"


class ContentType(Enum):
    """Content types for filtering"""
    DOCUMENTATION = "documentation"
    CODE_EXAMPLES = "code_examples"
    TUTORIALS = "tutorials"
    BEST_PRACTICES = "best_practices"
    PATTERNS = "patterns"
    FRAMEWORKS = "frameworks"


@dataclass
class RetrievalContext:
    """Context object for retrieval requests"""
    query: str
    domain: Optional[str] = None
    content_types: List[ContentType] = None
    max_results: int = 10
    relevance_threshold: float = 0.7
    include_metadata: bool = True
    search_strategy: SearchStrategy = SearchStrategy.HYBRID


@dataclass
class RetrievedDocument:
    """Represents a retrieved document with metadata"""
    id: str
    title: str
    content: str
    source: str
    relevance_score: float
    domain: str
    content_type: str
    timestamp: datetime
    metadata: Dict[str, Any]
    summary: Optional[str] = None


class RAGService:
    """
    Interface for RAGaaS Portal data retrieval and context enhancement.
    
    This class provides comprehensive RAG functionality including:
    - Context retrieval with multiple strategies
    - Domain-specific knowledge base access
    - Technology recommendation support
    - Caching and optimization mechanisms
    - Error handling and fallback strategies
    """
    
    def __init__(self, endpoint: str, api_key: str, credential_key: str = None):
        """
        Initialize RAG service with connection parameters.
        
        Args:
            endpoint: RAGaaS Portal API endpoint
            api_key: API key for authentication
            credential_key: Credential key for department authentication
        """
        self.endpoint = endpoint.rstrip('/')
        self.api_key = api_key
        self.credential_key = credential_key or "default_credential"
        self.session = None
        self.cache = {}
        self.cache_ttl = 3600  # 1 hour cache TTL
        self.max_retries = 3
        self.timeout = 30
        
        # Domain-specific configurations
        self.domain_configs = {
            "business_automation": {
                "index_name": "business_automation_index",
                "permission_group": ["automation_users"],
                "fields_exclude": ["v_merge_title_content", "raw_html"],
                "boost_keywords": ["process", "automation", "workflow"]
            },
            "data_science": {
                "index_name": "data_science_index", 
                "permission_group": ["data_users"],
                "fields_exclude": ["v_merge_title_content"],
                "boost_keywords": ["machine learning", "model", "dataset"]
            },
            "web_development": {
                "index_name": "web_dev_index",
                "permission_group": ["dev_users"],
                "fields_exclude": ["v_merge_title_content"],
                "boost_keywords": ["framework", "api", "frontend", "backend"]
            },
            "api_development": {
                "index_name": "api_dev_index",
                "permission_group": ["api_users"],
                "fields_exclude": ["v_merge_title_content"],
                "boost_keywords": ["rest", "graphql", "authentication", "documentation"]
            }
        }

    async def __aenter__(self):
        """Async context manager entry"""
        await self._ensure_session()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
            self.session = None

    async def _ensure_session(self):
        """Ensure aiohttp session is available"""
        if not self.session or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)

    async def retrieve_context(self, query: str, domain: str = None, 
                             strategy: SearchStrategy = SearchStrategy.HYBRID,
                             max_results: int = 10) -> List[RetrievedDocument]:
        """
        Retrieve relevant context for given query with specified strategy.
        
        Args:
            query: Search query for context retrieval
            domain: Target domain for domain-specific search
            strategy: Search strategy to use
            max_results: Maximum number of results to return
            
        Returns:
            List of retrieved documents with relevance scores
        """
        try:
            logger.info(f"Retrieving context for query: {query[:100]}...")
            
            # Check cache first
            cache_key = self._generate_cache_key(query, domain, strategy.value)
            cached_result = self._get_cached_result(cache_key)
            if cached_result:
                logger.info("Returning cached context retrieval result")
                return cached_result[:max_results]
            
            await self._ensure_session()
            
            # Prepare search request based on strategy
            search_request = self._prepare_search_request(query, domain, strategy, max_results)
            
            # Execute search with retry logic
            response_data = await self._execute_search_with_retry(search_request)
            
            # Process and rank results
            documents = self._process_search_results(response_data, query, domain)
            
            # Apply relevance filtering
            filtered_documents = [doc for doc in documents if doc.relevance_score >= 0.5]
            
            # Cache results
            self._cache_result(cache_key, filtered_documents)
            
            logger.info(f"Retrieved {len(filtered_documents)} relevant documents")
            return filtered_documents[:max_results]
            
        except Exception as e:
            logger.error(f"Error in context retrieval: {str(e)}")
            return await self._fallback_context_retrieval(query, domain)

    async def search_solutions(self, problem_type: str, tech_stack: List[str],
                             domain: str = None) -> List[RetrievedDocument]:
        """
        Search for existing solutions and implementation examples.
        
        Args:
            problem_type: Type of problem being solved
            tech_stack: List of technologies to consider
            domain: Target domain for search
            
        Returns:
            List of solution documents with implementation examples
        """
        try:
            logger.info(f"Searching solutions for {problem_type} with stack: {tech_stack}")
            
            # Construct solution-focused query
            tech_str = " ".join(tech_stack)
            query = f"{problem_type} implementation {tech_str} solution example"
            
            # Use domain-specific search with solution filtering
            context = RetrievalContext(
                query=query,
                domain=domain,
                content_types=[ContentType.CODE_EXAMPLES, ContentType.PATTERNS],
                max_results=15,
                search_strategy=SearchStrategy.DOMAIN_SPECIFIC
            )
            
            results = await self._domain_specific_search(context)
            
            # Post-process for solution relevance
            solution_docs = []
            for doc in results:
                if self._is_solution_relevant(doc, problem_type, tech_stack):
                    doc.metadata["solution_type"] = problem_type
                    doc.metadata["tech_match"] = self._calculate_tech_match(doc, tech_stack)
                    solution_docs.append(doc)
            
            # Sort by solution relevance
            solution_docs.sort(key=lambda x: (
                x.metadata.get("tech_match", 0),
                x.relevance_score
            ), reverse=True)
            
            logger.info(f"Found {len(solution_docs)} solution documents")
            return solution_docs
            
        except Exception as e:
            logger.error(f"Error in solution search: {str(e)}")
            return await self._fallback_solution_search(problem_type, tech_stack)

    async def get_tech_recommendations(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get technology stack recommendations based on requirements.
        
        Args:
            requirements: Dictionary containing project requirements
            
        Returns:
            List of technology recommendations with rationale
        """
        try:
            logger.info("Getting technology recommendations")
            
            # Extract key requirement elements
            problem_domain = requirements.get("domain", "general")
            complexity = requirements.get("complexity", "medium")
            constraints = requirements.get("constraints", [])
            features = requirements.get("features", [])
            
            # Construct recommendation query
            req_summary = self._summarize_requirements(requirements)
            query = f"technology stack recommendation {problem_domain} {complexity} {req_summary}"
            
            # Search for technology guidance
            context = RetrievalContext(
                query=query,
                domain=problem_domain,
                content_types=[ContentType.FRAMEWORKS, ContentType.BEST_PRACTICES],
                max_results=20,
                search_strategy=SearchStrategy.HYBRID
            )
            
            tech_docs = await self._domain_specific_search(context)
            
            # Process technology recommendations
            recommendations = self._extract_tech_recommendations(tech_docs, requirements)
            
            logger.info(f"Generated {len(recommendations)} technology recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in technology recommendations: {str(e)}")
            return self._fallback_tech_recommendations(requirements)

    async def enhance_agent_context(self, agent_type: str, current_context: Dict[str, Any],
                                  query: str) -> Dict[str, Any]:
        """
        Enhance agent context with relevant RAG-retrieved information.
        
        Args:
            agent_type: Type of agent requesting enhancement
            current_context: Current context available to agent
            query: Specific query for context enhancement
            
        Returns:
            Enhanced context dictionary
        """
        try:
            logger.info(f"Enhancing context for {agent_type} agent")
            
            # Agent-specific enhancement strategies
            enhancement_config = self._get_agent_enhancement_config(agent_type)
            
            # Retrieve relevant context
            retrieved_docs = await self.retrieve_context(
                query=query,
                domain=enhancement_config.get("domain"),
                strategy=SearchStrategy(enhancement_config.get("strategy", "hybrid")),
                max_results=enhancement_config.get("max_results", 10)
            )
            
            # Process and integrate context
            enhanced_context = current_context.copy()
            enhanced_context["rag_enhanced"] = True
            enhanced_context["retrieved_context"] = []
            enhanced_context["context_sources"] = []
            
            for doc in retrieved_docs:
                enhanced_context["retrieved_context"].append({
                    "title": doc.title,
                    "content": doc.summary or doc.content[:500],
                    "relevance": doc.relevance_score,
                    "source": doc.source,
                    "domain": doc.domain
                })
                enhanced_context["context_sources"].append(doc.source)
            
            # Add agent-specific enhancements
            if agent_type == "problem_analyzer":
                enhanced_context["domain_patterns"] = self._extract_domain_patterns(retrieved_docs)
            elif agent_type == "solution_designer":
                enhanced_context["solution_patterns"] = self._extract_solution_patterns(retrieved_docs)
            elif agent_type == "requirements_generator":
                enhanced_context["requirement_templates"] = self._extract_requirement_templates(retrieved_docs)
            
            logger.info(f"Enhanced context with {len(retrieved_docs)} documents")
            return enhanced_context
            
        except Exception as e:
            logger.error(f"Error in context enhancement: {str(e)}")
            return current_context

    def _prepare_search_request(self, query: str, domain: str, 
                               strategy: SearchStrategy, max_results: int) -> Dict[str, Any]:
        """Prepare search request based on parameters"""
        domain_config = self.domain_configs.get(domain, self.domain_configs["business_automation"])
        
        base_request = {
            "index_name": domain_config["index_name"],
            "permission_group": domain_config["permission_group"],
            "query_text": query,
            "num_results_doc": max_results,
            "fields_exclude": domain_config["fields_exclude"]
        }
        
        # Add strategy-specific modifications
        if strategy == SearchStrategy.SEMANTIC:
            base_request["search_type"] = "semantic"
            base_request["boost_semantic"] = 1.5
        elif strategy == SearchStrategy.KEYWORD:
            base_request["search_type"] = "keyword"
            base_request["boost_keywords"] = domain_config.get("boost_keywords", [])
        elif strategy == SearchStrategy.HYBRID:
            base_request["search_type"] = "hybrid"
            base_request["semantic_weight"] = 0.7
            base_request["keyword_weight"] = 0.3
        
        return base_request

    async def _execute_search_with_retry(self, search_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute search request with retry logic"""
        headers = {
            "Content-Type": "application/json",
            "x-dep-ticket": self.credential_key,
            "api-key": self.api_key
        }
        
        for attempt in range(self.max_retries):
            try:
                async with self.session.post(
                    f"{self.endpoint}/query-doc",
                    headers=headers,
                    data=json.dumps(search_request)
                ) as response:
                    if response.status == 200:
                        return await response.json()
                    else:
                        logger.warning(f"Search attempt {attempt + 1} failed with status {response.status}")
                        
            except Exception as e:
                logger.warning(f"Search attempt {attempt + 1} failed: {str(e)}")
                
            if attempt < self.max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        raise Exception(f"Search failed after {self.max_retries} attempts")

    def _process_search_results(self, response_data: Dict[str, Any], 
                               query: str, domain: str) -> List[RetrievedDocument]:
        """Process search results into RetrievedDocument objects"""
        documents = []
        
        if "hits" not in response_data or "hits" not in response_data["hits"]:
            return documents
        
        for hit in response_data["hits"]["hits"]:
            source = hit.get("_source", {})
            score = hit.get("_score", 0.0)
            
            doc = RetrievedDocument(
                id=hit.get("_id", ""),
                title=source.get("title", "Untitled"),
                content=source.get("content", ""),
                source=source.get("source", "Unknown"),
                relevance_score=min(score / 10.0, 1.0),  # Normalize score
                domain=domain or "general",
                content_type=source.get("content_type", "documentation"),
                timestamp=datetime.now(),
                metadata=source.get("metadata", {}),
                summary=source.get("summary")
            )
            
            documents.append(doc)
        
        return documents

    async def _domain_specific_search(self, context: RetrievalContext) -> List[RetrievedDocument]:
        """Perform domain-specific search with enhanced filtering"""
        # Use regular retrieval with domain-specific enhancements
        docs = await self.retrieve_context(
            query=context.query,
            domain=context.domain,
            strategy=context.search_strategy,
            max_results=context.max_results
        )
        
        # Apply content type filtering
        if context.content_types:
            filtered_docs = []
            for doc in docs:
                if any(ct.value in doc.content_type.lower() for ct in context.content_types):
                    filtered_docs.append(doc)
            docs = filtered_docs
        
        return docs

    def _is_solution_relevant(self, doc: RetrievedDocument, problem_type: str, 
                            tech_stack: List[str]) -> bool:
        """Check if document is relevant for solution search"""
        content_lower = doc.content.lower()
        title_lower = doc.title.lower()
        
        # Check for problem type relevance
        problem_keywords = problem_type.lower().split("_")
        problem_match = any(keyword in content_lower or keyword in title_lower 
                          for keyword in problem_keywords)
        
        # Check for technology stack relevance
        tech_match = any(tech.lower() in content_lower or tech.lower() in title_lower 
                        for tech in tech_stack)
        
        return problem_match or tech_match

    def _calculate_tech_match(self, doc: RetrievedDocument, tech_stack: List[str]) -> float:
        """Calculate technology match score for document"""
        content_lower = doc.content.lower()
        title_lower = doc.title.lower()
        
        matches = 0
        for tech in tech_stack:
            tech_lower = tech.lower()
            if tech_lower in content_lower or tech_lower in title_lower:
                matches += 1
        
        return matches / len(tech_stack) if tech_stack else 0.0

    def _summarize_requirements(self, requirements: Dict[str, Any]) -> str:
        """Summarize requirements for search query"""
        summary_parts = []
        
        if "features" in requirements:
            summary_parts.extend(requirements["features"][:3])
        
        if "constraints" in requirements:
            summary_parts.extend(requirements["constraints"][:2])
        
        return " ".join(summary_parts)

    def _extract_tech_recommendations(self, docs: List[RetrievedDocument], 
                                    requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract technology recommendations from retrieved documents"""
        recommendations = []
        
        # Common technology categories
        tech_categories = {
            "web_framework": ["flask", "django", "fastapi", "tornado"],
            "database": ["sqlite", "postgresql", "mysql", "mongodb"],
            "frontend": ["vanilla js", "react", "vue", "angular"],
            "ml_library": ["scikit-learn", "tensorflow", "pytorch", "pandas"],
            "automation": ["selenium", "requests", "schedule", "celery"]
        }
        
        # Analyze documents for technology mentions
        tech_mentions = {}
        for doc in docs:
            content_lower = doc.content.lower()
            for category, techs in tech_categories.items():
                for tech in techs:
                    if tech in content_lower:
                        if category not in tech_mentions:
                            tech_mentions[category] = {}
                        if tech not in tech_mentions[category]:
                            tech_mentions[category][tech] = 0
                        tech_mentions[category][tech] += 1
        
        # Generate recommendations based on mentions and requirements
        for category, techs in tech_mentions.items():
            if techs:
                top_tech = max(techs, key=techs.get)
                recommendations.append({
                    "category": category,
                    "technology": top_tech,
                    "confidence": min(techs[top_tech] / 5.0, 1.0),
                    "rationale": f"Most mentioned in {techs[top_tech]} relevant documents",
                    "alternatives": list(techs.keys())[:3]
                })
        
        return recommendations

    def _get_agent_enhancement_config(self, agent_type: str) -> Dict[str, Any]:
        """Get enhancement configuration for specific agent type"""
        configs = {
            "problem_analyzer": {
                "domain": "business_automation",
                "strategy": "hybrid",
                "max_results": 8
            },
            "context_collector": {
                "domain": "business_automation", 
                "strategy": "semantic",
                "max_results": 5
            },
            "requirements_generator": {
                "domain": "business_automation",
                "strategy": "keyword",
                "max_results": 10
            },
            "solution_designer": {
                "domain": "web_development",
                "strategy": "hybrid",
                "max_results": 12
            },
            "guide_creator": {
                "domain": "api_development",
                "strategy": "domain_specific",
                "max_results": 15
            }
        }
        
        return configs.get(agent_type, configs["problem_analyzer"])

    def _extract_domain_patterns(self, docs: List[RetrievedDocument]) -> List[str]:
        """Extract domain-specific patterns from documents"""
        patterns = []
        for doc in docs:
            if "pattern" in doc.title.lower() or "pattern" in doc.content.lower():
                patterns.append(f"{doc.title}: {doc.summary or doc.content[:200]}")
        return patterns[:5]

    def _extract_solution_patterns(self, docs: List[RetrievedDocument]) -> List[str]:
        """Extract solution patterns from documents"""
        patterns = []
        for doc in docs:
            if any(keyword in doc.content.lower() 
                  for keyword in ["solution", "implementation", "approach", "design"]):
                patterns.append(f"{doc.title}: {doc.summary or doc.content[:200]}")
        return patterns[:5]

    def _extract_requirement_templates(self, docs: List[RetrievedDocument]) -> List[str]:
        """Extract requirement templates from documents"""
        templates = []
        for doc in docs:
            if any(keyword in doc.content.lower() 
                  for keyword in ["requirement", "specification", "criteria", "acceptance"]):
                templates.append(f"{doc.title}: {doc.summary or doc.content[:200]}")
        return templates[:5]

    def _generate_cache_key(self, query: str, domain: str, strategy: str) -> str:
        """Generate cache key for query"""
        content = f"{query}_{domain}_{strategy}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cached_result(self, cache_key: str) -> Optional[List[RetrievedDocument]]:
        """Get cached result if valid"""
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return cached_data
            else:
                del self.cache[cache_key]
        return None

    def _cache_result(self, cache_key: str, result: List[RetrievedDocument]):
        """Cache search result"""
        self.cache[cache_key] = (result, datetime.now())
        
        # Simple cache size management
        if len(self.cache) > 100:
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]

    async def _fallback_context_retrieval(self, query: str, domain: str) -> List[RetrievedDocument]:
        """Fallback context retrieval when main service fails"""
        logger.info("Using fallback context retrieval")
        
        # Return mock documents for testing/fallback
        fallback_docs = [
            RetrievedDocument(
                id="fallback_1",
                title="General Problem Solving Guide",
                content="When facing automation challenges, consider breaking down the problem into smaller components...",
                source="internal_knowledge",
                relevance_score=0.6,
                domain=domain or "general",
                content_type="documentation",
                timestamp=datetime.now(),
                metadata={"fallback": True},
                summary="General guidance for problem solving"
            )
        ]
        
        return fallback_docs

    async def _fallback_solution_search(self, problem_type: str, tech_stack: List[str]) -> List[RetrievedDocument]:
        """Fallback solution search when main service fails"""
        logger.info("Using fallback solution search")
        
        return [
            RetrievedDocument(
                id="solution_fallback_1",
                title=f"Basic {problem_type} Implementation",
                content=f"Sample implementation using {', '.join(tech_stack[:2])}...",
                source="internal_templates",
                relevance_score=0.5,
                domain="general",
                content_type="code_examples",
                timestamp=datetime.now(),
                metadata={"fallback": True, "tech_stack": tech_stack},
                summary=f"Fallback solution for {problem_type}"
            )
        ]

    def _fallback_tech_recommendations(self, requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback technology recommendations when main service fails"""
        logger.info("Using fallback technology recommendations")
        
        # Basic recommendations based on common patterns
        complexity = requirements.get("complexity", "medium").lower()
        domain = requirements.get("domain", "general").lower()
        
        fallback_recommendations = []
        
        if "web" in domain or "api" in domain:
            fallback_recommendations.append({
                "category": "web_framework",
                "technology": "fastapi" if complexity in ["high", "medium"] else "flask",
                "confidence": 0.7,
                "rationale": "Popular choice for Python web development",
                "alternatives": ["django", "tornado"]
            })
        
        if "data" in domain or "ml" in domain:
            fallback_recommendations.append({
                "category": "ml_library",
                "technology": "scikit-learn",
                "confidence": 0.8,
                "rationale": "Standard choice for machine learning in Python",
                "alternatives": ["pandas", "numpy"]
            })
        
        return fallback_recommendations


# Singleton instance for easy access
_rag_service_instance = None


async def get_rag_service(endpoint: str = "http://localhost:8000", 
                         api_key: str = "default_api_key",
                         credential_key: str = "default_credential") -> RAGService:
    """
    Get or create RAG service instance.
    
    Args:
        endpoint: RAGaaS Portal endpoint
        api_key: API key for authentication
        credential_key: Credential key for department authentication
        
    Returns:
        RAGService instance
    """
    global _rag_service_instance
    
    if _rag_service_instance is None:
        _rag_service_instance = RAGService(endpoint, api_key, credential_key)
    
    return _rag_service_instance


async def enhance_llm_context(agent_type: str, query: str, current_context: Dict[str, Any],
                            domain: str = None) -> Dict[str, Any]:
    """
    Convenience function to enhance LLM context with RAG-retrieved information.
    
    Args:
        agent_type: Type of agent requesting enhancement
        query: Query for context retrieval
        current_context: Current context available
        domain: Target domain for search
        
    Returns:
        Enhanced context with RAG-retrieved information
    """
    try:
        rag_service = await get_rag_service()
        async with rag_service:
            enhanced_context = await rag_service.enhance_agent_context(
                agent_type=agent_type,
                current_context=current_context,
                query=query
            )
            return enhanced_context
    except Exception as e:
        logger.error(f"Failed to enhance context: {str(e)}")
        return current_context


async def retrieve_solution_examples(problem_type: str, tech_stack: List[str],
                                   domain: str = None) -> List[Dict[str, Any]]:
    """
    Convenience function to retrieve solution examples.
    
    Args:
        problem_type: Type of problem being solved
        tech_stack: List of technologies to consider
        domain: Target domain for search
        
    Returns:
        List of solution examples with metadata
    """
    try:
        rag_service = await get_rag_service()
        async with rag_service:
            solutions = await rag_service.search_solutions(problem_type, tech_stack, domain)
            return [
                {
                    "title": doc.title,
                    "content": doc.content,
                    "source": doc.source,
                    "relevance": doc.relevance_score,
                    "tech_match": doc.metadata.get("tech_match", 0),
                    "domain": doc.domain
                }
                for doc in solutions
            ]
    except Exception as e:
        logger.error(f"Failed to retrieve solution examples: {str(e)}")
        return []


async def get_technology_recommendations(requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convenience function to get technology recommendations.
    
    Args:
        requirements: Project requirements dictionary
        
    Returns:
        List of technology recommendations
    """
    try:
        rag_service = await get_rag_service()
        async with rag_service:
            recommendations = await rag_service.get_tech_recommendations(requirements)
            return recommendations
    except Exception as e:
        logger.error(f"Failed to get technology recommendations: {str(e)}")
        return []