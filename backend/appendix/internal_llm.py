"""
LLM Integration Service

Provides unified interface for LLM services supporting both internal OpenAI-compatible
API and external Ollama for development/testing. Includes environment-based
configuration switching, error handling, fallback mechanisms, and agent-specific
prompt engineering.

Phase 5 implementation as specified in plan.md and LLM Integration Agent requirements.
"""

import os
import json
import uuid
import httpx
import asyncio
from enum import Enum
from typing import Dict, Any, List, Optional, Union, AsyncGenerator
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import logging
from functools import wraps

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class LLMProvider(Enum):
    """Supported LLM providers"""
    INTERNAL = "internal"
    OLLAMA = "ollama"
    OPENAI = "openai"


class AgentType(Enum):
    """Agent types for specific prompt engineering"""
    PROBLEM_ANALYZER = "problem_analyzer"
    CONTEXT_COLLECTOR = "context_collector"
    REQUIREMENTS_GENERATOR = "requirements_generator"
    SOLUTION_DESIGNER = "solution_designer"
    GUIDE_CREATOR = "guide_creator"


@dataclass
class LLMConfig:
    """
    Configuration for LLM API connection
    """
    provider: LLMProvider
    base_url: str
    api_key: str = "dummy-key"
    model: str = "gpt-3.5-turbo"
    timeout: int = 60
    max_retries: int = 3
    retry_delay: float = 1.0
    max_tokens: Optional[int] = None
    temperature: float = 0.7
    custom_headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class LLMResponse:
    """Standardized LLM response format"""
    content: str
    usage: Optional[Dict[str, int]] = None
    model: Optional[str] = None
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        wait_time = delay * (2 ** attempt)
                        logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.error(f"All {max_retries} attempts failed for {func.__name__}")
            raise last_exception
        return wrapper
    return decorator


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients"""
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.client: Optional[httpx.AsyncClient] = None
    
    @abstractmethod
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion"""
        pass
    
    @abstractmethod
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion"""
        pass
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
            self.client = None


class InternalLLMClient(BaseLLMClient):
    """
    Client for internal OpenAI-compatible LLM API with enterprise headers
    """
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._setup_client()
    
    def _setup_client(self):
        """Setup HTTP client with proper headers"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            # Enterprise-specific headers
            "x-dep-ticket": self.config.custom_headers.get("x-dep-ticket", "default-ticket"),
            "Send-System-Name": self.config.custom_headers.get("Send-System-Name", "ATM-System"),
            "User-ID": self.config.custom_headers.get("User-ID", "system-user"),
            "User-Type": self.config.custom_headers.get("User-Type", "AD"),
            "Prompt-Msg-Id": str(uuid.uuid4()),
            "Completion-Msg-Id": str(uuid.uuid4()),
        }
        headers.update(self.config.custom_headers)
        
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers=headers
        )
    
    @retry_on_failure()
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion using internal API"""
        if not self.client:
            self._setup_client()
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            **kwargs
        }
        
        if max_tokens or self.config.max_tokens:
            payload["max_tokens"] = max_tokens or self.config.max_tokens
        
        response = await self.client.post(
            "/v1/chat/completions",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        
        return LLMResponse(
            content=result["choices"][0]["message"]["content"],
            usage=result.get("usage"),
            model=result.get("model"),
            finish_reason=result["choices"][0].get("finish_reason"),
            raw_response=result
        )
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming chat completion"""
        if not self.client:
            self._setup_client()
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": True,
            **kwargs
        }
        
        async with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


class OllamaLLMClient(BaseLLMClient):
    """Client for Ollama API (external/development use)"""
    
    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self._setup_client()
    
    def _setup_client(self):
        """Setup HTTP client for Ollama"""
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers={"Content-Type": "application/json"}
        )
    
    @retry_on_failure()
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion using Ollama API"""
        if not self.client:
            self._setup_client()
        
        # Convert messages to Ollama format
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature or self.config.temperature,
            }
        }
        
        response = await self.client.post(
            "/api/generate",
            json=payload
        )
        response.raise_for_status()
        
        result = response.json()
        
        return LLMResponse(
            content=result["response"],
            model=result.get("model"),
            finish_reason="stop" if result.get("done") else "length",
            raw_response=result
        )
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion using Ollama"""
        if not self.client:
            self._setup_client()
        
        prompt = self._messages_to_prompt(messages)
        
        payload = {
            "model": self.config.model,
            "prompt": prompt,
            "stream": True,
        }
        
        async with self.client.stream(
            "POST",
            "/api/generate",
            json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                try:
                    chunk = json.loads(line)
                    content = chunk.get("response", "")
                    if content:
                        yield content
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI-style messages to Ollama prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        return "\n\n".join(prompt_parts) + "\n\nAssistant:"


class LLMManager:
    """Main LLM service manager with environment-based provider switching"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or self._load_config_from_env()
        self.client: Optional[BaseLLMClient] = None
        self.fallback_clients: List[BaseLLMClient] = []
        self._initialize_clients()
    
    def _load_config_from_env(self) -> LLMConfig:
        """Load configuration from environment variables"""
        env = os.getenv("ENVIRONMENT", "development")
        
        # Environment-based configuration
        if env == "production":
            provider = LLMProvider.INTERNAL
            base_url = os.getenv("INTERNAL_LLM_BASE_URL", "https://internal-api.company.com")
            api_key = os.getenv("INTERNAL_LLM_API_KEY", "")
            model = os.getenv("INTERNAL_LLM_MODEL", "gpt-4")
            custom_headers = {
                "x-dep-ticket": os.getenv("INTERNAL_LLM_TICKET", ""),
                "Send-System-Name": "ATM-System",
                "User-ID": os.getenv("USER_ID", "system"),
                "User-Type": "AD"
            }
        else:
            # Development/testing environment
            provider = LLMProvider.OLLAMA
            base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
            api_key = "dummy-key"
            model = os.getenv("OLLAMA_MODEL", "llama3.1")
            custom_headers = {}
        
        return LLMConfig(
            provider=provider,
            base_url=base_url,
            api_key=api_key,
            model=model,
            custom_headers=custom_headers,
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "3")),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
        )
    
    def _initialize_clients(self):
        """Initialize primary and fallback clients"""
        # Primary client
        if self.config.provider == LLMProvider.INTERNAL:
            self.client = InternalLLMClient(self.config)
        elif self.config.provider == LLMProvider.OLLAMA:
            self.client = OllamaLLMClient(self.config)
        
        # Setup fallback clients
        self._setup_fallback_clients()
    
    def _setup_fallback_clients(self):
        """Setup fallback clients for resilience"""
        if self.config.provider == LLMProvider.INTERNAL:
            # Fallback to Ollama for internal
            ollama_config = LLMConfig(
                provider=LLMProvider.OLLAMA,
                base_url="http://localhost:11434",
                model="llama3.1"
            )
            self.fallback_clients.append(OllamaLLMClient(ollama_config))
        
        # Could add OpenAI fallback here if needed
        # openai_config = LLMConfig(
        #     provider=LLMProvider.OPENAI,
        #     base_url="https://api.openai.com/v1",
        #     api_key=os.getenv("OPENAI_API_KEY", ""),
        #     model="gpt-3.5-turbo"
        # )
        # self.fallback_clients.append(OpenAILLMClient(openai_config))
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> LLMResponse:
        """Generate chat completion with fallback support"""
        last_error = None
        
        # Try primary client
        try:
            return await self.client.chat_completion(messages, **kwargs)
        except Exception as e:
            logger.warning(f"Primary LLM client failed: {e}")
            last_error = e
        
        # Try fallback clients
        for fallback_client in self.fallback_clients:
            try:
                logger.info(f"Trying fallback client: {fallback_client.__class__.__name__}")
                return await fallback_client.chat_completion(messages, **kwargs)
            except Exception as e:
                logger.warning(f"Fallback client failed: {e}")
                last_error = e
        
        # All clients failed
        raise Exception(f"All LLM clients failed. Last error: {last_error}")
    
    async def simple_completion(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> str:
        """Simple completion interface for single prompts"""
        messages = []
        
        if system_message:
            messages.append({"role": "system", "content": system_message})
        
        messages.append({"role": "user", "content": prompt})
        
        response = await self.chat_completion(messages, **kwargs)
        return response.content
    
    async def stream_completion(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Generate streaming completion with fallback support"""
        try:
            async for chunk in self.client.stream_completion(messages, **kwargs):
                yield chunk
        except Exception as e:
            logger.warning(f"Streaming failed, falling back to regular completion: {e}")
            response = await self.chat_completion(messages, **kwargs)
            yield response.content
    
    async def close(self):
        """Close all clients"""
        if self.client:
            await self.client.close()
        
        for fallback_client in self.fallback_clients:
            await fallback_client.close()


class AgentPromptTemplates:
    """Centralized prompt templates for different agent types"""
    
    @staticmethod
    def get_system_prompt(agent_type: AgentType) -> str:
        """Get system prompt for specific agent type"""
        prompts = {
            AgentType.PROBLEM_ANALYZER: """
You are a business analyst specializing in problem decomposition and analysis.
You excel at breaking down complex business problems into structured, actionable components.

Analyze problems systematically:
1. Identify core issues and pain points
2. Categorize the problem type (AUTOMATION, RAG, ML_CLASSIFICATION, etc.)
3. Assess complexity and resource requirements
4. Identify stakeholders and success criteria

Always respond in valid JSON format with clear, actionable insights.
""",
            
            AgentType.CONTEXT_COLLECTOR: """
You are an expert at asking clarifying questions to gather complete context.
Your goal is to identify missing information that's critical for solution design.

Generate specific, actionable questions that will help:
1. Understand technical constraints and requirements
2. Clarify business objectives and success metrics
3. Identify integration points and dependencies
4. Understand user personas and workflows

Prioritize questions that will have the biggest impact on solution quality.
""",
            
            AgentType.REQUIREMENTS_GENERATOR: """
You are a technical writer creating professional Software Requirements Specifications (SRS).
You excel at translating business problems into comprehensive technical requirements.

Generate complete requirements including:
1. Functional requirements with clear acceptance criteria
2. Non-functional requirements (performance, security, scalability)
3. User stories with detailed scenarios
4. System constraints and assumptions
5. Integration requirements

Use professional SRS format with clear, testable requirements.
""",
            
            AgentType.SOLUTION_DESIGNER: """
You are a solution architect recommending optimal technology stacks and approaches.
You excel at matching problems to appropriate solution patterns.

Provide solutions that consider:
1. Problem complexity and scale
2. Available resources and constraints
3. Future maintenance and scalability
4. Integration with existing systems
5. Cost and time-to-market factors

Recommend specific technologies with clear justifications.
""",
            
            AgentType.GUIDE_CREATOR: """
You are a senior developer creating detailed implementation guides.
You excel at breaking down solutions into actionable development plans.

Create comprehensive guides including:
1. Detailed Work Breakdown Structure (WBS)
2. Code examples and best practices
3. Testing strategies and validation approaches
4. Deployment and monitoring considerations
5. Documentation and maintenance guidelines

Focus on practical, executable guidance that developers can follow immediately.
"""
        }
        
        return prompts.get(agent_type, "You are a helpful AI assistant.")
    
    @staticmethod
    def format_problem_analysis_prompt(
        problem_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format prompt for problem analysis"""
        context_str = ""
        if context:
            context_str = f"\n\nAdditional context: {json.dumps(context, indent=2)}"
        
        return f"""
Problem Description: {problem_description}{context_str}

Provide a structured analysis in the following JSON format:
{{
    "title": "Brief problem title",
    "domain": "Business domain (e.g., 'Data Processing', 'Automation')",
    "category": "Problem category (AUTOMATION, RAG, ML_CLASSIFICATION, etc.)",
    "complexity": "Complexity level (LOW, MEDIUM, HIGH)",
    "stakeholders": ["List of stakeholders"],
    "pain_points": ["List of pain points"],
    "current_state": "Description of current situation",
    "desired_state": "Description of desired outcome",
    "constraints": ["Technical or business constraints"],
    "success_criteria": ["Measurable success criteria"]
}}
"""
    
    @staticmethod
    def format_context_collection_prompt(
        problem_analysis: Dict[str, Any],
        existing_context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format prompt for context collection"""
        existing_str = ""
        if existing_context:
            existing_str = f"\n\nExisting context: {json.dumps(existing_context, indent=2)}"
        
        return f"""
Problem Analysis: {json.dumps(problem_analysis, indent=2)}{existing_str}

Based on this problem analysis, identify what additional information is needed.
Generate 3-5 specific questions that will help clarify:
1. Technical requirements and constraints
2. Business objectives and success metrics
3. User needs and workflows
4. Integration requirements
5. Resource and timeline constraints

Return as a JSON array of strings:
["Question 1?", "Question 2?", "Question 3?"]
"""


class LLMAgentService:
    """High-level service for agent-specific LLM operations"""
    
    def __init__(self, llm_manager: Optional[LLMManager] = None):
        self.llm_manager = llm_manager or LLMManager()
        self.templates = AgentPromptTemplates()
    
    async def analyze_problem(
        self,
        problem_description: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Analyze a problem using the Problem Analyzer agent"""
        system_prompt = self.templates.get_system_prompt(AgentType.PROBLEM_ANALYZER)
        user_prompt = self.templates.format_problem_analysis_prompt(problem_description, context)
        
        response = await self.llm_manager.simple_completion(
            user_prompt,
            system_prompt,
            temperature=0.3
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse problem analysis as JSON")
            return {
                "title": "Problem Analysis",
                "domain": "General",
                "category": "GENERAL_PROBLEM_SOLVING",
                "complexity": "MEDIUM",
                "raw_response": response
            }
    
    async def collect_context_questions(
        self,
        problem_analysis: Dict[str, Any],
        existing_context: Optional[Dict[str, Any]] = None
    ) -> List[str]:
        """Generate context collection questions"""
        system_prompt = self.templates.get_system_prompt(AgentType.CONTEXT_COLLECTOR)
        user_prompt = self.templates.format_context_collection_prompt(problem_analysis, existing_context)
        
        response = await self.llm_manager.simple_completion(
            user_prompt,
            system_prompt,
            temperature=0.4
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse context questions as JSON")
            return [
                "What are the main technical constraints for this solution?",
                "Who are the primary users and what are their workflows?",
                "What systems need to integrate with this solution?"
            ]
    
    async def generate_requirements(
        self,
        problem_analysis: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Generate requirements document"""
        system_prompt = self.templates.get_system_prompt(AgentType.REQUIREMENTS_GENERATOR)
        title = problem_analysis.get("title", "소프트웨어 솔루션")
        user_prompt = f"""
문제 분석(Problem Analysis): {json.dumps(problem_analysis, indent=2, ensure_ascii=False)}

맥락 정보(Context Information): {json.dumps(context, indent=2, ensure_ascii=False)}

아래 지침에 따라 한국어(ko-KR)로 완전한 소프트웨어 요구사항 명세서(SRS)를 마크다운으로 작성하세요.
- H1 제목은 반드시 다음을 그대로 사용: "소프트웨어 요구사항 명세서 (SRS) - {title}"
- 기능/비기능 요구사항은 테스트 가능한 수용 기준을 포함
- 사용자 스토리(As a, I want, so that)를 한국어로 자연스럽게 재작성
- 통합/제약/보안/성능 요구사항 포함
"""

        return await self.llm_manager.simple_completion(
            user_prompt,
            system_prompt,
            temperature=0.4
        )
    
    async def design_solution(
        self,
        requirements: str,
        problem_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Design solution architecture"""
        system_prompt = self.templates.get_system_prompt(AgentType.SOLUTION_DESIGNER)
        
        user_prompt = f"""
Requirements Document:
{requirements}

Problem Analysis: {json.dumps(problem_analysis, indent=2)}

Design a technical solution including:
1. Solution type classification (AUTOMATION, RAG, ML_CLASSIFICATION, etc.)
2. Recommended technology stack
3. Architecture overview
4. Implementation approach
5. Integration considerations

Return as JSON:
{{
    "solution_type": "Solution category",
    "technology_stack": {{
        "framework": "Recommended framework",
        "libraries": ["List of libraries"],
        "database": "Database recommendation",
        "deployment": "Deployment approach"
    }},
    "architecture_overview": "Architecture description",
    "implementation_approach": "Step-by-step approach",
    "integration_points": ["Integration requirements"]
}}
"""
        
        response = await self.llm_manager.simple_completion(
            user_prompt,
            system_prompt,
            temperature=0.4
        )
        
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            logger.warning("Failed to parse solution design as JSON")
            return {
                "solution_type": problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING"),
                "raw_response": response
            }
    
    async def create_implementation_guide(
        self,
        solution_design: Dict[str, Any],
        requirements: str
    ) -> str:
        """Create detailed implementation guide"""
        system_prompt = self.templates.get_system_prompt(AgentType.GUIDE_CREATOR)
        
        user_prompt = f"""
Solution Design: {json.dumps(solution_design, indent=2)}

Requirements Document:
{requirements}

Create a comprehensive implementation guide in Markdown format. Write the entire output in Korean (ko-KR).
Include the following sections:
1. Project Setup and Environment
2. Work Breakdown Structure (WBS)
3. Code Examples and Best Practices
4. Testing Strategy
5. Deployment Guidelines
6. Monitoring and Maintenance
7. Documentation Requirements

Provide specific, actionable guidance that developers can follow immediately.
"""
        
        return await self.llm_manager.simple_completion(
            user_prompt,
            system_prompt,
            temperature=0.5
        )
    
    async def close(self):
        """Close LLM manager"""
        await self.llm_manager.close()


# Global service instances
_llm_manager: Optional[LLMManager] = None
_agent_service: Optional[LLMAgentService] = None


async def get_llm_manager() -> LLMManager:
    """Get the global LLM manager instance"""
    global _llm_manager
    
    if _llm_manager is None:
        _llm_manager = LLMManager()
    
    return _llm_manager


async def get_agent_service() -> LLMAgentService:
    """Get the global LLM agent service instance"""
    global _agent_service
    
    if _agent_service is None:
        llm_manager = await get_llm_manager()
        _agent_service = LLMAgentService(llm_manager)
    
    return _agent_service


async def cleanup_llm_services():
    """Clean up all global LLM services"""
    global _llm_manager, _agent_service
    
    if _agent_service:
        await _agent_service.close()
        _agent_service = None
    
    if _llm_manager:
        await _llm_manager.close()
        _llm_manager = None


# Legacy compatibility functions
async def get_llm_client() -> LLMManager:
    """Legacy function for backward compatibility"""
    return await get_llm_manager()


async def cleanup_llm_client():
    """Legacy function for backward compatibility"""
    await cleanup_llm_services()


# Example usage and testing
if __name__ == "__main__":
    async def test_llm_integration():
        """Test the complete LLM integration system"""
        try:
            # Get agent service
            agent_service = await get_agent_service()
            
            print("Testing LLM Integration System...")
            
            # Test 1: Problem Analysis
            print("\n1. Testing Problem Analysis...")
            problem_description = "We need to automate our monthly report generation process that currently takes 2 days of manual work"
            analysis = await agent_service.analyze_problem(problem_description)
            print(f"Analysis: {json.dumps(analysis, indent=2)}")
            
            # Test 2: Context Collection
            print("\n2. Testing Context Collection...")
            questions = await agent_service.collect_context_questions(analysis)
            print(f"Context Questions: {questions}")
            
            # Test 3: Requirements Generation
            print("\n3. Testing Requirements Generation...")
            context = {
                "users": "Finance team, Management",
                "data_sources": "SQL database, Excel files",
                "current_tools": "Manual Excel processing",
                "timeline": "3 months"
            }
            requirements = await agent_service.generate_requirements(analysis, context)
            print(f"Requirements (first 500 chars): {requirements[:500]}...")
            
            # Test 4: Solution Design
            print("\n4. Testing Solution Design...")
            solution = await agent_service.design_solution(requirements, analysis)
            print(f"Solution: {json.dumps(solution, indent=2)}")
            
            # Test 5: Implementation Guide
            print("\n5. Testing Implementation Guide...")
            guide = await agent_service.create_implementation_guide(solution, requirements)
            print(f"Guide (first 500 chars): {guide[:500]}...")
            
            print("\nAll tests completed successfully!")
            
        except Exception as e:
            print(f"Test failed: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await cleanup_llm_services()
    
    async def test_environment_switching():
        """Test environment-based configuration switching"""
        print("\nTesting Environment Switching...")
        
        # Test development environment
        os.environ["ENVIRONMENT"] = "development"
        dev_manager = LLMManager()
        print(f"Development config: {dev_manager.config.provider}, {dev_manager.config.base_url}")
        
        # Test production environment
        os.environ["ENVIRONMENT"] = "production"
        prod_manager = LLMManager()
        print(f"Production config: {prod_manager.config.provider}, {prod_manager.config.base_url}")
        
        await dev_manager.close()
        await prod_manager.close()
    
    # Run comprehensive tests
    print("Starting LLM Integration Tests...")
    asyncio.run(test_environment_switching())
    asyncio.run(test_llm_integration())
