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

        # 디버깅을 위한 응답 로깅
        logger.info(f"=== OLLAMA RAW RESPONSE ===")
        logger.info(f"Response length: {len(result.get('response', ''))}")
        logger.info(f"Response preview: {result.get('response', '')[:200]}...")
        logger.info(f"Model: {result.get('model')}")
        logger.info(f"Done: {result.get('done')}")

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

**IMPORTANT**: Pay special attention to Machine Learning and Data Science problems. Look for keywords like:
- 예측 (prediction), 분류 (classification), 모델 (model), 패턴 (pattern)
- 이상치 (anomaly), 원인 파악 (root cause analysis), 데이터 분석 (data analysis)
- 머신러닝, 기계학습, 알고리즘, 신경망, 회귀분석
- English: predict, classify, model, pattern, anomaly, regression, neural network, algorithm

**Problem Categories**:
- **ML_CLASSIFICATION**: 분류, 예측, 패턴인식 문제 (예: 스팸 분류, 이미지 인식, 텍스트 분류)
- **ML_REGRESSION**: 수치 예측, 회귀분석 문제 (예: 가격 예측, 수요 예측, 성능 예측)
- **ML_CLUSTERING**: 군집화, 세그멘테이션 문제 (예: 고객 세분화, 데이터 그룹핑)
- **AUTOMATION**: 단순 반복 작업 자동화 (예: 파일 처리, 스케줄링, API 호출)
- **INFORMATION_RETRIEVAL**: 정보 검색, 문서 조회 시스템
- **DATA_VISUALIZATION**: 차트, 대시보드, 리포트 생성

**특히 폐수처리, 수질관리, 환경 모니터링 등의 문제는 대부분 ML_CLASSIFICATION 또는 ML_REGRESSION 문제입니다.**

Analyze problems systematically:
1. Identify core issues and pain points
2. **CAREFULLY** categorize the problem type based on the keywords above
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
        user_prompt = f"""
문제 설명: {problem_description}

다음 JSON 형식으로만 응답해주세요. 다른 설명 없이 오직 JSON만 반환하세요:

{{
    "title": "문제 제목",
    "domain": "비즈니스 도메인 (예: '데이터 처리', '자동화')",
    "category": "AUTOMATION",
    "complexity": "MEDIUM",
    "stakeholders": ["이해관계자 목록"],
    "pain_points": ["문제점 목록"],
    "current_state": "현재 상황 설명",
    "desired_state": "원하는 결과 설명",
    "constraints": ["기술적/비즈니스 제약사항"],
    "success_criteria": ["측정 가능한 성공 기준"]
}}

다시 한번 강조: JSON 형식으로만 응답하고, 앞뒤 설명은 넣지 마세요.
"""

        response = await self.llm_manager.simple_completion(
            user_prompt,
            system_prompt,
            temperature=0.1
        )

        try:
            # 응답에서 JSON 부분만 추출 시도
            response_cleaned = response.strip()

            # ```json 블록 제거
            if response_cleaned.startswith('```json'):
                response_cleaned = response_cleaned[7:]
            elif response_cleaned.startswith('```'):
                response_cleaned = response_cleaned[3:]

            if response_cleaned.endswith('```'):
                response_cleaned = response_cleaned[:-3]

            response_cleaned = response_cleaned.strip()

            # 첫 번째와 마지막 중괄호 찾기
            start_idx = response_cleaned.find('{')
            end_idx = response_cleaned.rfind('}')

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_cleaned[start_idx:end_idx+1]
                parsed_json = json.loads(json_str)
                logger.info(f"=== JSON PARSING SUCCESSFUL ===")
                logger.info(f"Parsed category: {parsed_json.get('category', 'None')}")

                # 카테고리 검증 및 수정
                corrected_json = self._validate_and_correct_category(problem_description, parsed_json)
                if corrected_json.get('category') != parsed_json.get('category'):
                    logger.info(f"=== CATEGORY CORRECTED ===")
                    logger.info(f"Original: {parsed_json.get('category')} -> Corrected: {corrected_json.get('category')}")

                return corrected_json
            else:
                raise json.JSONDecodeError("No JSON found", response, 0)

        except json.JSONDecodeError as e:
            logger.error(f"=== JSON PARSING FAILED - NO FALLBACK ===")
            logger.error(f"Failed to parse problem analysis as JSON: {e}")
            logger.error(f"Raw response: {response[:500]}...")
            logger.error(f"Cleaned response: {response_cleaned[:300]}...")
            if 'json_str' in locals():
                logger.error(f"Extracted JSON string: {json_str[:300]}...")

            # 바로 예외를 발생시켜 문제를 즉시 알림
            raise ValueError(f"LLM failed to generate valid JSON response: {e}. Response: {response[:200]}")

    def _validate_and_correct_category(self, problem_description: str, parsed_json: Dict[str, Any]) -> Dict[str, Any]:
        """JSON 응답의 카테고리를 검증하고 필요시 수정"""
        problem_lower = problem_description.lower()
        original_category = parsed_json.get('category', 'AUTOMATION')

        # INTEGRATION keywords (highest priority)
        integration_keywords = [
            "시스템 통합", "system integration", "연동", "api", "통합", "integrate", "connect",
            "sync", "synchronization", "동기화", "crm", "erp", "warehouse", "order management",
            "주문 관리", "창고 관리", "재고", "inventory", "dashboard", "대시보드", "reporting",
            "보고서", "자동화된 보고", "automated reporting"
        ]

        # ML keywords for environmental/industrial problems (but excluding common integration terms)
        specific_ml_keywords = [
            # Environmental/Industrial specific
            "폐수", "수질", "오염", "환경", "농도", "수치", "측정", "품질",
            "wastewater", "water", "quality", "pollution", "environmental", "concentration",
            "measurement", "anomaly detection", "outlier", "treatment", "처리",
            # ML specific terms
            "예측", "분류", "모델", "머신러닝", "기계학습", "알고리즘", "신경망", "회귀", "이상치", "원인", "파악",
            "predict", "classify", "model", "ml", "machine learning", "neural", "algorithm", "regression"
        ]

        # Check for INTEGRATION keywords first (takes priority)
        integration_keyword_count = sum(1 for keyword in integration_keywords if keyword in problem_lower)

        # Check for specific ML keywords (excluding generic terms like "data", "analysis")
        ml_keyword_count = sum(1 for keyword in specific_ml_keywords if keyword in problem_lower)

        logger.info(f"=== CATEGORY VALIDATION ===")
        logger.info(f"Integration keywords found: {integration_keyword_count}")
        logger.info(f"ML keywords found: {ml_keyword_count}")
        logger.info(f"Original category: {original_category}")

        # Priority 1: INTEGRATION (if integration keywords found)
        if integration_keyword_count >= 2:
            logger.info(f"=== CORRECTING TO INTEGRATION ===")
            corrected_json = parsed_json.copy()
            corrected_json['category'] = "INTEGRATION"
            corrected_json['domain'] = "시스템 통합 및 연동"
            corrected_json['title'] = "시스템 통합 솔루션"
            return corrected_json

        # Priority 2: ML categories (only if specific ML keywords found and no integration context)
        if ml_keyword_count >= 2 and integration_keyword_count == 0 and original_category == "AUTOMATION":
            logger.info(f"=== CORRECTING TO ML CATEGORY ===")
            logger.info(f"Found {ml_keyword_count} specific ML keywords")

            # Determine specific ML subcategory
            if any(word in problem_lower for word in ["예측", "predict", "회귀", "regression", "수치", "농도", "concentration"]):
                corrected_category = "ML_REGRESSION"
            elif any(word in problem_lower for word in ["분류", "classify", "classification", "이상", "anomaly", "탐지", "detection"]):
                corrected_category = "ML_CLASSIFICATION"
            else:
                corrected_category = "MACHINE_LEARNING"

            # Update category and related fields
            corrected_json = parsed_json.copy()
            corrected_json['category'] = corrected_category

            # Update domain and title to match new category
            domain_mapping = {
                "MACHINE_LEARNING": "데이터 과학 및 예측 분석",
                "ML_REGRESSION": "머신러닝 회귀 예측 분석",
                "ML_CLASSIFICATION": "머신러닝 분류 및 이상 탐지"
            }

            title_mapping = {
                "MACHINE_LEARNING": "머신러닝 예측 모델 시스템",
                "ML_REGRESSION": "머신러닝 회귀 예측 모델",
                "ML_CLASSIFICATION": "머신러닝 분류 및 이상 탐지 모델"
            }

            corrected_json['domain'] = domain_mapping.get(corrected_category, corrected_json.get('domain', '데이터 분석'))
            corrected_json['title'] = title_mapping.get(corrected_category, corrected_json.get('title', 'ML 모델 시스템'))

            return corrected_json

        logger.info(f"=== NO CATEGORY CORRECTION NEEDED ===")
        return parsed_json

    def _extract_info_from_natural_language(self, problem_description: str, response: str) -> Dict[str, Any]:
        """자연어 응답에서 구조화된 정보 추출"""
        response_lower = response.lower()
        problem_lower = problem_description.lower()

        # 카테고리 결정 - 우선순위 적용
        category = "AUTOMATION"

        # INTEGRATION keywords (highest priority)
        integration_keywords = [
            "시스템 통합", "system integration", "연동", "api", "통합", "integrate", "connect",
            "sync", "synchronization", "동기화", "crm", "erp", "warehouse", "order management",
            "주문 관리", "창고 관리", "재고", "inventory", "dashboard", "대시보드", "reporting",
            "보고서", "자동화된 보고", "automated reporting"
        ]

        # Specific ML keywords (excluding generic terms)
        specific_ml_keywords = [
            # Environmental/Industrial specific
            "폐수", "수질", "오염", "환경", "농도", "수치", "측정", "품질",
            "wastewater", "water", "quality", "pollution", "environmental", "concentration",
            "measurement", "anomaly detection", "outlier", "treatment", "처리",
            # ML specific terms
            "예측", "분류", "모델", "머신러닝", "기계학습", "알고리즘", "신경망", "회귀", "이상치", "원인", "파악",
            "predict", "classify", "model", "ml", "machine learning", "neural", "algorithm", "regression"
        ]

        # Data Visualization keywords
        viz_keywords = ["chart", "graph", "visualize", "plot",
                       "차트", "그래프", "시각화", "플롯"]

        # Information Retrieval keywords
        ir_keywords = ["search", "find", "document", "knowledge", "retrieve", "query",
                      "검색", "찾기", "문서", "지식", "조회", "쿼리"]

        # Apply priority-based categorization
        integration_keyword_count = sum(1 for keyword in integration_keywords if keyword in problem_lower)
        ml_keyword_count = sum(1 for keyword in specific_ml_keywords if keyword in problem_lower)

        # Priority 1: INTEGRATION
        if integration_keyword_count >= 2:
            category = "INTEGRATION"
        # Priority 2: ML categories (only if no integration context)
        elif ml_keyword_count >= 2 and integration_keyword_count == 0:
            if any(word in problem_lower for word in ["예측", "predict", "회귀", "regression", "수치", "농도", "concentration"]):
                category = "ML_REGRESSION"
            elif any(word in problem_lower for word in ["분류", "classify", "classification", "이상", "anomaly", "탐지", "detection"]):
                category = "ML_CLASSIFICATION"
            else:
                category = "MACHINE_LEARNING"
        # Priority 3: Other categories
        elif any(keyword in problem_lower for keyword in viz_keywords):
            category = "DATA_VISUALIZATION"
        elif any(keyword in problem_lower for keyword in ir_keywords):
            category = "INFORMATION_RETRIEVAL"

        # 복잡도 결정
        complexity = "MEDIUM"
        if len(problem_description.split()) > 100 or any(keyword in problem_lower for keyword in ["integration", "api", "database", "security"]):
            complexity = "HIGH"
        elif len(problem_description.split()) < 30:
            complexity = "LOW"

        # 카테고리별 도메인과 제목 설정
        domain_mapping = {
            "MACHINE_LEARNING": "데이터 과학 및 예측 분석",
            "ML_REGRESSION": "머신러닝 회귀 예측 분석",
            "ML_CLASSIFICATION": "머신러닝 분류 및 이상 탐지",
            "DATA_VISUALIZATION": "데이터 시각화 및 대시보드",
            "INFORMATION_RETRIEVAL": "정보 검색 및 지식 관리",
            "AUTOMATION": "비즈니스 프로세스 자동화"
        }

        title_mapping = {
            "MACHINE_LEARNING": "머신러닝 예측 모델 시스템",
            "ML_REGRESSION": "머신러닝 회귀 예측 모델",
            "ML_CLASSIFICATION": "머신러닝 분류 및 이상 탐지 모델",
            "DATA_VISUALIZATION": "데이터 시각화 대시보드",
            "INFORMATION_RETRIEVAL": "정보 검색 시스템",
            "AUTOMATION": "프로세스 자동화 솔루션"
        }

        # 기본 분석 반환
        return {
            "title": title_mapping.get(category, "비즈니스 솔루션"),
            "domain": domain_mapping.get(category, "비즈니스 프로세스 개선"),
            "category": category,
            "complexity": complexity,
            "stakeholders": ["최종 사용자", "관리자", "IT 팀"],
            "pain_points": [
                "수동 작업으로 인한 시간 소모",
                "인적 오류 발생 가능성",
                "비일관적인 프로세스"
            ],
            "current_state": "수동으로 처리되는 현재 프로세스",
            "desired_state": "자동화된 효율적인 프로세스",
            "constraints": [
                "Python 환경에서 구현 가능해야 함",
                "초보자가 유지보수 가능해야 함"
            ],
            "success_criteria": [
                "처리 시간 50% 단축",
                "오류율 90% 감소",
                "사용자 만족도 향상"
            ]
        }

    async def generate_requirements(
        self,
        problem_analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate requirements using the Requirements Generator agent"""
        system_prompt = self.templates.get_system_prompt(AgentType.REQUIREMENTS_GENERATOR)
        user_prompt = f"""
문제 분석 결과:
{json.dumps(problem_analysis, indent=2, ensure_ascii=False)}

다음 JSON 형식으로만 응답해주세요. 다른 설명 없이 오직 JSON만 반환하세요:

{{
    "functional_requirements": ["기능적 요구사항 목록"],
    "non_functional_requirements": ["비기능적 요구사항 목록"],
    "technical_requirements": ["기술적 요구사항 목록"],
    "integration_requirements": ["통합 요구사항 목록"],
    "security_requirements": ["보안 요구사항 목록"],
    "acceptance_criteria": ["승인 기준 목록"]
}}

다시 한번 강조: JSON 형식으로만 응답하고, 앞뒤 설명은 넣지 마세요.
"""

        try:
            response = await self.llm_manager.simple_completion(
                user_prompt,
                system_prompt,
                temperature=0.1
            )

            # JSON 추출 시도
            response_cleaned = response.strip()
            if response_cleaned.startswith('```json'):
                response_cleaned = response_cleaned[7:]
            elif response_cleaned.startswith('```'):
                response_cleaned = response_cleaned[3:]
            if response_cleaned.endswith('```'):
                response_cleaned = response_cleaned[:-3]
            response_cleaned = response_cleaned.strip()

            start_idx = response_cleaned.find('{')
            end_idx = response_cleaned.rfind('}')
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response_cleaned[start_idx:end_idx+1]
                return json.loads(json_str)
            else:
                raise json.JSONDecodeError("No JSON found", response, 0)

        except Exception as e:
            logger.error(f"Failed to generate requirements - NO FALLBACK: {e}")
            logger.error(f"Raw response: {response[:300]}...")

            # 바로 예외를 발생시켜 문제를 즉시 알림
            raise ValueError(f"Requirements generation failed: {e}. Response: {response[:200]}")

    def _generate_requirements_from_analysis(self, problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """문제 분석을 기반으로 동적 요구사항 생성"""
        category = problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
        complexity = problem_analysis.get("complexity", "MEDIUM")
        domain = problem_analysis.get("domain", "비즈니스 프로세스")

        # 카테고리별 맞춤 요구사항
        functional_requirements = []
        if category == "AUTOMATION":
            functional_requirements = [
                "사용자가 작업을 자동으로 실행할 수 있어야 함",
                "스케줄링 기능을 통해 정기적 실행이 가능해야 함",
                "작업 진행 상황을 모니터링할 수 있어야 함",
                "에러 발생 시 알림 및 로깅 기능이 있어야 함"
            ]
        elif category == "DATA_VISUALIZATION":
            functional_requirements = [
                "데이터를 차트 및 그래프로 시각화할 수 있어야 함",
                "대화형 대시보드를 제공해야 함",
                "다양한 파일 형식으로 결과를 내보낼 수 있어야 함",
                "실시간 데이터 업데이트가 가능해야 함"
            ]
        else:
            functional_requirements = [
                f"{domain} 관련 핵심 기능을 제공해야 함",
                "사용자 친화적인 인터페이스를 제공해야 함",
                "데이터 입력 및 처리 기능이 있어야 함",
                "결과 출력 및 저장 기능이 있어야 함"
            ]

        # 복잡도별 비기능적 요구사항
        non_functional_requirements = [
            "Python 3.8 이상에서 동작해야 함",
            "초보자도 쉽게 사용할 수 있어야 함"
        ]
        if complexity == "HIGH":
            non_functional_requirements.extend([
                "동시 처리 사용자 수: 100명 이상",
                "응답 시간: 3초 이내",
                "99.9% 가용성을 보장해야 함"
            ])
        elif complexity == "MEDIUM":
            non_functional_requirements.extend([
                "동시 처리 사용자 수: 10-50명",
                "응답 시간: 5초 이내"
            ])
        else:
            non_functional_requirements.extend([
                "단일 사용자 기준으로 설계",
                "응답 시간: 10초 이내"
            ])

        return {
            "functional_requirements": functional_requirements,
            "non_functional_requirements": non_functional_requirements,
            "technical_requirements": [
                "Python 표준 라이브러리 우선 사용",
                "외부 의존성 최소화",
                "코드 가독성 및 유지보수성 확보",
                "에러 처리 및 예외 상황 대응"
            ],
            "integration_requirements": [
                "기존 시스템과의 호환성 확보",
                "표준 파일 형식 지원 (CSV, JSON, Excel)",
                "로그 시스템 연동 가능"
            ],
            "security_requirements": [
                "사용자 입력 데이터 유효성 검증",
                "중요 데이터 암호화 저장",
                "접근 권한 제어"
            ],
            "acceptance_criteria": [
                "모든 핵심 기능이 정상 동작함",
                "사용자 매뉴얼 제공됨",
                "단위 테스트 커버리지 80% 이상",
                f"{domain} 전문가의 검토 완료"
            ]
        }

    async def design_solution(
        self,
        requirements_text: str,
        problem_analysis: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Design solution architecture using LLM"""
        try:
            system_prompt = self.templates.get_system_prompt(AgentType.SOLUTION_DESIGNER)

            prompt = f"""
문제 분석 결과:
{json.dumps(problem_analysis, indent=2, ensure_ascii=False)}

요구사항:
{requirements_text}

다음 JSON 형식으로만 솔루션을 설계해주세요:

{{
    "solution_type": "AUTOMATION",
    "architecture": {{
        "components": ["주요 구성요소 목록"],
        "data_flow": "데이터 흐름 설명",
        "integration_points": ["통합 지점 목록"]
    }},
    "technology_stack": {{
        "primary_language": "Python",
        "frameworks": ["추천 프레임워크"],
        "libraries": ["필요한 라이브러리"],
        "tools": ["개발 도구"]
    }},
    "implementation_approach": "구현 접근 방식",
    "estimated_complexity": "예상 복잡도"
}}

다시 한번 강조: JSON 형식으로만 응답하고, 앞뒤 설명은 넣지 마세요.
"""

            response = await self.llm_manager.simple_completion(prompt, system_prompt, temperature=0.1)

            try:
                # JSON 추출 시도
                response_cleaned = response.strip()
                if response_cleaned.startswith('```json'):
                    response_cleaned = response_cleaned[7:]
                elif response_cleaned.startswith('```'):
                    response_cleaned = response_cleaned[3:]
                if response_cleaned.endswith('```'):
                    response_cleaned = response_cleaned[:-3]
                response_cleaned = response_cleaned.strip()

                start_idx = response_cleaned.find('{')
                end_idx = response_cleaned.rfind('}')
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    json_str = response_cleaned[start_idx:end_idx+1]
                    return json.loads(json_str)
                else:
                    raise json.JSONDecodeError("No JSON found", response, 0)

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse solution design as JSON - NO FALLBACK: {e}")
                logger.error(f"Raw response: {response[:300]}...")
                # 바로 예외를 발생시켜 문제를 즉시 알림
                raise ValueError(f"Solution design JSON parsing failed: {e}. Response: {response[:200]}")

        except Exception as e:
            logger.error(f"Failed to design solution - NO FALLBACK: {str(e)}")
            # 바로 예외를 발생시켜 문제를 즉시 알림
            raise ValueError(f"Solution design failed: {e}")

    def _generate_solution_from_analysis(self, problem_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """문제 분석을 기반으로 동적 솔루션 생성"""
        category = problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
        complexity = problem_analysis.get("complexity", "MEDIUM")

        # 카테고리별 맞춤 기술 스택
        if category == "AUTOMATION":
            tech_stack = {
                "primary_language": "Python",
                "frameworks": ["pandas", "pathlib"],
                "libraries": ["schedule", "logging", "os"],
                "tools": ["cron", "Task Scheduler"]
            }
            components = ["main_automation_script.py", "config.json", "log_handler.py"]
        elif category == "DATA_VISUALIZATION":
            tech_stack = {
                "primary_language": "Python",
                "frameworks": ["matplotlib", "plotly"],
                "libraries": ["pandas", "seaborn", "dash"],
                "tools": ["Jupyter Notebook"]
            }
            components = ["data_processor.py", "visualization.py", "dashboard.py"]
        elif category == "INFORMATION_RETRIEVAL":
            tech_stack = {
                "primary_language": "Python",
                "frameworks": ["elasticsearch", "whoosh"],
                "libraries": ["requests", "beautifulsoup4", "nltk"],
                "tools": ["Elasticsearch", "SQLite"]
            }
            components = ["search_engine.py", "indexer.py", "data_crawler.py"]
        else:
            tech_stack = {
                "primary_language": "Python",
                "frameworks": ["pandas"],
                "libraries": ["requests", "json", "csv"],
                "tools": []
            }
            components = ["main_script.py", "utils.py", "config.py"]

        # 복잡도별 아키텍처 조정
        if complexity == "HIGH":
            components.extend(["database.py", "api_handler.py", "tests/"])
            data_flow = "입력 검증 -> 데이터 처리 -> 비즈니스 로직 -> 결과 저장 -> 출력"
        elif complexity == "MEDIUM":
            components.extend(["helpers.py", "tests.py"])
            data_flow = "입력 -> 데이터 처리 -> 비즈니스 로직 -> 출력"
        else:
            data_flow = "입력 -> 처리 -> 출력"

        return {
            "solution_type": category,
            "architecture": {
                "components": components,
                "data_flow": data_flow,
                "integration_points": ["파일 시스템", "사용자 인터페이스", "로그 시스템"]
            },
            "technology_stack": tech_stack,
            "implementation_approach": f"{complexity.lower()} 복잡도 수준의 단계별 구현",
            "estimated_complexity": complexity
        }

    async def create_implementation_guide(
        self,
        problem_analysis: Dict[str, Any],
        requirements_text: str,
        solution_design: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Create implementation guide using the Guide Creator agent"""
        system_prompt = self.templates.get_system_prompt(AgentType.GUIDE_CREATOR)

        prompt = f"""
문제 분석:
{json.dumps(problem_analysis, indent=2, ensure_ascii=False)}

요구사항:
{requirements_text}

솔루션 설계:
{json.dumps(solution_design, indent=2, ensure_ascii=False)}

위 정보를 바탕으로 개발자가 바로 구현할 수 있는 상세한 구현 가이드를 마크다운 형식으로 작성해주세요.
다음 구조를 포함해야 합니다:

1. 프로젝트 개요
2. 개발 환경 설정
3. 단계별 구현 계획
4. 주요 함수/클래스 구현
5. 테스트 방법
6. 배포 및 운영 가이드

한글로 작성하되, 코드 예제는 실제 동작하는 Python 코드로 제공해주세요.
"""

        try:
            response = await self.llm_manager.simple_completion(
                prompt,
                system_prompt,
                temperature=0.2
            )

            # 가이드가 너무 짧으면 기본 구조 추가
            if len(response) < 1000:
                return self._generate_implementation_guide_from_analysis(
                    problem_analysis, solution_design
                )

            return response

        except Exception as e:
            logger.error(f"Failed to create implementation guide - NO FALLBACK: {e}")
            # 바로 예외를 발생시켜 문제를 즉시 알림
            raise ValueError(f"Implementation guide creation failed: {e}")

    def _generate_implementation_guide_from_analysis(
        self,
        problem_analysis: Dict[str, Any],
        solution_design: Dict[str, Any]
    ) -> str:
        """문제 분석과 솔루션 설계를 기반으로 구현 가이드 생성"""
        category = problem_analysis.get("category", "GENERAL_PROBLEM_SOLVING")
        complexity = problem_analysis.get("complexity", "MEDIUM")
        tech_stack = solution_design.get("technology_stack", {})
        components = solution_design.get("architecture", {}).get("components", [])

        guide = f"""# {category.replace('_', ' ').title()} 구현 가이드

## 📋 프로젝트 개요
- **문제 유형**: {category}
- **복잡도**: {complexity}
- **주요 언어**: {tech_stack.get('primary_language', 'Python')}
- **예상 개발 기간**: {self._estimate_development_time(complexity)}

## 🛠 개발 환경 설정

### 필수 요구사항
- Python 3.8 이상
- pip 패키지 관리자

### 라이브러리 설치
```bash
pip install {' '.join(tech_stack.get('libraries', ['pandas', 'requests']))}
```

## 📁 프로젝트 구조
```
project/
├── {'/'.join(components[:3] if len(components) > 3 else components)}
{'├── ' + '/'.join(components[3:]) if len(components) > 3 else ''}
├── requirements.txt
├── README.md
└── tests/
    └── test_main.py
```

## 🚀 단계별 구현 계획

### 1단계: 기본 구조 설정
```python
# {components[0] if components else 'main.py'}
import logging
import sys
from pathlib import Path

def setup_logging():
    \"\"\"로깅 설정\"\"\"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def main():
    \"\"\"메인 실행 함수\"\"\"
    setup_logging()
    logging.info("프로그램 시작")

    try:
        # 메인 로직 구현
        pass
    except Exception as e:
        logging.error(f"오류 발생: {{e}}")
        return False

    logging.info("프로그램 완료")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### 2단계: 핵심 기능 구현
{self._generate_core_implementation(category, tech_stack)}

### 3단계: 에러 처리 및 로깅
```python
def handle_error(error: Exception, context: str = ""):
    \"\"\"통합 에러 처리\"\"\"
    logging.error(f"{{context}} 오류: {{str(error)}}")
    # 필요시 에러 상세 정보 저장
    return None
```

## 🧪 테스트 방법

### 단위 테스트
```python
# tests/test_main.py
import unittest
from main import main

class TestMain(unittest.TestCase):
    def test_main_execution(self):
        \"\"\"메인 함수 실행 테스트\"\"\"
        result = main()
        self.assertTrue(result)

if __name__ == "__main__":
    unittest.main()
```

### 실행 방법
```bash
python -m unittest tests/test_main.py -v
```

## 📦 배포 및 운영

### requirements.txt 생성
```txt
{chr(10).join(tech_stack.get('libraries', ['pandas', 'requests']))}
```

### 실행 스크립트
```bash
#!/bin/bash
cd "$(dirname "$0")"
python main.py
```

## 🔧 유지보수 가이드

1. **로그 확인**: 실행 로그를 정기적으로 검토
2. **성능 모니터링**: 처리 시간 및 메모리 사용량 확인
3. **에러 대응**: 로그에 기록된 에러 패턴 분석
4. **백업**: 중요 데이터 정기 백업

## 📚 추가 리소스
- Python 공식 문서: https://docs.python.org/ko/3/
- {category} 관련 모범 사례 검색 권장

🤖 **AI 생성 가이드**: 실제 요구사항에 맞게 수정해서 사용하세요.
"""
        return guide

    def _estimate_development_time(self, complexity: str) -> str:
        """복잡도에 따른 개발 시간 추정"""
        time_estimates = {
            "LOW": "1-2주",
            "MEDIUM": "3-4주",
            "HIGH": "6-8주"
        }
        return time_estimates.get(complexity, "3-4주")

    def _generate_core_implementation(self, category: str, tech_stack: Dict[str, Any]) -> str:
        """카테고리별 핵심 구현 예제 생성"""
        if category == "AUTOMATION":
            return """```python
def process_automation():
    \"\"\"자동화 프로세스 실행\"\"\"
    import schedule
    import time

    def job():
        print("자동화 작업 실행 중...")
        # 실제 작업 로직 구현
        return True

    # 스케줄 설정 (매일 오전 9시)
    schedule.every().day.at("09:00").do(job)

    while True:
        schedule.run_pending()
        time.sleep(60)
```"""
        elif category == "DATA_VISUALIZATION":
            return """```python
import matplotlib.pyplot as plt
import pandas as pd

def create_visualization(data):
    \"\"\"데이터 시각화 생성\"\"\"
    plt.figure(figsize=(10, 6))
    plt.plot(data['x'], data['y'])
    plt.title('데이터 시각화')
    plt.xlabel('X축')
    plt.ylabel('Y축')
    plt.save('output.png')
    plt.show()
```"""
        else:
            return """```python
def core_function():
    \"\"\"핵심 기능 구현\"\"\"
    # 데이터 입력
    input_data = get_input_data()

    # 데이터 처리
    processed_data = process_data(input_data)

    # 결과 출력
    save_result(processed_data)
    return processed_data
```"""

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