"""
RAG 서비스 유틸 (사내 RAG API 연동)

비개발자 요약:
- 루트 appendix/rag_retrieve.py의 구현 의도를 반영하여, 백엔드에서 직접 사내 RAG API를 호출합니다.
- .env에 설정한 URL/자격/인덱스 등을 사용하며, 실패 시 안전 폴백을 반환합니다.
"""

import logging
from typing import Dict, List, Any, Optional
import os
import httpx
from app.config import settings

logger = logging.getLogger(__name__)


# 환경 변수/설정 로드
RAG_API_URL = os.getenv("RAG_API_URL", settings.RAG_PORTAL_URL or "http://localhost:8000/retrieve-rrf")
RAG_TICKET = os.getenv("RAG_TICKET", os.getenv("INTERNAL_LLM_TICKET", ""))
RAG_API_KEY = settings.RAG_API_KEY or os.getenv("RAG_API_KEY", "")
RAG_INDEX_NAME = os.getenv("RAG_INDEX_NAME", os.getenv("RAG_DEFAULT_INDEX", "your_index_name"))
RAG_PERMISSION_GROUP = [p.strip() for p in os.getenv("RAG_PERMISSION_GROUP", "ds").split(",") if p.strip()]
RAG_TIMEOUT = float(os.getenv("RAG_TIMEOUT", "20"))


def _build_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "x-dep-ticket": RAG_TICKET,
        "api-key": RAG_API_KEY,
    }
    if extra:
        headers.update(extra)
    return headers


def _build_payload(query: str, num_results: int = 5, fields_exclude: Optional[List[str]] = None,
                   filters: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    return {
        "index_name": RAG_INDEX_NAME,
        "permission_group": RAG_PERMISSION_GROUP,
        "query_text": query,
        "num_results_doc": num_results,
        "fields_exclude": fields_exclude or ["v_merge_title_content"],
        "filter": filters or {},
    }


async def _post_rag(query: str, num_results: int = 5,
                    fields_exclude: Optional[List[str]] = None,
                    filters: Optional[Dict[str, List[str]]] = None) -> Dict[str, Any]:
    payload = _build_payload(query, num_results, fields_exclude, filters)
    try:
        async with httpx.AsyncClient(timeout=RAG_TIMEOUT) as client:
            resp = await client.post(RAG_API_URL, headers=_build_headers(), json=payload)
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        logger.error(f"RAG API call failed: {e}")
        return {"error": str(e), "results": []}


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


def _normalize_results(raw: Any) -> List[Dict[str, Any]]:
    """서버 응답을 통일된 문서 리스트로 변환 (Elasticsearch hits 지원)

    지원 포맷 예시
    - Elasticsearch: {"hits": {"hits": [{"_index": ..., "_id": ..., "_score": ..., "_source": {...}, "highlight": {...}}]}}
    - 단순 리스트: [{ title, content, ... }]
    - 래퍼: { results: [...] } / { documents: [...] } / { data: [...] }
    """
    # 1) Elasticsearch 포맷 감지
    if isinstance(raw, dict) and isinstance(raw.get("hits"), dict) and isinstance(raw["hits"].get("hits"), list):
        hits = raw["hits"]["hits"]
        out: List[Dict[str, Any]] = []
        for h in hits:
            if not isinstance(h, dict):
                continue
            src = h.get("_source") or {}
            hl = h.get("highlight") or {}
            # 제목 후보
            title = (
                src.get("title") or src.get("doc_title") or src.get("name") or src.get("filename") or ""
            )
            # 내용 후보(하이라이트 우선)
            hl_content = None
            if isinstance(hl, dict) and hl:
                # 가장 첫 번째 하이라이트 조각 사용
                for k, v in hl.items():
                    if isinstance(v, list) and v:
                        hl_content = v[0]
                        break
            content = hl_content or src.get("content") or src.get("text") or src.get("body") or src.get("summary") or ""
            # 출처/식별자
            source = src.get("url") or src.get("uri") or src.get("path") or src.get("source") or ""
            score = h.get("_score")
            out.append({
                "title": title,
                "content": content,
                "source": source,
                "score": score,
                "index": h.get("_index"),
                "id": h.get("_id"),
                "raw": h,
            })
        return out

    # 2) 일반 리스트/래퍼 형식
    if isinstance(raw, dict):
        docs = raw.get("results") or raw.get("documents") or raw.get("data") or []
    elif isinstance(raw, list):
        docs = raw
    else:
        docs = []

    normalized: List[Dict[str, Any]] = []
    for d in docs:
        if isinstance(d, dict):
            title = d.get("title") or d.get("doc_title") or d.get("name") or ""
            content = (
                d.get("content") or d.get("text") or d.get("snippet") or d.get("summary") or ""
            )
            source = d.get("source") or d.get("uri") or d.get("path") or ""
            score = d.get("score") or d.get("similarity") or d.get("relevance") or None
            normalized.append({
                "title": title,
                "content": content,
                "source": source,
                "score": score,
                "raw": d,
            })
        else:
            normalized.append({"title": "", "content": str(d), "source": "", "score": None, "raw": d})
    return normalized


async def enhance_llm_context(
    agent_type: str,
    query: str,
    current_context: Dict[str, Any],
    domain: Optional[str] = None,
) -> Dict[str, Any]:
    """사내 RAG API를 사용해 컨텍스트를 보강합니다."""
    result = await _post_rag(query, num_results=5)
    docs = _normalize_results(result)
    enhanced = (current_context or {}).copy()
    enhanced.update({
        "rag_enhanced": True if docs else False,
        "retrieved_context": docs,
        "context_sources": [d.get("source") for d in docs if d.get("source")],
        "rag_error": result.get("error") if isinstance(result, dict) else None,
    })
    return enhanced


async def retrieve_solution_examples(
    problem_type: str,
    tech_stack: List[str],
    domain: Optional[str] = None
) -> List[Dict[str, Any]]:
    """사내 RAG에서 유사 솔루션/레퍼런스 예시를 검색합니다."""
    query = f"solution example {problem_type} {domain or ''} {' '.join(tech_stack or [])}"
    result = await _post_rag(query, num_results=5)
    return _normalize_results(result)


async def get_technology_recommendations(requirements: Dict[str, Any]) -> List[Dict[str, Any]]:
    """요구사항에 기반한 기술 추천을 RAG에서 검색합니다(룰+검색 혼합)."""
    domain = (requirements or {}).get("domain", "general")
    complexity = (requirements or {}).get("complexity", "medium")
    seeds = []
    if "web" in str(domain).lower():
        seeds.append("web framework python best practice")
    if "api" in str(domain).lower():
        seeds.append("python api framework comparison")
    seeds = seeds or [f"technology recommendations python {domain} {complexity}"]

    # 간단히 첫 번째 시드로 조회
    result = await _post_rag(seeds[0], num_results=5)
    return _normalize_results(result)

# Backward compatibility functions
def retrieve_context(query: str) -> str:
    """Mock RAG retrieval function for backward compatibility"""
    return f"Mock retrieved context for query: {query[:50]}..."

async def get_rag_context(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Elasticsearch 기반 RAG 컨텍스트 조회 (정규화 리스트 반환)"""
    result = await _post_rag(query, num_results=max_results)
    return _normalize_results(result)

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
