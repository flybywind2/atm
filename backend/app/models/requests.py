"""
API 요청 모델(Pydantic)

비개발자 요약:
- 프런트엔드가 서버로 보낼 때의 데이터 모양(필수/선택 필드와 길이 제한 등)을 정의합니다.
  - 분석 시작: 문제 설명(problem_description)과 선택 컨텍스트(user_context)
  - 재개: 사용자 답변(user_input)과 선택 컨텍스트(context_data)
"""

from pydantic import BaseModel, Field
from typing import Dict, Any, Optional


class AnalysisRequest(BaseModel):
    """
    Request model for starting a new problem analysis
    
    Attributes:
        problem_description: User's description of the problem to be analyzed
        user_context: Optional additional context about the user's environment
    """
    problem_description: str = Field(
        ..., 
        description="Detailed description of the problem to be analyzed",
        min_length=10,
        max_length=5000
    )
    user_context: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context about user's environment, constraints, etc."
    )


class ResumeRequest(BaseModel):
    """
    Request model for resuming a paused workflow with user input
    
    Attributes:
        user_input: User's response to agent questions or additional information
        context_data: Optional structured data from context collection
    """
    user_input: str = Field(
        ...,
        description="User's response to continue the workflow",
        min_length=1,
        max_length=2000
    )
    context_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Structured data collected during context collection phase"
    )


class ProblemSolvingRequest(BaseModel):
    """
    Request model for starting problem solving workflow
    
    Attributes:
        problem_description: Detailed description of the problem
        context_data: Optional additional context
    """
    problem_description: str = Field(
        ...,
        description="Detailed description of the problem to solve",
        min_length=10,
        max_length=5000
    )
    context_data: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional context about the problem or requirements"
    )
