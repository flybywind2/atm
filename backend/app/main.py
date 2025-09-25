"""
AI 문제 해결 코파일럿 - FastAPI 진입점

비개발자 요약:
- 이 파일은 웹 서버의 시작점입니다. CORS(보안 설정), 오류 처리,
  API 라우터 연결(분석 시작/상태/재개), 정적 파일(프런트) 제공을 설정합니다.
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
import uvicorn
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

# Import configuration
from app.config import settings

# Import API routers
from app.api.analysis import analysis_router
from app.models.responses import ErrorResponse
from app.database.checkpointer import get_checkpointer_manager

# Configure logging
import sys
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan handler for startup and shutdown events
    """
    # Startup
    logger.info("Starting AI Problem Solving Copilot...")
    
    # Initialize database checkpointer
    try:
        checkpointer_manager = get_checkpointer_manager()
        checkpointer_manager.initialize_database()
        logger.info("Database checkpointer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        # Continue startup even if database fails (for development)
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Problem Solving Copilot...")
    
    # Cleanup database connections
    try:
        checkpointer_manager = get_checkpointer_manager()
        checkpointer_manager.close()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")


# Initialize FastAPI application
app = FastAPI(
    title="AI Problem Solving Copilot",
    description="AI-based problem solving copilot system with LangGraph workflow integration",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan
)

# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error="http_error",
            message=exc.detail,
            details={"status_code": exc.status_code}
        ).dict()
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors"""
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="validation_error",
            message="Request validation failed",
            details={"errors": exc.errors()}
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="internal_server_error",
            message="An internal error occurred",
            details={"error_type": type(exc).__name__}
        ).dict()
    )


# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers first (higher priority)
app.include_router(analysis_router, prefix="/api/v1", tags=["analysis"])

# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint for monitoring system status"""
    try:
        # Check database connectivity
        checkpointer_manager = get_checkpointer_manager()
        db_status = "healthy" if checkpointer_manager.connection else "disconnected"
        
        return {
            "status": "healthy",
            "service": "AI Problem Solving Copilot",
            "version": "1.0.0",
            "database": db_status,
            "components": {
                "api": "healthy",
                "workflow_engine": "healthy",
                "checkpointer": db_status
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "AI Problem Solving Copilot",
                "version": "1.0.0",
                "error": str(e)
            }
        )


# API root endpoint
@app.get("/api/v1/")
async def api_root():
    """API root endpoint with available routes information"""
    return {
        "message": "AI Problem Solving Copilot API",
        "version": "1.0.0",
        "description": "AI-based problem solving copilot with LangGraph workflow integration",
        "available_endpoints": {
            "health": "/api/health",
            "start_analysis": "POST /api/v1/start-analysis",
            "status": "GET /api/v1/status/{thread_id}",
            "resume": "POST /api/v1/resume/{thread_id}",
            "docs": "/api/docs",
            "redoc": "/api/redoc"
        },
        "features": [
            "Problem analysis workflow",
            "Human-in-the-loop interaction",
            "Document generation",
            "Solution recommendations",
            "Implementation guides"
        ]
    }


# Mount static files for frontend (after API routes)
frontend_path = settings.get_frontend_path()
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

if __name__ == "__main__":
    # Development server configuration
    uvicorn.run(
        "main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.RELOAD_MODE,
        log_level=settings.LOG_LEVEL
    )
