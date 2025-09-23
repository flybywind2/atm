"""
AI-based Problem Solving Copilot System - FastAPI Application Entry Point

This module initializes the FastAPI application with CORS configuration,
error handling, and sets up the API routes for the problem-solving workflow.
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

# Import API routers
from app.api.analysis import analysis_router
from app.models.responses import ErrorResponse
from app.database.checkpointer import get_checkpointer_manager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    allow_origins=["*"],  # In production, specify exact frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for frontend (assuming frontend is in ../frontend)
frontend_path = Path(__file__).parent.parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/", StaticFiles(directory=str(frontend_path), html=True), name="frontend")

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


# Include API routers
app.include_router(analysis_router, prefix="/api/v1", tags=["analysis"])

if __name__ == "__main__":
    # Development server configuration
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )