"""
애플리케이션 설정

비개발자 요약:
- `.env` 값을 읽어 서버 포트, LLM 종류(internal/ollama), RAG API 등을 설정합니다.
- 설정 변경은 서버 재시작 후 적용됩니다.
"""

import os
from pathlib import Path
from typing import Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(env_path)

class Settings:
    """Application settings loaded from environment variables"""
    
    # Server Configuration
    SERVER_HOST: str = os.getenv("SERVER_HOST", "0.0.0.0")
    SERVER_PORT: int = int(os.getenv("SERVER_PORT", "8080"))
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info")
    RELOAD_MODE: bool = os.getenv("RELOAD_MODE", "true").lower() == "true"
    
    # Database Configuration
    DATABASE_PATH: str = os.getenv("DATABASE_PATH", "workflow_checkpoints.db")
    ENABLE_WAL_MODE: bool = os.getenv("ENABLE_WAL_MODE", "true").lower() == "true"
    DATABASE_TIMEOUT: float = float(os.getenv("DATABASE_TIMEOUT", "30"))
    
    # LLM Configuration
    LLM_SERVICE_TYPE: str = os.getenv("LLM_SERVICE_TYPE", "internal")
    
    # Internal LLM API Configuration (OpenAI 호환)
    INTERNAL_LLM_API_URL: str = os.getenv("INTERNAL_LLM_API_URL", "http://localhost:11434/v1")
    INTERNAL_LLM_API_KEY: Optional[str] = os.getenv("INTERNAL_LLM_API_KEY")
    INTERNAL_LLM_MODEL: str = os.getenv("INTERNAL_LLM_MODEL", "llama3.2:latest")
    # Internal LLM Headers (dotenv)
    INTERNAL_LLM_TICKET: str = os.getenv("INTERNAL_LLM_TICKET", "")
    INTERNAL_LLM_SYSTEM_NAME: str = os.getenv("INTERNAL_LLM_SYSTEM_NAME", "ATM-System")
    INTERNAL_LLM_USER_ID: str = os.getenv("INTERNAL_LLM_USER_ID", os.getenv("USER_ID", "system-user"))
    INTERNAL_LLM_USER_TYPE: str = os.getenv("INTERNAL_LLM_USER_TYPE", "AD")
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
    
    # RAG Service Configuration
    RAG_SERVICE_ENABLED: bool = os.getenv("RAG_SERVICE_ENABLED", "true").lower() == "true"
    RAG_PORTAL_URL: str = os.getenv("RAG_PORTAL_URL", "http://localhost:8080/api/rag")
    RAG_API_KEY: Optional[str] = os.getenv("RAG_API_KEY")
    # Company RAG API direct call
    RAG_API_URL: str = os.getenv("RAG_API_URL", "")
    RAG_INDEX_NAME: str = os.getenv("RAG_INDEX_NAME", "")
    RAG_PERMISSION_GROUP: List[str] = [p.strip() for p in os.getenv("RAG_PERMISSION_GROUP", "ds").split(",") if p.strip()]
    RAG_TIMEOUT: float = float(os.getenv("RAG_TIMEOUT", "20"))
    
    # Workflow Configuration
    ENABLE_HUMAN_LOOP: bool = os.getenv("ENABLE_HUMAN_LOOP", "true").lower() == "true"
    USE_PERSISTENT_STORAGE: bool = os.getenv("USE_PERSISTENT_STORAGE", "true").lower() == "true"
    MAX_RETRY_COUNT: int = int(os.getenv("MAX_RETRY_COUNT", "3"))
    WORKFLOW_TIMEOUT: int = int(os.getenv("WORKFLOW_TIMEOUT", "1800"))  # 30 minutes
    
    # CORS Configuration
    CORS_ALLOWED_ORIGINS: List[str] = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
    CORS_ALLOW_CREDENTIALS: bool = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
    
    # Session Management
    SESSION_CLEANUP_DAYS: int = int(os.getenv("SESSION_CLEANUP_DAYS", "30"))
    AUTO_CLEANUP_ENABLED: bool = os.getenv("AUTO_CLEANUP_ENABLED", "true").lower() == "true"
    
    # Development Configuration
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", "false").lower() == "true"
    ENABLE_DETAILED_LOGGING: bool = os.getenv("ENABLE_DETAILED_LOGGING", "false").lower() == "true"
    
    # Performance Configuration
    MAX_CONCURRENT_WORKFLOWS: int = int(os.getenv("MAX_CONCURRENT_WORKFLOWS", "10"))
    STREAM_BUFFER_SIZE: int = int(os.getenv("STREAM_BUFFER_SIZE", "1024"))
    
    # Frontend Configuration
    FRONTEND_URL: str = os.getenv("FRONTEND_URL", "http://localhost:8080")
    API_PREFIX: str = os.getenv("API_PREFIX", "/api/v1")
    
    # File Paths
    FRONTEND_PATH: str = os.getenv("FRONTEND_PATH", "../frontend")
    STATIC_FILES_PATH: str = os.getenv("STATIC_FILES_PATH", "../frontend")
    
    # Backup Configuration
    BACKUP_ENABLED: bool = os.getenv("BACKUP_ENABLED", "false").lower() == "true"
    BACKUP_INTERVAL_HOURS: int = int(os.getenv("BACKUP_INTERVAL_HOURS", "24"))
    BACKUP_RETENTION_DAYS: int = int(os.getenv("BACKUP_RETENTION_DAYS", "7"))
    
    @classmethod
    def get_database_path(cls) -> str:
        """Get the full database path"""
        db_path = Path(cls.DATABASE_PATH)
        if not db_path.is_absolute():
            # Make relative to backend directory
            backend_dir = Path(__file__).parent.parent.parent
            db_path = backend_dir / db_path
        return str(db_path)
    
    @classmethod
    def get_frontend_path(cls) -> Path:
        """Get the full frontend path"""
        frontend_path = Path(cls.FRONTEND_PATH)
        if not frontend_path.is_absolute():
            # Make relative to backend directory
            backend_dir = Path(__file__).parent.parent.parent
            frontend_path = backend_dir / frontend_path
        return frontend_path
    
    @classmethod
    def is_development(cls) -> bool:
        """Check if running in development mode"""
        return cls.DEBUG_MODE or cls.RELOAD_MODE
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """Get LLM configuration based on service type"""
        if cls.LLM_SERVICE_TYPE == "ollama":
            return {
                "service_type": "ollama",
                "base_url": cls.OLLAMA_BASE_URL,
                "model": cls.OLLAMA_MODEL
            }
        else:  # internal
            return {
                "service_type": "internal",
                "api_url": cls.INTERNAL_LLM_API_URL,
                "api_key": cls.INTERNAL_LLM_API_KEY,
                "model": cls.INTERNAL_LLM_MODEL
            }
    
    @classmethod
    def validate_config(cls) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        warnings = []
        
        # Check critical configurations
        # 서비스 타입 유효성
        if cls.LLM_SERVICE_TYPE not in ("internal", "ollama"):
            warnings.append(f"Unsupported LLM_SERVICE_TYPE: {cls.LLM_SERVICE_TYPE} (use internal/ollama)")
        # internal 설정 점검
        if cls.LLM_SERVICE_TYPE == "internal" and not cls.INTERNAL_LLM_API_URL:
            warnings.append("INTERNAL_LLM_API_URL is not set while using internal LLM")

        if cls.RAG_SERVICE_ENABLED and not cls.RAG_API_KEY:
            warnings.append("RAG_API_KEY is not set but RAG service is enabled")
        # 사내 RAG 직접 호출 설정 점검(선택)
        if cls.RAG_API_URL and not cls.RAG_INDEX_NAME:
            warnings.append("RAG_API_URL is set but RAG_INDEX_NAME is empty")
        
        if cls.SERVER_PORT < 1024 and os.name != 'nt':  # Unix systems
            warnings.append(f"Server port {cls.SERVER_PORT} may require root privileges on Unix systems")
        
        if cls.WORKFLOW_TIMEOUT < 60:
            warnings.append("WORKFLOW_TIMEOUT is very low, workflows may timeout prematurely")
        
        if cls.MAX_CONCURRENT_WORKFLOWS > 50:
            warnings.append("MAX_CONCURRENT_WORKFLOWS is very high, may cause resource issues")
        
        return warnings
    
    @classmethod
    def print_config_summary(cls):
        """Print a summary of the current configuration"""
        print("=" * 60)
        print("AI Problem Solving Copilot - Configuration Summary")
        print("=" * 60)
        print(f"Server: {cls.SERVER_HOST}:{cls.SERVER_PORT}")
        print(f"Database: {cls.get_database_path()}")
        print(f"LLM Service: {cls.LLM_SERVICE_TYPE}")
        print(f"Frontend: {cls.get_frontend_path()}")
        print(f"Debug Mode: {cls.DEBUG_MODE}")
        print(f"Human Loop: {cls.ENABLE_HUMAN_LOOP}")
        print(f"RAG Enabled: {cls.RAG_SERVICE_ENABLED}")
        
        warnings = cls.validate_config()
        if warnings:
            print("\nConfiguration Warnings:")
            for warning in warnings:
                print(f"  ⚠️  {warning}")
        
        print("=" * 60)


# Create global settings instance
settings = Settings()

# Print configuration on import if in development mode
if settings.is_development():
    settings.print_config_summary()
