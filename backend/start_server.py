"""
FastAPI server startup script with enhanced configuration
"""

import uvicorn
import sys
import os
from pathlib import Path

# Add the backend directory to the Python path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

if __name__ == "__main__":
    # Development server configuration
    print("Starting AI Problem Solving Copilot FastAPI Server...")
    print("Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/api/docs")
    print("Alternative Documentation: http://localhost:8000/api/redoc")
    print("\nPress Ctrl+C to stop the server")
    
    try:
        uvicorn.run(
            "app.main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            reload_dirs=[str(backend_dir)],
            workers=1  # Single worker for development
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Server failed to start: {e}")
        sys.exit(1)