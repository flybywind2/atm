"""
Simple test script to verify FastAPI backend implementation
"""

import asyncio
import httpx
import json
import time
from typing import Dict, Any


async def test_health_endpoint():
    """Test the health check endpoint"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/api/health")
            print(f"Health check: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"Health check failed: {e}")
            return False


async def test_api_root():
    """Test the API root endpoint"""
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get("http://localhost:8000/api/v1/")
            print(f"API root: {response.status_code}")
            print(f"Response: {response.json()}")
            return response.status_code == 200
        except Exception as e:
            print(f"API root test failed: {e}")
            return False


async def test_start_analysis():
    """Test the start analysis endpoint"""
    async with httpx.AsyncClient() as client:
        try:
            test_request = {
                "problem_description": "I need to automate a repetitive data entry task that takes me 2 hours every day. The task involves copying data from Excel files to a web form.",
                "user_context": {
                    "technical_level": "beginner",
                    "environment": "Windows 10",
                    "tools": ["Excel", "Chrome browser"]
                }
            }
            
            response = await client.post(
                "http://localhost:8000/api/v1/start-analysis",
                json=test_request,
                timeout=30.0
            )
            
            print(f"Start analysis: {response.status_code}")
            if response.status_code == 202:
                result = response.json()
                print(f"Response: {result}")
                return result.get("thread_id")
            else:
                print(f"Error response: {response.text}")
                return None
                
        except Exception as e:
            print(f"Start analysis test failed: {e}")
            return None


async def test_status_endpoint(thread_id: str):
    """Test the status endpoint"""
    async with httpx.AsyncClient() as client:
        try:
            # Poll status multiple times to see progress
            for i in range(5):
                response = await client.get(f"http://localhost:8000/api/v1/status/{thread_id}")
                print(f"Status check {i+1}: {response.status_code}")
                
                if response.status_code == 200:
                    result = response.json()
                    print(f"Status: {result.get('status')}")
                    print(f"Step: {result.get('current_step')}")
                    print(f"Progress: {result.get('progress_percentage')}%")
                    print(f"Message: {result.get('message')}")
                    
                    if result.get("requires_input"):
                        print("Workflow is waiting for user input!")
                        print(f"Questions: {result.get('questions')}")
                        return "awaiting_input"
                    
                    if result.get("status") == "completed":
                        print("Workflow completed!")
                        return "completed"
                    
                    if result.get("status") == "error":
                        print(f"Workflow failed: {result.get('message')}")
                        return "error"
                else:
                    print(f"Error response: {response.text}")
                
                # Wait before next poll
                await asyncio.sleep(2)
            
            return "polling_completed"
            
        except Exception as e:
            print(f"Status test failed: {e}")
            return "error"


async def test_resume_endpoint(thread_id: str):
    """Test the resume endpoint"""
    async with httpx.AsyncClient() as client:
        try:
            resume_request = {
                "user_input": "I'm using Windows 10 with Python 3.9 installed. I have access to Excel via Office 365 and usually work with Chrome browser. The web form is on our internal company portal.",
                "context_data": {
                    "os": "Windows 10",
                    "python_version": "3.9",
                    "office_version": "Office 365",
                    "browser": "Chrome",
                    "target_system": "Internal company portal"
                }
            }
            
            response = await client.post(
                f"http://localhost:8000/api/v1/resume/{thread_id}",
                json=resume_request,
                timeout=30.0
            )
            
            print(f"Resume workflow: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Response: {result}")
                return True
            else:
                print(f"Error response: {response.text}")
                return False
                
        except Exception as e:
            print(f"Resume test failed: {e}")
            return False


async def run_comprehensive_test():
    """Run a comprehensive test of all endpoints"""
    print("=== FastAPI Backend Comprehensive Test ===\n")
    
    # Test 1: Health check
    print("1. Testing health endpoint...")
    health_ok = await test_health_endpoint()
    print(f"Health check: {'✓ PASSED' if health_ok else '✗ FAILED'}\n")
    
    if not health_ok:
        print("Health check failed. Make sure the server is running on localhost:8000")
        return
    
    # Test 2: API root
    print("2. Testing API root endpoint...")
    root_ok = await test_api_root()
    print(f"API root: {'✓ PASSED' if root_ok else '✗ FAILED'}\n")
    
    # Test 3: Start analysis
    print("3. Testing start analysis endpoint...")
    thread_id = await test_start_analysis()
    
    if thread_id:
        print(f"Start analysis: ✓ PASSED (Thread ID: {thread_id})\n")
        
        # Test 4: Status polling
        print("4. Testing status endpoint (polling)...")
        status_result = await test_status_endpoint(thread_id)
        print(f"Status polling: ✓ COMPLETED ({status_result})\n")
        
        # Test 5: Resume workflow (if needed)
        if status_result == "awaiting_input":
            print("5. Testing resume endpoint...")
            resume_ok = await test_resume_endpoint(thread_id)
            print(f"Resume workflow: {'✓ PASSED' if resume_ok else '✗ FAILED'}\n")
            
            # Continue polling after resume
            print("6. Continue status polling after resume...")
            final_status = await test_status_endpoint(thread_id)
            print(f"Final status: {final_status}\n")
    else:
        print("Start analysis: ✗ FAILED\n")
    
    print("=== Test Complete ===")


if __name__ == "__main__":
    print("Make sure the FastAPI server is running:")
    print("cd /d/Python/atm/backend && python -m uvicorn app.main:app --reload")
    print("\nPress Enter to continue with testing...")
    input()
    
    # Run the test
    asyncio.run(run_comprehensive_test())