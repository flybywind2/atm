"""
Load testing and performance tests for the ATM system
"""

import pytest
import asyncio
import time
import statistics
from unittest.mock import patch, AsyncMock
import httpx
from typing import List, Dict, Any


class TestLoadTesting:
    """Load testing and performance benchmarking."""

    @pytest.mark.asyncio
    async def test_api_endpoint_performance(self, performance_config):
        """Test API endpoint performance under load."""
        base_url = "http://localhost:8000"
        
        async def make_health_request():
            async with httpx.AsyncClient() as client:
                start_time = time.time()
                try:
                    response = await client.get(f"{base_url}/api/health", timeout=5.0)
                    end_time = time.time()
                    return {
                        "success": response.status_code == 200,
                        "response_time": end_time - start_time,
                        "status_code": response.status_code
                    }
                except Exception as e:
                    end_time = time.time()
                    return {
                        "success": False,
                        "response_time": end_time - start_time,
                        "error": str(e)
                    }
        
        # Execute concurrent requests
        start_time = time.time()
        tasks = [make_health_request() for _ in range(performance_config["concurrent_users"])]
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = [r for r in results if r["success"]]
        response_times = [r["response_time"] for r in successful_requests]
        
        success_rate = len(successful_requests) / len(results)
        avg_response_time = statistics.mean(response_times) if response_times else float('inf')
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) > 1 else avg_response_time
        
        # Performance assertions
        assert success_rate >= performance_config["success_rate_threshold"]
        assert avg_response_time < performance_config["max_response_time"]
        assert p95_response_time < performance_config["max_response_time"] * 2
        
        print(f"Performance Results:")
        print(f"  Total requests: {len(results)}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average response time: {avg_response_time:.3f}s")
        print(f"  95th percentile: {p95_response_time:.3f}s")
        print(f"  Total time: {total_time:.3f}s")

    @pytest.mark.asyncio
    async def test_workflow_performance_under_load(self, performance_config, mock_llm_service):
        """Test workflow performance under concurrent load."""
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            # Mock fast LLM responses
            mock_llm_service.generate_response.return_value = {
                "content": '{"problem_type": "automation", "complexity": "low", "solution_category": "SIMPLE_AUTOMATION"}',
                "usage": {"tokens": 100}
            }
            
            base_url = "http://localhost:8000"
            
            async def start_workflow_request(user_id: int):
                async with httpx.AsyncClient() as client:
                    start_time = time.time()
                    try:
                        request_data = {
                            "problem_description": f"Automation task {user_id}",
                            "user_context": {
                                "technical_level": "beginner",
                                "user_id": f"load_test_user_{user_id}"
                            }
                        }
                        
                        response = await client.post(
                            f"{base_url}/api/v1/start-analysis",
                            json=request_data,
                            timeout=10.0
                        )
                        end_time = time.time()
                        
                        return {
                            "success": response.status_code == 202,
                            "response_time": end_time - start_time,
                            "thread_id": response.json().get("thread_id") if response.status_code == 202 else None,
                            "user_id": user_id
                        }
                    except Exception as e:
                        end_time = time.time()
                        return {
                            "success": False,
                            "response_time": end_time - start_time,
                            "error": str(e),
                            "user_id": user_id
                        }
            
            # Execute concurrent workflow starts
            concurrent_users = min(performance_config["concurrent_users"], 10)  # Limit for workflow tests
            tasks = [start_workflow_request(i) for i in range(concurrent_users)]
            results = await asyncio.gather(*tasks)
            
            # Analyze workflow start performance
            successful_starts = [r for r in results if r["success"]]
            start_times = [r["response_time"] for r in successful_starts]
            
            success_rate = len(successful_starts) / len(results)
            avg_start_time = statistics.mean(start_times) if start_times else float('inf')
            
            assert success_rate >= 0.8  # 80% success rate for concurrent workflow starts
            assert avg_start_time < 5.0  # Workflow start should be under 5 seconds
            
            print(f"Workflow Start Performance:")
            print(f"  Concurrent users: {concurrent_users}")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Average start time: {avg_start_time:.3f}s")

    @pytest.mark.asyncio
    async def test_database_performance_under_load(self, temp_db_path, sample_workflow_state):
        """Test database performance under concurrent access."""
        from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
        
        # Create checkpointer
        checkpointer = await create_checkpointer(temp_db_path)
        
        async def save_checkpoint_task(thread_id: str, state: Dict[str, Any]):
            start_time = time.time()
            try:
                success = await save_checkpoint(checkpointer, thread_id, state)
                end_time = time.time()
                return {
                    "success": success,
                    "response_time": end_time - start_time,
                    "thread_id": thread_id
                }
            except Exception as e:
                end_time = time.time()
                return {
                    "success": False,
                    "response_time": end_time - start_time,
                    "error": str(e),
                    "thread_id": thread_id
                }
        
        # Execute concurrent database operations
        num_operations = 50
        tasks = []
        for i in range(num_operations):
            thread_id = f"perf_test_thread_{i}"
            state = {**sample_workflow_state, "thread_id": thread_id, "iteration": i}
            task = save_checkpoint_task(thread_id, state)
            tasks.append(task)
        
        start_time = time.time()
        results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time
        
        # Analyze database performance
        successful_ops = [r for r in results if r["success"]]
        db_times = [r["response_time"] for r in successful_ops]
        
        success_rate = len(successful_ops) / len(results)
        avg_db_time = statistics.mean(db_times) if db_times else float('inf')
        throughput = num_operations / total_time
        
        assert success_rate >= 0.9  # 90% success rate for database operations
        assert avg_db_time < 0.5  # Database operations should be under 500ms
        assert throughput >= 10  # At least 10 operations per second
        
        print(f"Database Performance:")
        print(f"  Operations: {num_operations}")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Average operation time: {avg_db_time:.3f}s")
        print(f"  Throughput: {throughput:.1f} ops/sec")

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, temp_db_path, sample_workflow_state):
        """Test memory usage during high load scenarios."""
        import psutil
        import os
        
        from backend.app.database.checkpointer import create_checkpointer, save_checkpoint
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        # Create many large workflow states
        async def create_large_state_task(i: int):
            large_state = {
                **sample_workflow_state,
                "large_data": ["x" * 1000] * 100,  # 100KB of data
                "thread_id": f"memory_test_{i}",
                "conversation_history": [
                    {"role": "user", "content": f"Message {j}" * 100}
                    for j in range(50)
                ]
            }
            return await save_checkpoint(checkpointer, f"memory_test_{i}", large_state)
        
        # Execute memory-intensive operations
        num_operations = 20
        tasks = [create_large_state_task(i) for i in range(num_operations)]
        results = await asyncio.gather(*tasks)
        
        # Check memory usage after operations
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable
        assert memory_increase < 500  # Should not use more than 500MB additional
        
        # Most operations should succeed
        success_rate = sum(1 for r in results if r) / len(results)
        assert success_rate >= 0.8
        
        print(f"Memory Usage:")
        print(f"  Initial memory: {initial_memory:.1f}MB")
        print(f"  Final memory: {final_memory:.1f}MB")
        print(f"  Memory increase: {memory_increase:.1f}MB")
        print(f"  Success rate: {success_rate:.2%}")

    @pytest.mark.asyncio
    async def test_sustained_load_performance(self, performance_config):
        """Test performance under sustained load over time."""
        base_url = "http://localhost:8000"
        test_duration = min(performance_config["test_duration"], 30)  # Max 30 seconds for tests
        
        async def sustained_request_worker():
            request_times = []
            start_time = time.time()
            
            async with httpx.AsyncClient() as client:
                while time.time() - start_time < test_duration:
                    req_start = time.time()
                    try:
                        response = await client.get(f"{base_url}/api/health", timeout=2.0)
                        req_end = time.time()
                        
                        request_times.append({
                            "timestamp": req_end,
                            "response_time": req_end - req_start,
                            "success": response.status_code == 200
                        })
                    except Exception:
                        req_end = time.time()
                        request_times.append({
                            "timestamp": req_end,
                            "response_time": req_end - req_start,
                            "success": False
                        })
                    
                    await asyncio.sleep(0.1)  # 10 requests per second per worker
            
            return request_times
        
        # Run multiple workers for sustained load
        num_workers = 3
        tasks = [sustained_request_worker() for _ in range(num_workers)]
        all_results = await asyncio.gather(*tasks)
        
        # Flatten results
        all_requests = []
        for worker_results in all_results:
            all_requests.extend(worker_results)
        
        # Analyze sustained performance
        successful_requests = [r for r in all_requests if r["success"]]
        total_requests = len(all_requests)
        success_rate = len(successful_requests) / total_requests if total_requests > 0 else 0
        
        # Calculate performance over time
        if successful_requests:
            response_times = [r["response_time"] for r in successful_requests]
            avg_response_time = statistics.mean(response_times)
            
            # Check for performance degradation over time
            first_half = successful_requests[:len(successful_requests)//2]
            second_half = successful_requests[len(successful_requests)//2:]
            
            first_half_avg = statistics.mean([r["response_time"] for r in first_half]) if first_half else 0
            second_half_avg = statistics.mean([r["response_time"] for r in second_half]) if second_half else 0
            
            degradation = (second_half_avg - first_half_avg) / first_half_avg if first_half_avg > 0 else 0
            
            assert success_rate >= 0.9  # 90% success rate under sustained load
            assert avg_response_time < 1.0  # Average response time under 1 second
            assert degradation < 0.5  # Performance shouldn't degrade more than 50%
            
            print(f"Sustained Load Performance:")
            print(f"  Test duration: {test_duration}s")
            print(f"  Total requests: {total_requests}")
            print(f"  Success rate: {success_rate:.2%}")
            print(f"  Average response time: {avg_response_time:.3f}s")
            print(f"  Performance degradation: {degradation:.2%}")

    @pytest.mark.asyncio
    async def test_resource_cleanup_performance(self, temp_db_path):
        """Test performance of resource cleanup operations."""
        from backend.app.database.checkpointer import (
            create_checkpointer, save_checkpoint, cleanup_old_checkpoints
        )
        from datetime import datetime, timedelta
        
        checkpointer = await create_checkpointer(temp_db_path)
        
        # Create many old checkpoints
        num_checkpoints = 100
        old_state = {"old": True, "data": "x" * 1000}
        
        create_start = time.time()
        for i in range(num_checkpoints):
            await save_checkpoint(checkpointer, f"old_thread_{i}", old_state)
        create_time = time.time() - create_start
        
        # Cleanup old checkpoints
        cutoff_time = datetime.now() + timedelta(hours=1)  # Future time to clean all
        
        cleanup_start = time.time()
        cleaned_count = await cleanup_old_checkpoints(checkpointer, cutoff_time)
        cleanup_time = time.time() - cleanup_start
        
        # Performance assertions
        assert create_time < 30  # Creating 100 checkpoints should take under 30 seconds
        assert cleanup_time < 10  # Cleanup should take under 10 seconds
        assert cleaned_count >= 0  # Should report number of cleaned checkpoints
        
        print(f"Cleanup Performance:")
        print(f"  Created {num_checkpoints} checkpoints in {create_time:.3f}s")
        print(f"  Cleaned {cleaned_count} checkpoints in {cleanup_time:.3f}s")

    @pytest.mark.asyncio
    async def test_concurrent_workflow_status_polling(self, mock_llm_service):
        """Test performance of concurrent status polling."""
        base_url = "http://localhost:8000"
        
        with patch('backend.app.agents.analyzer.llm_service', mock_llm_service):
            mock_llm_service.generate_response.return_value = {
                "content": '{"problem_type": "automation", "complexity": "low", "solution_category": "SIMPLE_AUTOMATION"}',
                "usage": {"tokens": 100}
            }
            
            # Start a workflow
            async with httpx.AsyncClient() as client:
                start_response = await client.post(
                    f"{base_url}/api/v1/start-analysis",
                    json={
                        "problem_description": "Test polling performance",
                        "user_context": {"technical_level": "beginner"}
                    }
                )
                
                thread_id = start_response.json()["thread_id"]
            
            # Concurrent status polling
            async def poll_status():
                poll_times = []
                for _ in range(10):  # 10 polls per worker
                    start_time = time.time()
                    async with httpx.AsyncClient() as client:
                        try:
                            response = await client.get(f"{base_url}/api/v1/status/{thread_id}")
                            end_time = time.time()
                            poll_times.append({
                                "response_time": end_time - start_time,
                                "success": response.status_code == 200
                            })
                        except Exception:
                            end_time = time.time()
                            poll_times.append({
                                "response_time": end_time - start_time,
                                "success": False
                            })
                    await asyncio.sleep(0.1)
                
                return poll_times
            
            # Run concurrent polling
            num_pollers = 5
            tasks = [poll_status() for _ in range(num_pollers)]
            all_results = await asyncio.gather(*tasks)
            
            # Analyze polling performance
            all_polls = []
            for worker_results in all_results:
                all_polls.extend(worker_results)
            
            successful_polls = [p for p in all_polls if p["success"]]
            success_rate = len(successful_polls) / len(all_polls)
            
            if successful_polls:
                avg_poll_time = statistics.mean([p["response_time"] for p in successful_polls])
                
                assert success_rate >= 0.95  # 95% success rate for polling
                assert avg_poll_time < 0.5  # Polling should be under 500ms
                
                print(f"Polling Performance:")
                print(f"  Total polls: {len(all_polls)}")
                print(f"  Success rate: {success_rate:.2%}")
                print(f"  Average poll time: {avg_poll_time:.3f}s")