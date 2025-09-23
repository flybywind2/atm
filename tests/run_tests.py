#!/usr/bin/env python3
"""
Comprehensive test runner for the ATM system
"""

import os
import sys
import subprocess
import argparse
import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TestResult:
    """Test execution result."""
    suite: str
    passed: int
    failed: int
    skipped: int
    errors: int
    duration: float
    coverage: Optional[float] = None
    exit_code: int = 0


class TestRunner:
    """Comprehensive test runner for ATM system."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.tests_dir = project_root / "tests"
        self.backend_dir = project_root / "backend"
        self.frontend_dir = project_root / "frontend"
        
        # Test suites configuration
        self.test_suites = {
            "unit": {
                "description": "Unit tests for individual components",
                "paths": [
                    "tests/backend/unit",
                    "tests/frontend/unit"
                ],
                "markers": "not integration and not system and not performance",
                "timeout": 300  # 5 minutes
            },
            "integration": {
                "description": "Integration tests for component interactions",
                "paths": [
                    "tests/backend/integration",
                    "tests/frontend/integration"
                ],
                "markers": "integration",
                "timeout": 600  # 10 minutes
            },
            "system": {
                "description": "End-to-end system tests",
                "paths": ["tests/system"],
                "markers": "system",
                "timeout": 900  # 15 minutes
            },
            "performance": {
                "description": "Performance and load tests",
                "paths": ["tests/backend/performance"],
                "markers": "performance",
                "timeout": 1800  # 30 minutes
            },
            "quality": {
                "description": "Output quality validation tests",
                "paths": ["tests/quality"],
                "markers": "quality",
                "timeout": 300  # 5 minutes
            }
        }
    
    def run_backend_tests(self, suite: str = "all", coverage: bool = True, verbose: bool = False) -> List[TestResult]:
        """Run Python backend tests."""
        results = []
        
        if suite == "all":
            suites_to_run = [s for s in self.test_suites.keys() if s != "frontend"]
        else:
            suites_to_run = [suite] if suite in self.test_suites else []
        
        for suite_name in suites_to_run:
            print(f"\n🧪 Running {suite_name} tests...")
            result = self._run_pytest_suite(suite_name, coverage, verbose)
            results.append(result)
            
            if result.exit_code != 0:
                print(f"❌ {suite_name} tests failed!")
            else:
                print(f"✅ {suite_name} tests passed!")
        
        return results
    
    def run_frontend_tests(self, suite: str = "all", verbose: bool = False) -> List[TestResult]:
        """Run JavaScript frontend tests."""
        results = []
        
        # Check if Node.js and npm are available
        if not self._check_nodejs():
            print("⚠️  Node.js not found, skipping frontend tests")
            return results
        
        # Install dependencies if needed
        if not self._ensure_frontend_deps():
            print("❌ Failed to install frontend dependencies")
            return results
        
        frontend_suites = ["unit", "integration"] if suite == "all" else [suite]
        
        for suite_name in frontend_suites:
            if suite_name in ["unit", "integration"]:
                print(f"\n🧪 Running frontend {suite_name} tests...")
                result = self._run_jest_suite(suite_name, verbose)
                results.append(result)
        
        return results
    
    def run_all_tests(self, coverage: bool = True, verbose: bool = False) -> Dict[str, List[TestResult]]:
        """Run all test suites."""
        print("🚀 Starting comprehensive test execution...")
        
        results = {
            "backend": self.run_backend_tests("all", coverage, verbose),
            "frontend": self.run_frontend_tests("all", verbose)
        }
        
        return results
    
    def _run_pytest_suite(self, suite_name: str, coverage: bool, verbose: bool) -> TestResult:
        """Run a specific pytest suite."""
        suite_config = self.test_suites[suite_name]
        
        # Build pytest command
        cmd = ["python", "-m", "pytest"]
        
        # Add paths
        for path in suite_config["paths"]:
            if (self.project_root / path).exists():
                cmd.append(str(self.project_root / path))
        
        # Add markers
        if suite_config.get("markers"):
            cmd.extend(["-m", suite_config["markers"]])
        
        # Add coverage
        if coverage:
            cmd.extend([
                "--cov=backend/app",
                "--cov-report=term-missing",
                "--cov-report=html:tests/reports/coverage",
                "--cov-report=json:tests/reports/coverage.json"
            ])
        
        # Add verbosity
        if verbose:
            cmd.append("-v")
        else:
            cmd.append("-q")
        
        # Add output formatting
        cmd.extend([
            "--tb=short",
            "--junit-xml=tests/reports/junit.xml",
            f"--timeout={suite_config['timeout']}"
        ])
        
        # Ensure reports directory exists
        reports_dir = self.project_root / "tests" / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Run tests
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=suite_config["timeout"]
            )
            
            duration = time.time() - start_time
            
            # Parse pytest output
            return self._parse_pytest_result(suite_name, result, duration, coverage)
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ {suite_name} tests timed out after {duration:.1f}s")
            
            return TestResult(
                suite=suite_name,
                passed=0,
                failed=1,
                skipped=0,
                errors=1,
                duration=duration,
                exit_code=1
            )
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 Error running {suite_name} tests: {e}")
            
            return TestResult(
                suite=suite_name,
                passed=0,
                failed=1,
                skipped=0,
                errors=1,
                duration=duration,
                exit_code=1
            )
    
    def _run_jest_suite(self, suite_name: str, verbose: bool) -> TestResult:
        """Run a specific Jest suite."""
        # Build Jest command
        cmd = ["npm", "test"]
        
        if suite_name == "unit":
            cmd.append("--testPathPattern=unit")
        elif suite_name == "integration":
            cmd.append("--testPathPattern=integration")
        
        if verbose:
            cmd.append("--verbose")
        
        cmd.extend([
            "--coverage",
            "--coverageDirectory=tests/reports/frontend-coverage",
            "--testResultsProcessor=jest-junit"
        ])
        
        start_time = time.time()
        
        try:
            result = subprocess.run(
                cmd,
                cwd=self.frontend_dir,
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes
            )
            
            duration = time.time() - start_time
            
            return self._parse_jest_result(f"frontend-{suite_name}", result, duration)
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            print(f"⏰ Frontend {suite_name} tests timed out after {duration:.1f}s")
            
            return TestResult(
                suite=f"frontend-{suite_name}",
                passed=0,
                failed=1,
                skipped=0,
                errors=1,
                duration=duration,
                exit_code=1
            )
        except Exception as e:
            duration = time.time() - start_time
            print(f"💥 Error running frontend {suite_name} tests: {e}")
            
            return TestResult(
                suite=f"frontend-{suite_name}",
                passed=0,
                failed=1,
                skipped=0,
                errors=1,
                duration=duration,
                exit_code=1
            )
    
    def _parse_pytest_result(self, suite_name: str, result: subprocess.CompletedProcess, 
                           duration: float, coverage: bool) -> TestResult:
        """Parse pytest execution result."""
        output = result.stdout + result.stderr
        
        # Extract test counts
        passed = 0
        failed = 0
        skipped = 0
        errors = 0
        
        # Parse pytest summary line
        import re
        
        # Look for patterns like "5 passed, 1 failed, 2 skipped"
        summary_pattern = r'(\d+) passed|(\d+) failed|(\d+) skipped|(\d+) error'
        matches = re.findall(summary_pattern, output)
        
        for match in matches:
            if match[0]:  # passed
                passed = int(match[0])
            elif match[1]:  # failed
                failed = int(match[1])
            elif match[2]:  # skipped
                skipped = int(match[2])
            elif match[3]:  # errors
                errors = int(match[3])
        
        # Extract coverage if available
        coverage_percent = None
        if coverage:
            coverage_pattern = r'TOTAL.*?(\d+)%'
            coverage_match = re.search(coverage_pattern, output)
            if coverage_match:
                coverage_percent = float(coverage_match.group(1))
        
        return TestResult(
            suite=suite_name,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration=duration,
            coverage=coverage_percent,
            exit_code=result.returncode
        )
    
    def _parse_jest_result(self, suite_name: str, result: subprocess.CompletedProcess, 
                          duration: float) -> TestResult:
        """Parse Jest execution result."""
        output = result.stdout + result.stderr
        
        # Parse Jest output (simplified)
        import re
        
        passed = 0
        failed = 0
        
        # Look for Jest summary
        test_pattern = r'Tests:\s+(\d+) failed.*?(\d+) passed'
        match = re.search(test_pattern, output)
        if match:
            failed = int(match.group(1))
            passed = int(match.group(2))
        else:
            # Alternative pattern
            passed_pattern = r'(\d+) passing'
            failed_pattern = r'(\d+) failing'
            
            passed_match = re.search(passed_pattern, output)
            failed_match = re.search(failed_pattern, output)
            
            if passed_match:
                passed = int(passed_match.group(1))
            if failed_match:
                failed = int(failed_match.group(1))
        
        return TestResult(
            suite=suite_name,
            passed=passed,
            failed=failed,
            skipped=0,
            errors=0,
            duration=duration,
            exit_code=result.returncode
        )
    
    def _check_nodejs(self) -> bool:
        """Check if Node.js is available."""
        try:
            subprocess.run(["node", "--version"], capture_output=True, check=True)
            subprocess.run(["npm", "--version"], capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
    
    def _ensure_frontend_deps(self) -> bool:
        """Ensure frontend dependencies are installed."""
        package_json = self.frontend_dir / "package.json"
        node_modules = self.frontend_dir / "node_modules"
        
        if not package_json.exists():
            print("📦 Creating basic package.json for frontend tests...")
            self._create_frontend_package_json()
        
        if not node_modules.exists():
            print("📦 Installing frontend dependencies...")
            try:
                subprocess.run(
                    ["npm", "install"],
                    cwd=self.frontend_dir,
                    check=True,
                    capture_output=True
                )
                return True
            except subprocess.CalledProcessError:
                return False
        
        return True
    
    def _create_frontend_package_json(self):
        """Create basic package.json for frontend testing."""
        package_json = {
            "name": "atm-frontend-tests",
            "version": "1.0.0",
            "description": "Frontend tests for ATM system",
            "scripts": {
                "test": "jest",
                "test:watch": "jest --watch",
                "test:coverage": "jest --coverage"
            },
            "devDependencies": {
                "jest": "^29.0.0",
                "jest-environment-jsdom": "^29.0.0",
                "jest-junit": "^15.0.0",
                "@babel/core": "^7.0.0",
                "@babel/preset-env": "^7.0.0"
            },
            "jest": {
                "testEnvironment": "jsdom",
                "testMatch": ["**/tests/**/*.js"],
                "collectCoverageFrom": [
                    "js/**/*.js",
                    "!js/libs/**"
                ],
                "coverageReporters": ["text", "html", "json"],
                "reporters": [
                    "default",
                    ["jest-junit", {"outputDirectory": "tests/reports"}]
                ]
            }
        }
        
        package_json_path = self.frontend_dir / "package.json"
        with open(package_json_path, 'w') as f:
            json.dump(package_json, f, indent=2)
    
    def generate_report(self, results: Dict[str, List[TestResult]], output_file: Optional[Path] = None) -> Dict[str, Any]:
        """Generate comprehensive test report."""
        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "total_suites": 0,
                "total_tests": 0,
                "total_passed": 0,
                "total_failed": 0,
                "total_skipped": 0,
                "total_errors": 0,
                "total_duration": 0.0,
                "overall_success": True
            },
            "suites": {},
            "coverage": {}
        }
        
        # Aggregate results
        for test_type, suite_results in results.items():
            for result in suite_results:
                report["summary"]["total_suites"] += 1
                report["summary"]["total_tests"] += result.passed + result.failed + result.skipped
                report["summary"]["total_passed"] += result.passed
                report["summary"]["total_failed"] += result.failed
                report["summary"]["total_skipped"] += result.skipped
                report["summary"]["total_errors"] += result.errors
                report["summary"]["total_duration"] += result.duration
                
                if result.exit_code != 0:
                    report["summary"]["overall_success"] = False
                
                report["suites"][result.suite] = asdict(result)
                
                if result.coverage is not None:
                    report["coverage"][result.suite] = result.coverage
        
        # Calculate success rate
        total_tests = report["summary"]["total_tests"]
        if total_tests > 0:
            success_rate = (report["summary"]["total_passed"] / total_tests) * 100
            report["summary"]["success_rate"] = success_rate
        else:
            report["summary"]["success_rate"] = 0.0
        
        # Save report if output file specified
        if output_file:
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
        
        return report
    
    def print_summary(self, results: Dict[str, List[TestResult]]):
        """Print test execution summary."""
        print("\n" + "="*60)
        print("🏁 TEST EXECUTION SUMMARY")
        print("="*60)
        
        all_results = []
        for suite_results in results.values():
            all_results.extend(suite_results)
        
        if not all_results:
            print("No tests were executed.")
            return
        
        total_passed = sum(r.passed for r in all_results)
        total_failed = sum(r.failed for r in all_results)
        total_skipped = sum(r.skipped for r in all_results)
        total_errors = sum(r.errors for r in all_results)
        total_duration = sum(r.duration for r in all_results)
        
        print(f"📊 Results:")
        print(f"   ✅ Passed:  {total_passed}")
        print(f"   ❌ Failed:  {total_failed}")
        print(f"   ⏭️  Skipped: {total_skipped}")
        print(f"   💥 Errors:  {total_errors}")
        print(f"   ⏱️  Duration: {total_duration:.2f}s")
        
        # Per suite breakdown
        print(f"\n📋 Suite breakdown:")
        for result in all_results:
            status = "✅" if result.exit_code == 0 else "❌"
            coverage_info = f" ({result.coverage:.1f}% coverage)" if result.coverage else ""
            print(f"   {status} {result.suite}: {result.passed}P/{result.failed}F/{result.skipped}S in {result.duration:.1f}s{coverage_info}")
        
        # Overall status
        overall_success = all(r.exit_code == 0 for r in all_results)
        if overall_success:
            print(f"\n🎉 All tests passed! Total execution time: {total_duration:.2f}s")
        else:
            failed_suites = [r.suite for r in all_results if r.exit_code != 0]
            print(f"\n💔 Test failures in: {', '.join(failed_suites)}")
        
        print("="*60)


def main():
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(description="ATM System Test Runner")
    parser.add_argument("--suite", choices=["unit", "integration", "system", "performance", "quality", "all"], 
                       default="all", help="Test suite to run")
    parser.add_argument("--backend-only", action="store_true", help="Run only backend tests")
    parser.add_argument("--frontend-only", action="store_true", help="Run only frontend tests")
    parser.add_argument("--no-coverage", action="store_true", help="Disable coverage reporting")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--report", type=str, help="Output file for JSON report")
    parser.add_argument("--project-root", type=str, help="Project root directory", 
                       default=str(Path(__file__).parent.parent))
    
    args = parser.parse_args()
    
    project_root = Path(args.project_root).resolve()
    runner = TestRunner(project_root)
    
    # Determine what to run
    if args.backend_only:
        results = {"backend": runner.run_backend_tests(args.suite, not args.no_coverage, args.verbose)}
    elif args.frontend_only:
        results = {"frontend": runner.run_frontend_tests(args.suite, args.verbose)}
    else:
        results = runner.run_all_tests(not args.no_coverage, args.verbose)
    
    # Generate and display results
    report_file = Path(args.report) if args.report else None
    report = runner.generate_report(results, report_file)
    runner.print_summary(results)
    
    # Exit with appropriate code
    overall_success = report["summary"]["overall_success"]
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()