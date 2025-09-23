"""
Output quality validation tests
"""

import pytest
import re
from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of validation check."""
    is_valid: bool
    score: float  # 0.0 to 1.0
    issues: List[str]
    suggestions: List[str]


class RequirementsDocumentValidator:
    """Validator for requirements documents."""
    
    def __init__(self):
        self.required_sections = [
            "functional requirements",
            "non-functional requirements", 
            "acceptance criteria"
        ]
        self.optional_sections = [
            "project overview",
            "technical requirements",
            "user stories",
            "constraints"
        ]
    
    def validate(self, document: str) -> ValidationResult:
        """Validate requirements document structure and content."""
        issues = []
        suggestions = []
        score = 1.0
        
        # Check document length
        if len(document) < 500:
            issues.append("Document is too short (minimum 500 characters)")
            score -= 0.2
        
        # Check for required sections
        doc_lower = document.lower()
        missing_sections = []
        
        for section in self.required_sections:
            if section not in doc_lower:
                missing_sections.append(section)
                score -= 0.3
        
        if missing_sections:
            issues.append(f"Missing required sections: {', '.join(missing_sections)}")
        
        # Check for functional requirements structure
        if "fr-" not in doc_lower and "functional requirement" in doc_lower:
            suggestions.append("Consider using FR-001, FR-002 format for functional requirements")
            score -= 0.1
        
        # Check for non-functional requirements structure
        if "nfr-" not in doc_lower and "non-functional requirement" in doc_lower:
            suggestions.append("Consider using NFR-001, NFR-002 format for non-functional requirements")
            score -= 0.1
        
        # Check for acceptance criteria
        if "acceptance criteria" in doc_lower:
            if not self._has_testable_criteria(document):
                issues.append("Acceptance criteria should be testable and measurable")
                score -= 0.2
        
        # Check for technical details
        if not self._has_technical_details(document):
            suggestions.append("Consider adding more technical specifications")
            score -= 0.1
        
        # Validate markdown structure
        markdown_issues = self._validate_markdown_structure(document)
        issues.extend(markdown_issues)
        score -= len(markdown_issues) * 0.05
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            suggestions=suggestions
        )
    
    def _has_testable_criteria(self, document: str) -> bool:
        """Check if acceptance criteria are testable."""
        criteria_section = self._extract_section(document, "acceptance criteria")
        if not criteria_section:
            return False
        
        # Look for measurable terms
        measurable_terms = [
            "shall", "must", "will", "should", "within", "less than", 
            "greater than", "equal to", "successfully", "accurately"
        ]
        
        return any(term in criteria_section.lower() for term in measurable_terms)
    
    def _has_technical_details(self, document: str) -> bool:
        """Check if document has sufficient technical details."""
        technical_indicators = [
            "api", "database", "server", "client", "framework", "library",
            "algorithm", "data structure", "protocol", "interface"
        ]
        
        count = sum(1 for term in technical_indicators if term in document.lower())
        return count >= 3
    
    def _extract_section(self, document: str, section_name: str) -> str:
        """Extract a specific section from the document."""
        pattern = rf"#{1,3}\s*{re.escape(section_name)}.*?(?=#{1,3}|\Z)"
        match = re.search(pattern, document, re.IGNORECASE | re.DOTALL)
        return match.group(0) if match else ""
    
    def _validate_markdown_structure(self, document: str) -> List[str]:
        """Validate markdown structure."""
        issues = []
        
        # Check for headers
        if not re.search(r'^#{1,6}\s+.+', document, re.MULTILINE):
            issues.append("Document should have proper markdown headers")
        
        # Check for lists
        if re.search(r'^\d+\.\s+.+', document, re.MULTILINE):
            # Has numbered lists, good
            pass
        elif re.search(r'^[-*+]\s+.+', document, re.MULTILINE):
            # Has bullet lists, good
            pass
        else:
            issues.append("Consider using lists for better readability")
        
        return issues


class ImplementationGuideValidator:
    """Validator for implementation guides."""
    
    def __init__(self):
        self.required_elements = [
            "prerequisites",
            "step-by-step",
            "code examples",
            "testing"
        ]
    
    def validate(self, guide: str) -> ValidationResult:
        """Validate implementation guide quality."""
        issues = []
        suggestions = []
        score = 1.0
        
        # Check document length
        if len(guide) < 1000:
            issues.append("Implementation guide is too short (minimum 1000 characters)")
            score -= 0.3
        
        # Check for required elements
        guide_lower = guide.lower()
        missing_elements = []
        
        for element in self.required_elements:
            if element.replace("-", " ") not in guide_lower and element.replace("-", "_") not in guide_lower:
                missing_elements.append(element)
                score -= 0.25
        
        if missing_elements:
            issues.append(f"Missing required elements: {', '.join(missing_elements)}")
        
        # Check for code blocks
        code_blocks = re.findall(r'```[\s\S]*?```', guide)
        if len(code_blocks) < 2:
            issues.append("Implementation guide should include multiple code examples")
            score -= 0.2
        
        # Check for language specification in code blocks
        unspecified_blocks = re.findall(r'```\n', guide)
        if unspecified_blocks:
            suggestions.append("Specify programming language for code blocks (e.g., ```python)")
            score -= 0.05
        
        # Check for installation/setup instructions
        if not self._has_installation_instructions(guide):
            issues.append("Missing installation or setup instructions")
            score -= 0.15
        
        # Check for testing instructions
        if not self._has_testing_instructions(guide):
            suggestions.append("Consider adding testing instructions")
            score -= 0.1
        
        # Validate practical usability
        usability_score = self._validate_usability(guide)
        score *= usability_score
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            suggestions=suggestions
        )
    
    def _has_installation_instructions(self, guide: str) -> bool:
        """Check for installation instructions."""
        installation_indicators = [
            "pip install", "npm install", "yarn add", "apt-get", "brew install",
            "download", "setup", "install", "requirements.txt", "package.json"
        ]
        
        return any(indicator in guide.lower() for indicator in installation_indicators)
    
    def _has_testing_instructions(self, guide: str) -> bool:
        """Check for testing instructions."""
        testing_indicators = [
            "test", "pytest", "unittest", "jest", "testing", "assert",
            "run tests", "test cases", "unit test"
        ]
        
        return any(indicator in guide.lower() for indicator in testing_indicators)
    
    def _validate_usability(self, guide: str) -> float:
        """Validate practical usability of the guide."""
        usability_score = 1.0
        
        # Check for step numbering
        if not re.search(r'step\s+\d+', guide, re.IGNORECASE):
            usability_score -= 0.1
        
        # Check for clear structure
        if guide.count('#') < 3:  # Not enough sections
            usability_score -= 0.1
        
        # Check for command examples
        command_patterns = [
            r'`[^`]+`',  # Inline code
            r'```[\s\S]*?```',  # Code blocks
            r'\$\s+\w+',  # Shell commands
        ]
        
        command_count = sum(len(re.findall(pattern, guide)) for pattern in command_patterns)
        if command_count < 5:
            usability_score -= 0.15
        
        return max(0.0, usability_score)


class UserJourneyValidator:
    """Validator for user journey documents."""
    
    def __init__(self):
        self.required_sections = [
            "current state",
            "future state", 
            "pain points",
            "benefits"
        ]
    
    def validate(self, journey: str) -> ValidationResult:
        """Validate user journey document."""
        issues = []
        suggestions = []
        score = 1.0
        
        # Check document length
        if len(journey) < 800:
            issues.append("User journey is too short (minimum 800 characters)")
            score -= 0.2
        
        # Check for required sections
        journey_lower = journey.lower()
        missing_sections = []
        
        for section in self.required_sections:
            if section not in journey_lower and section.replace(" ", "_") not in journey_lower:
                missing_sections.append(section)
                score -= 0.25
        
        if missing_sections:
            issues.append(f"Missing required sections: {', '.join(missing_sections)}")
        
        # Check for specific journey elements
        if not self._has_timeline_elements(journey):
            suggestions.append("Consider adding timeline elements (time stamps, duration)")
            score -= 0.1
        
        if not self._has_pain_point_analysis(journey):
            issues.append("Missing detailed pain point analysis")
            score -= 0.2
        
        if not self._has_quantifiable_benefits(journey):
            suggestions.append("Consider adding quantifiable benefits (time savings, cost reduction)")
            score -= 0.15
        
        # Check for user personas
        if not self._has_user_personas(journey):
            suggestions.append("Consider adding user persona information")
            score -= 0.1
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            suggestions=suggestions
        )
    
    def _has_timeline_elements(self, journey: str) -> bool:
        """Check for timeline elements."""
        time_patterns = [
            r'\d+:\d+\s*(AM|PM|am|pm)',  # Time stamps
            r'\d+\s*(minutes?|hours?|days?)',  # Duration
            r'step\s+\d+',  # Numbered steps
        ]
        
        return any(re.search(pattern, journey) for pattern in time_patterns)
    
    def _has_pain_point_analysis(self, journey: str) -> bool:
        """Check for pain point analysis."""
        pain_indicators = [
            "pain point", "problem", "issue", "challenge", "difficulty",
            "frustration", "bottleneck", "inefficiency", "error"
        ]
        
        count = sum(1 for indicator in pain_indicators if indicator in journey.lower())
        return count >= 3
    
    def _has_quantifiable_benefits(self, journey: str) -> bool:
        """Check for quantifiable benefits."""
        benefit_patterns = [
            r'\d+%',  # Percentages
            r'\d+\s*(hours?|minutes?|days?)',  # Time savings
            r'\$\d+',  # Cost savings
            r'\d+x\s*(faster|quicker)',  # Multiplier improvements
        ]
        
        return any(re.search(pattern, journey) for pattern in benefit_patterns)
    
    def _has_user_personas(self, journey: str) -> bool:
        """Check for user persona information."""
        persona_indicators = [
            "user", "role", "persona", "stakeholder", "actor",
            "business analyst", "developer", "manager", "administrator"
        ]
        
        return any(indicator in journey.lower() for indicator in persona_indicators)


class SolutionTypeClassificationValidator:
    """Validator for solution type classification accuracy."""
    
    def __init__(self):
        self.expected_mappings = {
            "automation": "SIMPLE_AUTOMATION",
            "automate": "SIMPLE_AUTOMATION", 
            "document search": "RAG",
            "question answering": "RAG",
            "knowledge base": "RAG",
            "classify": "ML_CLASSIFICATION",
            "classification": "ML_CLASSIFICATION",
            "categorize": "ML_CLASSIFICATION",
            "machine learning": "ML_CLASSIFICATION"
        }
    
    def validate_classification(self, problem_description: str, predicted_type: str) -> ValidationResult:
        """Validate solution type classification."""
        issues = []
        suggestions = []
        score = 1.0
        
        # Get expected type based on keywords
        expected_type = self._get_expected_type(problem_description)
        
        if expected_type and expected_type != predicted_type:
            issues.append(f"Classification mismatch: expected {expected_type}, got {predicted_type}")
            score = 0.0
        
        # Check for confidence indicators
        if not self._has_confidence_indicators(problem_description, predicted_type):
            suggestions.append("Classification confidence could be improved with more context")
            score -= 0.1
        
        return ValidationResult(
            is_valid=len(issues) == 0,
            score=max(0.0, score),
            issues=issues,
            suggestions=suggestions
        )
    
    def _get_expected_type(self, description: str) -> str:
        """Get expected solution type based on keywords."""
        description_lower = description.lower()
        
        for keyword, solution_type in self.expected_mappings.items():
            if keyword in description_lower:
                return solution_type
        
        return None
    
    def _has_confidence_indicators(self, description: str, predicted_type: str) -> bool:
        """Check if the classification has good confidence indicators."""
        description_lower = description.lower()
        
        type_keywords = {
            "SIMPLE_AUTOMATION": ["automate", "process", "generate", "daily", "manual"],
            "RAG": ["search", "query", "document", "knowledge", "answer"],
            "ML_CLASSIFICATION": ["classify", "categorize", "predict", "label", "training"]
        }
        
        if predicted_type in type_keywords:
            relevant_keywords = type_keywords[predicted_type]
            matches = sum(1 for keyword in relevant_keywords if keyword in description_lower)
            return matches >= 2
        
        return False


class ComprehensiveQualityValidator:
    """Comprehensive quality validator for all output types."""
    
    def __init__(self):
        self.requirements_validator = RequirementsDocumentValidator()
        self.guide_validator = ImplementationGuideValidator()
        self.journey_validator = UserJourneyValidator()
        self.classification_validator = SolutionTypeClassificationValidator()
    
    def validate_complete_output(self, output: Dict[str, Any]) -> Dict[str, ValidationResult]:
        """Validate complete system output."""
        results = {}
        
        # Validate requirements document
        if "requirements_doc" in output:
            results["requirements"] = self.requirements_validator.validate(output["requirements_doc"])
        
        # Validate implementation guide
        if "implementation_guide" in output:
            results["implementation_guide"] = self.guide_validator.validate(output["implementation_guide"])
        
        # Validate user journey
        if "user_journey" in output:
            results["user_journey"] = self.journey_validator.validate(output["user_journey"])
        
        # Validate solution classification
        if "solution_type" in output and "problem_description" in output:
            results["classification"] = self.classification_validator.validate_classification(
                output["problem_description"], output["solution_type"]
            )
        
        return results
    
    def calculate_overall_score(self, validation_results: Dict[str, ValidationResult]) -> float:
        """Calculate overall quality score."""
        if not validation_results:
            return 0.0
        
        scores = [result.score for result in validation_results.values()]
        return sum(scores) / len(scores)
    
    def get_quality_report(self, validation_results: Dict[str, ValidationResult]) -> Dict[str, Any]:
        """Generate a comprehensive quality report."""
        overall_score = self.calculate_overall_score(validation_results)
        
        all_issues = []
        all_suggestions = []
        
        for component, result in validation_results.items():
            for issue in result.issues:
                all_issues.append(f"{component}: {issue}")
            for suggestion in result.suggestions:
                all_suggestions.append(f"{component}: {suggestion}")
        
        # Determine quality grade
        if overall_score >= 0.9:
            grade = "A"
        elif overall_score >= 0.8:
            grade = "B"
        elif overall_score >= 0.7:
            grade = "C"
        elif overall_score >= 0.6:
            grade = "D"
        else:
            grade = "F"
        
        return {
            "overall_score": overall_score,
            "grade": grade,
            "component_scores": {k: v.score for k, v in validation_results.items()},
            "issues": all_issues,
            "suggestions": all_suggestions,
            "is_acceptable": overall_score >= 0.7,  # Minimum acceptable quality
            "needs_improvement": all_issues,
            "enhancement_opportunities": all_suggestions
        }


# Test cases for validation
class TestOutputValidation:
    """Test cases for output validation."""
    
    def test_requirements_validation_good_document(self):
        """Test validation of a good requirements document."""
        good_requirements = """
        # Software Requirements Specification
        
        ## Functional Requirements
        
        ### FR-001: Data Processing
        - System shall process Excel files automatically
        - System shall validate data integrity
        
        ### FR-002: Report Generation
        - System shall generate summary reports
        - System shall export to PDF format
        
        ## Non-Functional Requirements
        
        ### NFR-001: Performance
        - System shall process files within 30 seconds
        - System shall handle up to 100 files per batch
        
        ### NFR-002: Usability
        - System shall provide clear error messages
        - System shall have intuitive user interface
        
        ## Acceptance Criteria
        - System successfully processes test files without errors
        - Generated reports accurately reflect input data
        - Performance meets specified benchmarks
        """
        
        validator = RequirementsDocumentValidator()
        result = validator.validate(good_requirements)
        
        assert result.is_valid == True
        assert result.score >= 0.8
        assert len(result.issues) == 0
    
    def test_requirements_validation_poor_document(self):
        """Test validation of a poor requirements document."""
        poor_requirements = "This is a very short requirements document without proper structure."
        
        validator = RequirementsDocumentValidator()
        result = validator.validate(poor_requirements)
        
        assert result.is_valid == False
        assert result.score < 0.5
        assert len(result.issues) > 0
    
    def test_implementation_guide_validation(self):
        """Test validation of implementation guide."""
        good_guide = """
        # Implementation Guide
        
        ## Prerequisites
        - Python 3.8+
        - pip package manager
        
        ## Step-by-Step Implementation
        
        ### Step 1: Environment Setup
        ```bash
        pip install pandas fastapi uvicorn
        ```
        
        ### Step 2: Create Main Application
        ```python
        from fastapi import FastAPI
        
        app = FastAPI()
        
        @app.get("/")
        def read_root():
            return {"message": "Hello World"}
        ```
        
        ### Step 3: Run Application
        ```bash
        uvicorn main:app --reload
        ```
        
        ## Testing
        Run unit tests with pytest:
        ```bash
        pytest tests/
        ```
        """
        
        validator = ImplementationGuideValidator()
        result = validator.validate(good_guide)
        
        assert result.is_valid == True
        assert result.score >= 0.7
    
    def test_user_journey_validation(self):
        """Test validation of user journey."""
        good_journey = """
        # User Journey: Report Automation
        
        ## Current State (As-Is)
        
        ### Daily Workflow
        1. **8:00 AM** - Receive data files via email
        2. **9:00 AM** - Manual data processing in Excel
        3. **11:00 AM** - Create summary reports
        4. **12:00 PM** - Send reports to stakeholders
        
        ### Pain Points
        - Manual data entry errors
        - Time-consuming validation process
        - Inconsistent report formatting
        - Delayed delivery to stakeholders
        
        ## Future State (To-Be)
        
        ### Automated Workflow
        1. **8:00 AM** - System automatically processes files
        2. **8:15 AM** - Validation and error checking
        3. **8:30 AM** - Report generation
        4. **8:35 AM** - Automatic distribution
        
        ## Benefits
        - 75% reduction in processing time
        - 99% accuracy improvement
        - Consistent report formatting
        - Real-time delivery
        """
        
        validator = UserJourneyValidator()
        result = validator.validate(good_journey)
        
        assert result.is_valid == True
        assert result.score >= 0.7
    
    def test_solution_classification_validation(self):
        """Test solution type classification validation."""
        test_cases = [
            ("Automate daily report generation", "SIMPLE_AUTOMATION", True),
            ("Build document search system", "RAG", True),
            ("Classify customer feedback", "ML_CLASSIFICATION", True),
            ("Automate daily reports", "RAG", False),  # Misclassification
        ]
        
        validator = SolutionTypeClassificationValidator()
        
        for description, predicted_type, should_be_valid in test_cases:
            result = validator.validate_classification(description, predicted_type)
            assert result.is_valid == should_be_valid
    
    def test_comprehensive_validation(self):
        """Test comprehensive validation of complete output."""
        complete_output = {
            "problem_description": "Automate daily Excel report generation",
            "solution_type": "SIMPLE_AUTOMATION",
            "requirements_doc": """
            # Requirements
            ## Functional Requirements
            ### FR-001: Process Excel files
            ## Non-Functional Requirements
            ### NFR-001: Performance requirements
            ## Acceptance Criteria
            - System shall process files successfully
            """,
            "implementation_guide": """
            # Implementation Guide
            ## Prerequisites
            - Python 3.8+
            ## Step-by-Step Implementation
            ### Step 1: Setup
            ```bash
            pip install pandas
            ```
            ## Testing
            Run tests with pytest
            """,
            "user_journey": """
            # User Journey
            ## Current State
            Manual processing takes 2 hours daily
            ## Future State  
            Automated processing in 5 minutes
            ## Pain Points
            - Time consuming manual work
            ## Benefits
            - 95% time savings
            """
        }
        
        validator = ComprehensiveQualityValidator()
        results = validator.validate_complete_output(complete_output)
        
        assert "requirements" in results
        assert "implementation_guide" in results
        assert "user_journey" in results
        assert "classification" in results
        
        overall_score = validator.calculate_overall_score(results)
        assert overall_score >= 0.0
        assert overall_score <= 1.0
        
        quality_report = validator.get_quality_report(results)
        assert "overall_score" in quality_report
        assert "grade" in quality_report
        assert "is_acceptable" in quality_report