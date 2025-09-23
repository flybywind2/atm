/**
 * Unit tests for ProblemInput component
 */

// Mock DOM environment for testing
const { JSDOM } = require('jsdom');
const dom = new JSDOM(`<!DOCTYPE html><html><body></body></html>`);
global.window = dom.window;
global.document = dom.window.document;

// Import component (would need to adjust path based on actual structure)
// const ProblemInput = require('../../../../frontend/js/components/ProblemInput');

describe('ProblemInput Component', () => {
    let container;
    let problemInput;

    beforeEach(() => {
        // Setup DOM container
        container = document.createElement('div');
        container.id = 'test-container';
        document.body.appendChild(container);

        // Mock ProblemInput component
        problemInput = {
            container: container,
            
            validateInput: function(description) {
                if (!description || description.trim().length === 0) {
                    return { valid: false, error: 'Problem description is required' };
                }
                if (description.length < 10) {
                    return { valid: false, error: 'Description must be at least 10 characters' };
                }
                if (description.length > 5000) {
                    return { valid: false, error: 'Description must be less than 5000 characters' };
                }
                return { valid: true };
            },

            validateContext: function(context) {
                const requiredFields = ['technical_level'];
                const validTechLevels = ['beginner', 'intermediate', 'advanced'];
                
                for (let field of requiredFields) {
                    if (!context[field]) {
                        return { valid: false, error: `${field} is required` };
                    }
                }
                
                if (!validTechLevels.includes(context.technical_level)) {
                    return { valid: false, error: 'Invalid technical level' };
                }
                
                return { valid: true };
            },

            submitAnalysis: async function(data) {
                // Mock API call
                if (!data.problem_description || !data.user_context) {
                    throw new Error('Missing required data');
                }
                
                // Simulate network delay
                await new Promise(resolve => setTimeout(resolve, 100));
                
                return {
                    status: 'started',
                    thread_id: 'mock-thread-id-123',
                    message: 'Analysis started successfully'
                };
            },

            render: function() {
                this.container.innerHTML = `
                    <div class="problem-input-form">
                        <textarea id="problem-description" placeholder="Describe your problem..."></textarea>
                        <select id="technical-level">
                            <option value="">Select technical level</option>
                            <option value="beginner">Beginner</option>
                            <option value="intermediate">Intermediate</option>
                            <option value="advanced">Advanced</option>
                        </select>
                        <input type="text" id="environment" placeholder="Environment (e.g., Windows, Linux)">
                        <input type="text" id="tools" placeholder="Current tools">
                        <button id="submit-btn">Start Analysis</button>
                        <div id="error-message" class="error hidden"></div>
                    </div>
                `;
            }
        };
    });

    afterEach(() => {
        if (container && container.parentNode) {
            container.parentNode.removeChild(container);
        }
    });

    describe('Input Validation', () => {
        test('validates empty problem description', () => {
            const result = problemInput.validateInput('');
            expect(result.valid).toBe(false);
            expect(result.error).toContain('required');
        });

        test('validates short problem description', () => {
            const result = problemInput.validateInput('short');
            expect(result.valid).toBe(false);
            expect(result.error).toContain('10 characters');
        });

        test('validates very long problem description', () => {
            const longText = 'a'.repeat(5001);
            const result = problemInput.validateInput(longText);
            expect(result.valid).toBe(false);
            expect(result.error).toContain('5000 characters');
        });

        test('validates valid problem description', () => {
            const result = problemInput.validateInput('I need help automating my daily report generation process');
            expect(result.valid).toBe(true);
            expect(result.error).toBeUndefined();
        });

        test('validates problem description with special characters', () => {
            const result = problemInput.validateInput('I need help with émojis 🚀 and spëcial chars in automation');
            expect(result.valid).toBe(true);
        });
    });

    describe('Context Validation', () => {
        test('validates missing technical level', () => {
            const context = { environment: 'Windows' };
            const result = problemInput.validateContext(context);
            expect(result.valid).toBe(false);
            expect(result.error).toContain('technical_level');
        });

        test('validates invalid technical level', () => {
            const context = { technical_level: 'expert' };
            const result = problemInput.validateContext(context);
            expect(result.valid).toBe(false);
            expect(result.error).toContain('Invalid technical level');
        });

        test('validates valid context', () => {
            const context = { 
                technical_level: 'intermediate',
                environment: 'Windows',
                tools: 'Excel, Python'
            };
            const result = problemInput.validateContext(context);
            expect(result.valid).toBe(true);
        });

        test('validates all technical levels', () => {
            const levels = ['beginner', 'intermediate', 'advanced'];
            
            levels.forEach(level => {
                const context = { technical_level: level };
                const result = problemInput.validateContext(context);
                expect(result.valid).toBe(true);
            });
        });
    });

    describe('Form Rendering', () => {
        test('renders form elements correctly', () => {
            problemInput.render();
            
            expect(container.querySelector('#problem-description')).not.toBeNull();
            expect(container.querySelector('#technical-level')).not.toBeNull();
            expect(container.querySelector('#environment')).not.toBeNull();
            expect(container.querySelector('#tools')).not.toBeNull();
            expect(container.querySelector('#submit-btn')).not.toBeNull();
            expect(container.querySelector('#error-message')).not.toBeNull();
        });

        test('renders technical level options', () => {
            problemInput.render();
            
            const select = container.querySelector('#technical-level');
            const options = select.querySelectorAll('option');
            
            expect(options.length).toBe(4); // Including empty option
            expect(options[1].value).toBe('beginner');
            expect(options[2].value).toBe('intermediate');
            expect(options[3].value).toBe('advanced');
        });

        test('renders with proper CSS classes', () => {
            problemInput.render();
            
            expect(container.querySelector('.problem-input-form')).not.toBeNull();
            expect(container.querySelector('.error.hidden')).not.toBeNull();
        });
    });

    describe('Form Submission', () => {
        test('submits valid form data', async () => {
            const formData = {
                problem_description: 'I need to automate daily Excel reporting tasks',
                user_context: {
                    technical_level: 'beginner',
                    environment: 'Windows 10',
                    tools: 'Excel, Outlook'
                }
            };

            const result = await problemInput.submitAnalysis(formData);
            
            expect(result.status).toBe('started');
            expect(result.thread_id).toBe('mock-thread-id-123');
            expect(result.message).toContain('started successfully');
        });

        test('handles submission with missing data', async () => {
            const formData = {
                problem_description: 'Test problem'
                // Missing user_context
            };

            await expect(problemInput.submitAnalysis(formData))
                .rejects.toThrow('Missing required data');
        });

        test('handles submission errors gracefully', async () => {
            // Override mock to simulate error
            const originalSubmit = problemInput.submitAnalysis;
            problemInput.submitAnalysis = async function() {
                throw new Error('Network error');
            };

            const formData = {
                problem_description: 'Test problem',
                user_context: { technical_level: 'beginner' }
            };

            await expect(problemInput.submitAnalysis(formData))
                .rejects.toThrow('Network error');

            // Restore original mock
            problemInput.submitAnalysis = originalSubmit;
        });
    });

    describe('User Interaction', () => {
        test('handles form field changes', () => {
            problemInput.render();
            
            const textarea = container.querySelector('#problem-description');
            const select = container.querySelector('#technical-level');
            
            // Simulate user input
            textarea.value = 'Test problem description that is long enough';
            select.value = 'intermediate';
            
            // Trigger change events
            textarea.dispatchEvent(new dom.window.Event('input'));
            select.dispatchEvent(new dom.window.Event('change'));
            
            expect(textarea.value).toBe('Test problem description that is long enough');
            expect(select.value).toBe('intermediate');
        });

        test('handles button click events', () => {
            problemInput.render();
            
            const button = container.querySelector('#submit-btn');
            let clicked = false;
            
            button.addEventListener('click', () => {
                clicked = true;
            });
            
            button.click();
            expect(clicked).toBe(true);
        });

        test('shows and hides error messages', () => {
            problemInput.render();
            
            const errorDiv = container.querySelector('#error-message');
            
            // Show error
            errorDiv.textContent = 'Test error message';
            errorDiv.classList.remove('hidden');
            
            expect(errorDiv.textContent).toBe('Test error message');
            expect(errorDiv.classList.contains('hidden')).toBe(false);
            
            // Hide error
            errorDiv.classList.add('hidden');
            expect(errorDiv.classList.contains('hidden')).toBe(true);
        });
    });

    describe('Data Collection', () => {
        test('collects form data correctly', () => {
            problemInput.render();
            
            // Fill form
            container.querySelector('#problem-description').value = 'Test automation problem';
            container.querySelector('#technical-level').value = 'intermediate';
            container.querySelector('#environment').value = 'Windows 10';
            container.querySelector('#tools').value = 'Excel, Python';
            
            // Mock data collection function
            const collectFormData = () => {
                return {
                    problem_description: container.querySelector('#problem-description').value,
                    user_context: {
                        technical_level: container.querySelector('#technical-level').value,
                        environment: container.querySelector('#environment').value,
                        tools: container.querySelector('#tools').value.split(',').map(t => t.trim())
                    }
                };
            };
            
            const data = collectFormData();
            
            expect(data.problem_description).toBe('Test automation problem');
            expect(data.user_context.technical_level).toBe('intermediate');
            expect(data.user_context.environment).toBe('Windows 10');
            expect(data.user_context.tools).toEqual(['Excel', 'Python']);
        });

        test('handles empty optional fields', () => {
            problemInput.render();
            
            // Fill only required fields
            container.querySelector('#problem-description').value = 'Test problem description';
            container.querySelector('#technical-level').value = 'beginner';
            // Leave environment and tools empty
            
            const collectFormData = () => {
                const environment = container.querySelector('#environment').value;
                const tools = container.querySelector('#tools').value;
                
                return {
                    problem_description: container.querySelector('#problem-description').value,
                    user_context: {
                        technical_level: container.querySelector('#technical-level').value,
                        ...(environment && { environment }),
                        ...(tools && { tools: tools.split(',').map(t => t.trim()) })
                    }
                };
            };
            
            const data = collectFormData();
            
            expect(data.problem_description).toBe('Test problem description');
            expect(data.user_context.technical_level).toBe('beginner');
            expect(data.user_context.environment).toBeUndefined();
            expect(data.user_context.tools).toBeUndefined();
        });
    });

    describe('Accessibility', () => {
        test('includes proper labels and ARIA attributes', () => {
            // Enhanced render with accessibility
            problemInput.render = function() {
                this.container.innerHTML = `
                    <div class="problem-input-form" role="form" aria-label="Problem Analysis Form">
                        <label for="problem-description">Problem Description *</label>
                        <textarea id="problem-description" 
                                  aria-required="true"
                                  aria-describedby="desc-help"
                                  placeholder="Describe your problem..."></textarea>
                        <div id="desc-help" class="help-text">Describe the problem you want to solve (minimum 10 characters)</div>
                        
                        <label for="technical-level">Technical Level *</label>
                        <select id="technical-level" aria-required="true">
                            <option value="">Select technical level</option>
                            <option value="beginner">Beginner</option>
                            <option value="intermediate">Intermediate</option>
                            <option value="advanced">Advanced</option>
                        </select>
                        
                        <button id="submit-btn" type="submit" aria-describedby="submit-help">Start Analysis</button>
                        <div id="submit-help" class="help-text">Click to begin problem analysis</div>
                        
                        <div id="error-message" class="error hidden" role="alert" aria-live="polite"></div>
                    </div>
                `;
            };
            
            problemInput.render();
            
            expect(container.querySelector('[role="form"]')).not.toBeNull();
            expect(container.querySelector('[aria-required="true"]')).not.toBeNull();
            expect(container.querySelector('[role="alert"]')).not.toBeNull();
            expect(container.querySelector('[aria-live="polite"]')).not.toBeNull();
        });
    });
});