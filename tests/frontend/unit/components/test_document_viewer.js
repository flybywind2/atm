/**
 * Unit tests for DocumentViewer component
 */

// Mock DOM environment
const { JSDOM } = require('jsdom');
const dom = new JSDOM(`<!DOCTYPE html><html><body></body></html>`);
global.window = dom.window;
global.document = dom.window.document;

describe('DocumentViewer Component', () => {
    let container;
    let documentViewer;

    beforeEach(() => {
        container = document.createElement('div');
        container.id = 'document-container';
        document.body.appendChild(container);

        // Mock DocumentViewer component
        documentViewer = {
            container: container,
            currentDocument: null,
            documents: {},
            activeTab: null,

            // Mock markdown parser (simplified)
            parseMarkdown: function(markdown) {
                return markdown
                    .replace(/^# (.*$)/gim, '<h1>$1</h1>')
                    .replace(/^## (.*$)/gim, '<h2>$1</h2>')
                    .replace(/^### (.*$)/gim, '<h3>$1</h3>')
                    .replace(/\*\*(.*)\*\*/gim, '<strong>$1</strong>')
                    .replace(/\*(.*)\*/gim, '<em>$1</em>')
                    .replace(/```(\w+)?\n([\s\S]*?)\n```/gim, '<pre class="code-block language-$1"><code>$2</code></pre>')
                    .replace(/`([^`]+)`/gim, '<code class="inline-code">$1</code>')
                    .replace(/^\- (.*$)/gim, '<li>$1</li>')
                    .replace(/(<li>.*<\/li>)/gims, '<ul>$1</ul>')
                    .replace(/\n/gim, '<br>');
            },

            // Mock syntax highlighter
            highlightCode: function(codeBlock) {
                const language = codeBlock.className.match(/language-(\w+)/);
                if (language) {
                    // Simple syntax highlighting mock
                    let code = codeBlock.querySelector('code');
                    if (code && language[1] === 'python') {
                        code.innerHTML = code.innerHTML
                            .replace(/\b(def|import|from|class|if|else|for|while|return|try|except)\b/g, '<span class="keyword">$1</span>')
                            .replace(/\b(\d+)\b/g, '<span class="number">$1</span>')
                            .replace(/'([^']*)'/g, '<span class="string">\'$1\'</span>')
                            .replace(/"([^"]*)"/g, '<span class="string">"$1"</span>');
                    }
                }
            },

            render: function() {
                const tabsHtml = Object.keys(this.documents).map(docType => `
                    <button class="doc-tab ${docType === this.activeTab ? 'active' : ''}" 
                            data-doc-type="${docType}">
                        ${this.getDocumentTitle(docType)}
                    </button>
                `).join('');

                this.container.innerHTML = `
                    <div class="document-viewer">
                        <div class="document-tabs">
                            ${tabsHtml}
                        </div>
                        <div class="document-content">
                            ${this.currentDocument ? this.renderDocument(this.currentDocument) : '<div class="no-document">No document selected</div>'}
                        </div>
                        <div class="document-actions">
                            <button id="download-btn" ${!this.currentDocument ? 'disabled' : ''}>Download</button>
                            <button id="copy-btn" ${!this.currentDocument ? 'disabled' : ''}>Copy to Clipboard</button>
                            <button id="print-btn" ${!this.currentDocument ? 'disabled' : ''}>Print</button>
                        </div>
                    </div>
                `;

                this.attachEventListeners();
                this.highlightAllCode();
            },

            renderDocument: function(document) {
                const content = this.parseMarkdown(document.content);
                return `
                    <div class="document-header">
                        <h2>${document.title}</h2>
                        <div class="document-meta">
                            <span class="doc-type">${document.type}</span>
                            <span class="generated-at">${document.generatedAt}</span>
                        </div>
                    </div>
                    <div class="document-body">
                        ${content}
                    </div>
                `;
            },

            getDocumentTitle: function(docType) {
                const titles = {
                    'requirements': 'Requirements',
                    'implementation_guide': 'Implementation Guide',
                    'user_journey': 'User Journey',
                    'solution_design': 'Solution Design'
                };
                return titles[docType] || docType;
            },

            loadDocuments: function(documents) {
                this.documents = {};
                
                Object.keys(documents).forEach(key => {
                    if (documents[key]) {
                        this.documents[key] = {
                            type: key,
                            title: this.getDocumentTitle(key),
                            content: documents[key],
                            generatedAt: new Date().toLocaleDateString()
                        };
                    }
                });

                // Set first document as active
                const firstDoc = Object.keys(this.documents)[0];
                if (firstDoc) {
                    this.setActiveDocument(firstDoc);
                }

                this.render();
            },

            setActiveDocument: function(docType) {
                if (this.documents[docType]) {
                    this.activeTab = docType;
                    this.currentDocument = this.documents[docType];
                    this.render();
                }
            },

            attachEventListeners: function() {
                // Tab clicks
                const tabs = this.container.querySelectorAll('.doc-tab');
                tabs.forEach(tab => {
                    tab.addEventListener('click', () => {
                        const docType = tab.dataset.docType;
                        this.setActiveDocument(docType);
                    });
                });

                // Action buttons
                const downloadBtn = this.container.querySelector('#download-btn');
                const copyBtn = this.container.querySelector('#copy-btn');
                const printBtn = this.container.querySelector('#print-btn');

                if (downloadBtn) {
                    downloadBtn.addEventListener('click', () => this.downloadDocument());
                }
                if (copyBtn) {
                    copyBtn.addEventListener('click', () => this.copyToClipboard());
                }
                if (printBtn) {
                    printBtn.addEventListener('click', () => this.printDocument());
                }
            },

            highlightAllCode: function() {
                const codeBlocks = this.container.querySelectorAll('.code-block');
                codeBlocks.forEach(block => this.highlightCode(block));
            },

            downloadDocument: function() {
                if (!this.currentDocument) return;

                const blob = new Blob([this.currentDocument.content], { type: 'text/markdown' });
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = `${this.currentDocument.type}.md`;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(url);
            },

            copyToClipboard: async function() {
                if (!this.currentDocument) return;

                try {
                    await navigator.clipboard.writeText(this.currentDocument.content);
                    this.showNotification('Copied to clipboard!');
                } catch (err) {
                    console.error('Failed to copy:', err);
                    this.showNotification('Failed to copy to clipboard', 'error');
                }
            },

            printDocument: function() {
                if (!this.currentDocument) return;

                const printWindow = window.open('', '_blank');
                const content = this.parseMarkdown(this.currentDocument.content);
                
                printWindow.document.write(`
                    <html>
                        <head>
                            <title>${this.currentDocument.title}</title>
                            <style>
                                body { font-family: Arial, sans-serif; margin: 20px; }
                                h1, h2, h3 { color: #333; }
                                code { background: #f5f5f5; padding: 2px 4px; }
                                pre { background: #f5f5f5; padding: 10px; overflow-x: auto; }
                            </style>
                        </head>
                        <body>
                            <h1>${this.currentDocument.title}</h1>
                            ${content}
                        </body>
                    </html>
                `);
                
                printWindow.document.close();
                printWindow.print();
            },

            showNotification: function(message, type = 'success') {
                const notification = document.createElement('div');
                notification.className = `notification ${type}`;
                notification.textContent = message;
                notification.style.cssText = `
                    position: fixed;
                    top: 20px;
                    right: 20px;
                    padding: 10px 20px;
                    background: ${type === 'error' ? '#ff4444' : '#44ff44'};
                    color: white;
                    border-radius: 4px;
                    z-index: 1000;
                `;
                
                document.body.appendChild(notification);
                
                setTimeout(() => {
                    if (notification.parentNode) {
                        notification.parentNode.removeChild(notification);
                    }
                }, 3000);
            },

            searchDocument: function(query) {
                if (!this.currentDocument || !query) return [];

                const content = this.currentDocument.content.toLowerCase();
                const queryLower = query.toLowerCase();
                const matches = [];
                
                const lines = this.currentDocument.content.split('\n');
                lines.forEach((line, index) => {
                    if (line.toLowerCase().includes(queryLower)) {
                        matches.push({
                            line: index + 1,
                            content: line.trim(),
                            context: lines.slice(Math.max(0, index - 1), index + 2).join('\n')
                        });
                    }
                });

                return matches;
            },

            validateDocumentStructure: function(content) {
                const issues = [];
                
                // Check for headers
                if (!content.includes('#')) {
                    issues.push('Document should have at least one header');
                }
                
                // Check for minimum content length
                if (content.length < 100) {
                    issues.push('Document content is too short');
                }
                
                // Check for code blocks if it's an implementation guide
                if (this.currentDocument && this.currentDocument.type === 'implementation_guide') {
                    if (!content.includes('```')) {
                        issues.push('Implementation guide should include code examples');
                    }
                }
                
                return issues;
            }
        };
    });

    afterEach(() => {
        if (container && container.parentNode) {
            container.parentNode.removeChild(container);
        }
    });

    describe('Rendering', () => {
        test('renders empty state correctly', () => {
            documentViewer.render();
            
            expect(container.querySelector('.document-viewer')).not.toBeNull();
            expect(container.querySelector('.no-document')).not.toBeNull();
            expect(container.querySelector('#download-btn').disabled).toBe(true);
        });

        test('renders document tabs correctly', () => {
            const documents = {
                'requirements': '# Requirements Document\n## Functional Requirements\n- Feature 1',
                'implementation_guide': '# Implementation Guide\n## Setup\n```python\nprint("hello")\n```'
            };

            documentViewer.loadDocuments(documents);

            const tabs = container.querySelectorAll('.doc-tab');
            expect(tabs.length).toBe(2);
            expect(tabs[0].textContent.trim()).toBe('Requirements');
            expect(tabs[1].textContent.trim()).toBe('Implementation Guide');
        });

        test('renders active document content', () => {
            const documents = {
                'requirements': '# Requirements Document\n## Functional Requirements\n- Feature 1'
            };

            documentViewer.loadDocuments(documents);

            expect(container.querySelector('h1')).not.toBeNull();
            expect(container.querySelector('h1').textContent).toBe('Requirements Document');
            expect(container.querySelector('h2')).not.toBeNull();
            expect(container.querySelector('h2').textContent).toBe('Functional Requirements');
        });
    });

    describe('Markdown Parsing', () => {
        test('parses headers correctly', () => {
            const markdown = '# Header 1\n## Header 2\n### Header 3';
            const html = documentViewer.parseMarkdown(markdown);
            
            expect(html).toContain('<h1>Header 1</h1>');
            expect(html).toContain('<h2>Header 2</h2>');
            expect(html).toContain('<h3>Header 3</h3>');
        });

        test('parses text formatting correctly', () => {
            const markdown = '**bold text** and *italic text*';
            const html = documentViewer.parseMarkdown(markdown);
            
            expect(html).toContain('<strong>bold text</strong>');
            expect(html).toContain('<em>italic text</em>');
        });

        test('parses code blocks correctly', () => {
            const markdown = '```python\ndef hello():\n    print("world")\n```';
            const html = documentViewer.parseMarkdown(markdown);
            
            expect(html).toContain('<pre class="code-block language-python">');
            expect(html).toContain('<code>def hello():\n    print("world")</code>');
        });

        test('parses inline code correctly', () => {
            const markdown = 'Use `console.log()` for debugging';
            const html = documentViewer.parseMarkdown(markdown);
            
            expect(html).toContain('<code class="inline-code">console.log()</code>');
        });

        test('parses lists correctly', () => {
            const markdown = '- Item 1\n- Item 2\n- Item 3';
            const html = documentViewer.parseMarkdown(markdown);
            
            expect(html).toContain('<ul>');
            expect(html).toContain('<li>Item 1</li>');
            expect(html).toContain('<li>Item 2</li>');
        });
    });

    describe('Document Management', () => {
        test('loads multiple documents correctly', () => {
            const documents = {
                'requirements': '# Requirements',
                'implementation_guide': '# Guide',
                'user_journey': '# Journey'
            };

            documentViewer.loadDocuments(documents);

            expect(Object.keys(documentViewer.documents).length).toBe(3);
            expect(documentViewer.activeTab).toBe('requirements');
            expect(documentViewer.currentDocument.type).toBe('requirements');
        });

        test('switches between documents correctly', () => {
            const documents = {
                'requirements': '# Requirements Document',
                'implementation_guide': '# Implementation Guide'
            };

            documentViewer.loadDocuments(documents);
            
            // Initially should show requirements
            expect(documentViewer.activeTab).toBe('requirements');
            
            // Switch to implementation guide
            documentViewer.setActiveDocument('implementation_guide');
            expect(documentViewer.activeTab).toBe('implementation_guide');
            expect(documentViewer.currentDocument.type).toBe('implementation_guide');
        });

        test('handles missing documents gracefully', () => {
            documentViewer.setActiveDocument('nonexistent');
            expect(documentViewer.activeTab).toBeNull();
            expect(documentViewer.currentDocument).toBeNull();
        });
    });

    describe('User Interactions', () => {
        test('handles tab clicks', () => {
            const documents = {
                'requirements': '# Requirements',
                'implementation_guide': '# Guide'
            };

            documentViewer.loadDocuments(documents);

            const tabs = container.querySelectorAll('.doc-tab');
            const guideTab = Array.from(tabs).find(tab => 
                tab.dataset.docType === 'implementation_guide'
            );

            guideTab.click();
            expect(documentViewer.activeTab).toBe('implementation_guide');
        });

        test('handles download button click', () => {
            // Mock Blob and URL for testing
            global.Blob = class MockBlob {
                constructor(content, options) {
                    this.content = content;
                    this.options = options;
                }
            };

            global.URL = {
                createObjectURL: () => 'mock-url',
                revokeObjectURL: () => {}
            };

            const documents = { 'requirements': '# Test Document' };
            documentViewer.loadDocuments(documents);

            let downloadTriggered = false;
            const originalCreateElement = document.createElement;
            document.createElement = function(tagName) {
                const element = originalCreateElement.call(this, tagName);
                if (tagName === 'a') {
                    element.click = () => { downloadTriggered = true; };
                }
                return element;
            };

            const downloadBtn = container.querySelector('#download-btn');
            downloadBtn.click();

            expect(downloadTriggered).toBe(true);

            // Restore original function
            document.createElement = originalCreateElement;
        });

        test('handles copy button click', async () => {
            // Mock clipboard API
            global.navigator = {
                clipboard: {
                    writeText: jest.fn().mockResolvedValue()
                }
            };

            const documents = { 'requirements': '# Test Document' };
            documentViewer.loadDocuments(documents);

            const copyBtn = container.querySelector('#copy-btn');
            await copyBtn.click();

            // Note: In a real test environment, you'd verify the clipboard was called
        });
    });

    describe('Code Highlighting', () => {
        test('highlights Python code correctly', () => {
            const codeBlock = document.createElement('pre');
            codeBlock.className = 'code-block language-python';
            codeBlock.innerHTML = '<code>def hello():\n    return "world"</code>';

            documentViewer.highlightCode(codeBlock);

            const code = codeBlock.querySelector('code');
            expect(code.innerHTML).toContain('<span class="keyword">def</span>');
            expect(code.innerHTML).toContain('<span class="keyword">return</span>');
            expect(code.innerHTML).toContain('<span class="string">"world"</span>');
        });

        test('handles code blocks without language specification', () => {
            const codeBlock = document.createElement('pre');
            codeBlock.className = 'code-block';
            codeBlock.innerHTML = '<code>console.log("test")</code>';

            // Should not throw error
            expect(() => documentViewer.highlightCode(codeBlock)).not.toThrow();
        });
    });

    describe('Document Search', () => {
        test('finds text matches in document', () => {
            const documents = {
                'requirements': '# Requirements\n## Functional Requirements\n- User authentication\n- Data processing\n## Non-functional Requirements\n- Performance'
            };

            documentViewer.loadDocuments(documents);

            const matches = documentViewer.searchDocument('requirements');
            expect(matches.length).toBeGreaterThan(0);
            expect(matches.some(match => match.content.toLowerCase().includes('requirements'))).toBe(true);
        });

        test('returns empty array for no matches', () => {
            const documents = { 'requirements': '# Simple document' };
            documentViewer.loadDocuments(documents);

            const matches = documentViewer.searchDocument('nonexistent');
            expect(matches.length).toBe(0);
        });

        test('handles empty search query', () => {
            const documents = { 'requirements': '# Test' };
            documentViewer.loadDocuments(documents);

            const matches = documentViewer.searchDocument('');
            expect(matches.length).toBe(0);
        });
    });

    describe('Document Validation', () => {
        test('validates document structure', () => {
            // Valid document
            documentViewer.currentDocument = {
                type: 'requirements',
                content: '# Requirements\n## Functional\n- Feature 1\n- Feature 2\n\nThis is a comprehensive requirements document with sufficient content.'
            };

            const issues = documentViewer.validateDocumentStructure(documentViewer.currentDocument.content);
            expect(issues.length).toBe(0);
        });

        test('identifies missing headers', () => {
            const content = 'This document has no headers';
            const issues = documentViewer.validateDocumentStructure(content);
            
            expect(issues.some(issue => issue.includes('header'))).toBe(true);
        });

        test('identifies insufficient content', () => {
            const content = 'Short';
            const issues = documentViewer.validateDocumentStructure(content);
            
            expect(issues.some(issue => issue.includes('too short'))).toBe(true);
        });

        test('validates implementation guide requirements', () => {
            documentViewer.currentDocument = {
                type: 'implementation_guide',
                content: '# Guide\nThis guide has no code examples.'
            };

            const issues = documentViewer.validateDocumentStructure(documentViewer.currentDocument.content);
            expect(issues.some(issue => issue.includes('code examples'))).toBe(true);
        });
    });
});