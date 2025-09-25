/**
 * Markdown Rendering Utilities
 * 
 * This module provides utilities for rendering markdown content
 * with syntax highlighting support.
 */

class MarkdownRenderer {
    constructor() {
        this.marked = window.marked;
        this.hljs = window.hljs;
        
        this.configureMarked();
    }
    
    /**
     * Configure marked.js settings according to agent specifications
     */
    configureMarked() {
        if (!this.marked) {
            console.warn('marked.js not loaded, markdown rendering will be limited');
            return;
        }
        
        // Configure marked with modern API
        this.marked.use({
            highlight: (code, lang) => {
                if (this.hljs && lang && this.hljs.getLanguage(lang)) {
                    try {
                        return this.hljs.highlight(code, { language: lang }).value;
                    } catch (error) {
                        console.warn('Syntax highlighting failed:', error);
                    }
                }
                // Return escaped code for unknown languages
                return this.escapeHtml(code);
            },
            breaks: true,
            gfm: true,
            renderer: {
                code: (code, language) => {
                    const validLang = this.hljs && language && this.hljs.getLanguage(language) ? language : 'text';
                    const highlighted = this.hljs && validLang !== 'text' ? 
                        this.hljs.highlight(code, { language: validLang }).value : 
                        this.escapeHtml(code);
                    
                    return `<pre><code class="hljs language-${validLang}">${highlighted}</code></pre>`;
                },
                codespan: (code) => {
                    return `<code class="inline-code">${this.escapeHtml(code)}</code>`;
                }
            }
        });
    }
    
    /**
     * Escape HTML entities for security
     */
    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }
    
    /**
     * Render markdown to HTML with syntax highlighting
     */
    render(markdown) {
        if (!markdown) {
            return '';
        }
        
        if (!this.marked) {
            // Fallback to basic HTML conversion
            return this.basicMarkdownToHtml(markdown);
        }
        
        try {
            const html = this.marked.parse(markdown);
            
            // Apply syntax highlighting to any code blocks that weren't processed
            const tempDiv = document.createElement('div');
            tempDiv.innerHTML = html;
            
            // Apply highlight.js to code blocks
            if (this.hljs) {
                tempDiv.querySelectorAll('pre code').forEach((block) => {
                    if (!block.classList.contains('hljs')) {
                        this.hljs.highlightElement(block);
                    }
                });
            }
            
            return tempDiv.innerHTML;
            
        } catch (error) {
            console.error('Markdown rendering failed:', error);
            return this.basicMarkdownToHtml(markdown);
        }
    }
    
    /**
     * Basic markdown to HTML conversion (fallback)
     */
    basicMarkdownToHtml(markdown) {
        return markdown
            // Headers
            .replace(/^### (.*$)/gm, '<h3>$1</h3>')
            .replace(/^## (.*$)/gm, '<h2>$1</h2>')
            .replace(/^# (.*$)/gm, '<h1>$1</h1>')
            // Bold and italic
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            // Code blocks
            .replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>')
            // Inline code
            .replace(/`([^`]+)`/g, '<code>$1</code>')
            // Links
            .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2">$1</a>')
            // Line breaks
            .replace(/\n/g, '<br>');
    }
    
    /**
     * Render markdown and wrap in container with styling
     */
    renderWithContainer(markdown, containerClass = 'markdown-content') {
        const html = this.render(markdown);
        return `<div class="${containerClass}">${html}</div>`;
    }
    
    /**
     * Extract plain text from markdown
     */
    extractText(markdown) {
        if (!markdown) {
            return '';
        }
        
        // Remove markdown syntax
        return markdown
            .replace(/#+\s/g, '') // Headers
            .replace(/\*\*(.*?)\*\*/g, '$1') // Bold
            .replace(/\*(.*?)\*/g, '$1') // Italic
            .replace(/```[\s\S]*?```/g, '') // Code blocks
            .replace(/`([^`]+)`/g, '$1') // Inline code
            .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1') // Links
            .replace(/\n+/g, ' ') // Line breaks
            .trim();
    }
    
    /**
     * Get table of contents from markdown
     */
    getTableOfContents(markdown) {
        if (!markdown) {
            return [];
        }
        
        const toc = [];
        const lines = markdown.split('\n');
        
        for (const line of lines) {
            const match = line.match(/^(#{1,6})\s+(.+)$/);
            if (match) {
                const level = match[1].length;
                const title = match[2].trim();
                const id = this.generateId(title);
                
                toc.push({
                    level,
                    title,
                    id,
                    line
                });
            }
        }
        
        return toc;
    }
    
    /**
     * Generate ID from title text
     */
    generateId(title) {
        return title
            .toLowerCase()
            .replace(/[^\w\s-]/g, '')
            .replace(/\s+/g, '-')
            .trim();
    }
    
    /**
     * Render document with table of contents
     */
    renderWithTOC(markdown) {
        const toc = this.getTableOfContents(markdown);
        const content = this.render(markdown);
        
        if (toc.length === 0) {
            return content;
        }
        
        // Generate TOC HTML
        const tocHTML = toc
            .map(item => {
                const indent = '  '.repeat(item.level - 1);
                return `${indent}<li><a href="#${item.id}">${item.title}</a></li>`;
            })
            .join('\n');
        
        return `
            <div class="document-with-toc">
                <div class="table-of-contents">
                    <h3>Table of Contents</h3>
                    <ul>
                        ${tocHTML}
                    </ul>
                </div>
                <div class="document-content">
                    ${content}
                </div>
            </div>
        `;
    }
    
    /**
     * Create downloadable content
     */
    createDownloadableContent(markdown, filename) {
        const blob = new Blob([markdown], { type: 'text/markdown' });
        const url = URL.createObjectURL(blob);
        
        const link = document.createElement('a');
        link.href = url;
        link.download = filename;
        link.style.display = 'none';
        
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
        
        URL.revokeObjectURL(url);
    }
    
    /**
     * Copy content to clipboard
     */
    async copyToClipboard(content) {
        try {
            await navigator.clipboard.writeText(content);
            return true;
        } catch (error) {
            console.warn('Clipboard API not available, using fallback');
            
            // Fallback method
            const textArea = document.createElement('textarea');
            textArea.value = content;
            textArea.style.position = 'fixed';
            textArea.style.opacity = '0';
            
            document.body.appendChild(textArea);
            textArea.select();
            
            try {
                document.execCommand('copy');
                document.body.removeChild(textArea);
                return true;
            } catch (fallbackError) {
                document.body.removeChild(textArea);
                return false;
            }
        }
    }
}

// Create singleton instance
const markdownRenderer = new MarkdownRenderer();

// Export globally
window.MarkdownRenderer = markdownRenderer;

// Provide convenient static methods
window.MarkdownRenderer.render = (markdown) => markdownRenderer.render(markdown);
window.MarkdownRenderer.renderWithContainer = (markdown, className) => 
    markdownRenderer.renderWithContainer(markdown, className);
window.MarkdownRenderer.extractText = (markdown) => markdownRenderer.extractText(markdown);
window.MarkdownRenderer.getTableOfContents = (markdown) => markdownRenderer.getTableOfContents(markdown);
window.MarkdownRenderer.renderWithTOC = (markdown) => markdownRenderer.renderWithTOC(markdown);
window.MarkdownRenderer.createDownloadableContent = (markdown, filename) => 
    markdownRenderer.createDownloadableContent(markdown, filename);
window.MarkdownRenderer.copyToClipboard = (content) => markdownRenderer.copyToClipboard(content);

// Export for Node.js if needed
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MarkdownRenderer;
}