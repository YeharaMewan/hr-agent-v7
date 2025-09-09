"""
PDF Service - Generates PDF documents from HTML content
Uses WeasyPrint for high-quality PDF generation with CSS support.
"""

import os
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import tempfile
from datetime import datetime

try:
    from weasyprint import HTML, CSS
    from weasyprint.text.fonts import FontConfiguration
    WEASYPRINT_AVAILABLE = True
except ImportError:
    WEASYPRINT_AVAILABLE = False
    logging.warning("WeasyPrint not available. PDF generation will be limited.")

logger = logging.getLogger(__name__)

class PDFService:
    """Service for generating PDF documents from HTML"""
    
    def __init__(self):
        self.font_config = None
        if WEASYPRINT_AVAILABLE:
            try:
                self.font_config = FontConfiguration()
            except Exception as e:
                logger.warning(f"Could not initialize font configuration: {e}")
        
        # PDF generation options
        self.pdf_options = {
            "presentational_hints": True,
            "optimize_images": True,
        }
        
        # Default CSS for professional documents
        self.default_css = """
        @page {
            size: A4;
            margin: 1in;
            @top-center {
                content: "";
            }
            @bottom-center {
                content: counter(page) " of " counter(pages);
                font-size: 10pt;
                color: #666;
            }
        }
        
        body {
            font-family: 'Times New Roman', serif;
            font-size: 12pt;
            line-height: 1.4;
            color: #333;
            margin: 0;
            padding: 0;
        }
        
        .letterhead {
            text-align: center;
            margin-bottom: 30px;
            border-bottom: 2px solid #333;
            padding-bottom: 20px;
        }
        
        .company-name {
            font-size: 18pt;
            font-weight: bold;
            color: #2c5aa0;
            margin-bottom: 10px;
        }
        
        .company-info {
            font-size: 10pt;
            color: #666;
            line-height: 1.2;
        }
        
        .letter-content {
            margin-top: 30px;
            margin-bottom: 30px;
        }
        
        .letter-title {
            text-align: center;
            font-size: 14pt;
            font-weight: bold;
            text-decoration: underline;
            margin-bottom: 30px;
        }
        
        .date-line {
            text-align: right;
            margin-bottom: 20px;
            font-weight: bold;
        }
        
        .recipient-info {
            margin-bottom: 20px;
        }
        
        .salutation {
            margin-bottom: 20px;
            font-weight: bold;
        }
        
        .letter-body {
            text-align: justify;
            margin-bottom: 30px;
        }
        
        .letter-body p {
            margin-bottom: 15px;
        }
        
        .employee-details {
            background-color: #f9f9f9;
            padding: 15px;
            border-left: 3px solid #2c5aa0;
            margin: 20px 0;
        }
        
        .employee-details table {
            width: 100%;
            border-collapse: collapse;
        }
        
        .employee-details td {
            padding: 5px 10px;
            border-bottom: 1px solid #ddd;
        }
        
        .employee-details td:first-child {
            font-weight: bold;
            width: 30%;
        }
        
        .signature-section {
            margin-top: 50px;
        }
        
        .signature-block {
            float: right;
            text-align: center;
            width: 200px;
        }
        
        .signature-line {
            border-bottom: 1px solid #333;
            margin-bottom: 5px;
            height: 40px;
        }
        
        .signature-title {
            font-size: 10pt;
            font-weight: bold;
        }
        
        .signature-name {
            font-size: 10pt;
        }
        
        .footer {
            clear: both;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ccc;
            font-size: 9pt;
            color: #666;
            text-align: center;
        }
        
        /* Print-specific styles */
        @media print {
            body { margin: 0; }
            .no-print { display: none; }
        }
        """
    
    async def generate_pdf(
        self,
        html_content: str,
        output_path: str,
        custom_css: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate PDF from HTML content
        
        Args:
            html_content: HTML content to convert
            output_path: Path where PDF should be saved
            custom_css: Optional custom CSS to apply
            
        Returns:
            Dictionary with generation result info
        """
        if not WEASYPRINT_AVAILABLE:
            return await self._generate_fallback_pdf(html_content, output_path)
        
        try:
            logger.info(f"Generating PDF: {output_path}")
            
            # Prepare CSS
            css_content = self.default_css
            if custom_css:
                css_content += "\n\n" + custom_css
            
            # Create HTML document
            html_doc = HTML(string=html_content, base_url=".")
            
            # Create CSS stylesheet
            css_stylesheet = CSS(string=css_content, font_config=self.font_config)
            
            # Generate PDF
            await asyncio.to_thread(
                self._generate_pdf_sync,
                html_doc,
                css_stylesheet,
                output_path
            )
            
            # Verify file was created
            if not os.path.exists(output_path):
                raise Exception("PDF file was not created")
            
            file_size = os.path.getsize(output_path)
            
            logger.info(f"PDF generated successfully: {output_path} ({file_size} bytes)")
            
            return {
                "success": True,
                "file_path": output_path,
                "file_size": file_size,
                "generated_at": datetime.now().isoformat(),
                "generator": "WeasyPrint"
            }
            
        except Exception as e:
            logger.error(f"Error generating PDF: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": output_path,
                "generated_at": datetime.now().isoformat(),
                "generator": "WeasyPrint"
            }
    
    def _generate_pdf_sync(self, html_doc, css_stylesheet, output_path: str):
        """Synchronous PDF generation (run in thread)"""
        html_doc.write_pdf(
            output_path,
            stylesheets=[css_stylesheet],
            font_config=self.font_config,
            **self.pdf_options
        )
    
    async def _generate_fallback_pdf(
        self,
        html_content: str,
        output_path: str
    ) -> Dict[str, Any]:
        """
        Fallback PDF generation when WeasyPrint is not available
        Creates a simple text-based PDF using reportlab
        """
        try:
            # Try importing reportlab as fallback
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter, A4
            from reportlab.lib.styles import getSampleStyleSheet
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
            from reportlab.lib.units import inch
            
            # Create PDF document
            doc = SimpleDocTemplate(
                output_path,
                pagesize=A4,
                rightMargin=72, leftMargin=72,
                topMargin=72, bottomMargin=18
            )
            
            # Get styles
            styles = getSampleStyleSheet()
            story = []
            
            # Add simple content (basic HTML parsing)
            text_content = self._extract_text_from_html(html_content)
            
            # Add paragraphs
            for paragraph in text_content.split('\n\n'):
                if paragraph.strip():
                    story.append(Paragraph(paragraph.strip(), styles['Normal']))
                    story.append(Spacer(1, 12))
            
            # Build PDF
            await asyncio.to_thread(doc.build, story)
            
            file_size = os.path.getsize(output_path)
            
            logger.info(f"Fallback PDF generated: {output_path} ({file_size} bytes)")
            
            return {
                "success": True,
                "file_path": output_path,
                "file_size": file_size,
                "generated_at": datetime.now().isoformat(),
                "generator": "ReportLab (Fallback)"
            }
            
        except ImportError:
            # No PDF generation available
            return await self._create_text_file_fallback(html_content, output_path)
        except Exception as e:
            logger.error(f"Error in fallback PDF generation: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": output_path,
                "generated_at": datetime.now().isoformat(),
                "generator": "ReportLab (Fallback)"
            }
    
    async def _create_text_file_fallback(
        self,
        html_content: str,
        output_path: str
    ) -> Dict[str, Any]:
        """Ultimate fallback - create a text file instead of PDF"""
        try:
            text_content = self._extract_text_from_html(html_content)
            
            # Change extension to .txt
            txt_path = output_path.replace('.pdf', '.txt')
            
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write("DOCUMENT CONTENT\n")
                f.write("=" * 50 + "\n\n")
                f.write(text_content)
                f.write("\n\n" + "=" * 50)
                f.write("\nNote: This is a text version. PDF generation not available.")
            
            file_size = os.path.getsize(txt_path)
            
            logger.warning(f"Created text file fallback: {txt_path}")
            
            return {
                "success": True,
                "file_path": txt_path,
                "file_size": file_size,
                "generated_at": datetime.now().isoformat(),
                "generator": "Text Fallback",
                "warning": "PDF generation not available - created text file instead"
            }
            
        except Exception as e:
            logger.error(f"Error creating text fallback: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "file_path": output_path,
                "generated_at": datetime.now().isoformat(),
                "generator": "Text Fallback"
            }
    
    def _extract_text_from_html(self, html_content: str) -> str:
        """Extract plain text from HTML content"""
        try:
            from html import unescape
            import re
            
            # Remove script and style elements
            html_content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            html_content = re.sub(r'<style[^>]*>.*?</style>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            
            # Replace common HTML elements with text formatting
            html_content = re.sub(r'<br\s*/?>', '\n', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'<p[^>]*>', '\n\n', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'</p>', '', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'<h[1-6][^>]*>', '\n\n', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'</h[1-6]>', '\n', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'<li[^>]*>', '\n• ', html_content, flags=re.IGNORECASE)
            html_content = re.sub(r'</li>', '', html_content, flags=re.IGNORECASE)
            
            # Remove all other HTML tags
            html_content = re.sub(r'<[^>]+>', '', html_content)
            
            # Decode HTML entities
            text_content = unescape(html_content)
            
            # Clean up whitespace
            text_content = re.sub(r'\n\s*\n', '\n\n', text_content)
            text_content = re.sub(r' +', ' ', text_content)
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from HTML: {str(e)}")
            return html_content  # Return original if parsing fails
    
    async def validate_html_for_pdf(self, html_content: str) -> Dict[str, Any]:
        """Validate HTML content for PDF generation"""
        try:
            # Basic validation checks
            issues = []
            warnings = []
            
            # Check for basic HTML structure
            if '<html' not in html_content.lower():
                warnings.append("HTML content should include <html> tag")
            
            if '<body' not in html_content.lower():
                warnings.append("HTML content should include <body> tag")
            
            # Check for potential CSS issues
            if 'position: fixed' in html_content.lower():
                issues.append("Fixed positioning may not work well in PDF")
            
            if 'position: absolute' in html_content.lower():
                warnings.append("Absolute positioning may cause layout issues in PDF")
            
            # Check for unsupported elements
            unsupported_elements = ['video', 'audio', 'iframe', 'object', 'embed']
            for element in unsupported_elements:
                if f'<{element}' in html_content.lower():
                    issues.append(f"<{element}> elements are not supported in PDF")
            
            # Basic size estimation
            estimated_size = len(html_content) * 0.1  # Rough estimate
            
            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "warnings": warnings,
                "estimated_pdf_size": f"{estimated_size:.1f}KB",
                "html_size": len(html_content)
            }
            
        except Exception as e:
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "estimated_pdf_size": "Unknown",
                "html_size": len(html_content) if html_content else 0
            }
    
    async def get_pdf_info(self, pdf_path: str) -> Dict[str, Any]:
        """Get information about a generated PDF file"""
        try:
            if not os.path.exists(pdf_path):
                return {
                    "exists": False,
                    "error": "File not found"
                }
            
            file_stats = os.stat(pdf_path)
            
            return {
                "exists": True,
                "file_path": pdf_path,
                "file_size": file_stats.st_size,
                "created_at": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                "modified_at": datetime.fromtimestamp(file_stats.st_mtime).isoformat(),
                "readable": os.access(pdf_path, os.R_OK),
                "size_human": self._format_file_size(file_stats.st_size)
            }
            
        except Exception as e:
            return {
                "exists": False,
                "error": str(e),
                "file_path": pdf_path
            }
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        else:
            return f"{size_bytes / (1024 * 1024):.1f} MB"

# Singleton instance
pdf_service = PDFService()