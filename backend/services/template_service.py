"""
Template Service - Manages HTML/CSS templates for document generation
Handles template loading, rendering, and template data injection.
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging
from jinja2 import Environment, FileSystemLoader, Template, TemplateNotFound
import asyncio

logger = logging.getLogger(__name__)

class TemplateService:
    """Service for managing document templates"""
    
    def __init__(self):
        self.templates_dir = Path("backend/templates")
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Jinja2 environment
        self.jinja_env = Environment(
            loader=FileSystemLoader(str(self.templates_dir)),
            autoescape=True,
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Add custom filters
        self.jinja_env.filters['format_date'] = self._format_date_filter
        self.jinja_env.filters['title_case'] = self._title_case_filter
        self.jinja_env.filters['upper'] = lambda x: str(x).upper()
        self.jinja_env.filters['lower'] = lambda x: str(x).lower()
        
    async def render_template(
        self, 
        template_name: str, 
        data: Dict[str, Any]
    ) -> str:
        """
        Render a template with provided data
        
        Args:
            template_name: Name of the template (without .html extension)
            data: Data to inject into the template
            
        Returns:
            Rendered HTML string
        """
        try:
            template_file = f"{template_name}.html"
            logger.info(f"Rendering template: {template_file}")
            
            template = self.jinja_env.get_template(template_file)
            rendered_html = template.render(data)
            
            logger.info(f"Template rendered successfully: {template_file}")
            return rendered_html
            
        except TemplateNotFound as e:
            logger.error(f"Template not found: {template_file}")
            # Return a fallback template if main template is not found
            return await self._render_fallback_template(template_name, data)
            
        except Exception as e:
            logger.error(f"Error rendering template {template_file}: {str(e)}")
            return await self._render_error_template(template_name, data, str(e))
    
    async def _render_fallback_template(
        self, 
        template_name: str, 
        data: Dict[str, Any]
    ) -> str:
        """Render a fallback template when the main template is not found"""
        
        fallback_html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{{ letter_title | default('Official Letter') }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; color: #333; }
                .header { text-align: center; margin-bottom: 40px; }
                .company-name { font-size: 24px; font-weight: bold; color: #2c5aa0; }
                .content { line-height: 1.6; }
                .footer { margin-top: 40px; text-align: center; font-size: 12px; color: #666; }
            </style>
        </head>
        <body>
            <div class="header">
                <div class="company-name">{{ company_name | default('Rise Tech Village') }}</div>
                <p>{{ company_address | default('') }}</p>
            </div>
            
            <div class="content">
                <p><strong>Date:</strong> {{ current_date }}</p>
                
                <h2>{{ letter_title | default('Official Letter') }}</h2>
                
                <p>Dear {{ name | default('Employee') }},</p>
                
                <p>This letter serves as confirmation of your employment with our organization.</p>
                
                <p><strong>Employee Details:</strong></p>
                <ul>
                    <li><strong>Name:</strong> {{ name | default('N/A') }}</li>
                    <li><strong>Position:</strong> {{ formatted_role | default('Employee') }}</li>
                    <li><strong>Department:</strong> {{ department_name | default('N/A') }}</li>
                    <li><strong>Email:</strong> {{ email | default('N/A') }}</li>
                </ul>
                
                <p>If you have any questions, please contact our HR department.</p>
                
                <p>Sincerely,</p>
                <p><strong>HR Department</strong><br>
                {{ company_name | default('Rise Tech Village') }}</p>
            </div>
            
            <div class="footer">
                <p>This is an automatically generated document.</p>
            </div>
        </body>
        </html>
        """
        
        template = Template(fallback_html)
        return template.render(data)
    
    async def _render_error_template(
        self, 
        template_name: str, 
        data: Dict[str, Any], 
        error: str
    ) -> str:
        """Render an error template when rendering fails"""
        
        error_html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <title>Template Error</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .error {{ background: #ffe6e6; padding: 20px; border: 1px solid #ff9999; }}
            </style>
        </head>
        <body>
            <div class="error">
                <h2>Template Rendering Error</h2>
                <p><strong>Template:</strong> {template_name}</p>
                <p><strong>Error:</strong> {error}</p>
                <p>Please contact system administrator.</p>
            </div>
        </body>
        </html>
        """
        
        return error_html
    
    def _format_date_filter(self, date_value, format_string="%B %d, %Y"):
        """Custom Jinja2 filter for date formatting"""
        if not date_value:
            return "N/A"
        
        try:
            from datetime import datetime
            if isinstance(date_value, str):
                # Try to parse string date
                date_obj = datetime.fromisoformat(date_value.replace('Z', '+00:00'))
            elif hasattr(date_value, 'strftime'):
                date_obj = date_value
            else:
                return str(date_value)
            
            return date_obj.strftime(format_string)
        except:
            return str(date_value)
    
    def _title_case_filter(self, value):
        """Custom Jinja2 filter for title case conversion"""
        if not value:
            return ""
        return str(value).replace('_', ' ').title()
    
    async def get_available_templates(self) -> Dict[str, Dict[str, Any]]:
        """Get list of available templates with their metadata"""
        templates = {}
        
        try:
            for template_file in self.templates_dir.glob("*.html"):
                template_name = template_file.stem
                template_info = await self._get_template_info(template_name)
                templates[template_name] = template_info
                
        except Exception as e:
            logger.error(f"Error getting available templates: {str(e)}")
        
        return templates
    
    async def _get_template_info(self, template_name: str) -> Dict[str, Any]:
        """Get metadata about a template"""
        template_file = self.templates_dir / f"{template_name}.html"
        
        info = {
            "name": template_name,
            "file_path": str(template_file),
            "exists": template_file.exists(),
            "size": template_file.stat().st_size if template_file.exists() else 0,
            "modified": template_file.stat().st_mtime if template_file.exists() else 0,
            "description": self._get_template_description(template_name)
        }
        
        return info
    
    def _get_template_description(self, template_name: str) -> str:
        """Get human-readable description of template"""
        descriptions = {
            "service_letter": "Official service letter for employees",
            "confirmation_letter": "Employment confirmation letter",
            "offer_letter": "Job offer letter for new employees",
            "reference_letter": "Professional reference letter",
            "experience_certificate": "Work experience certificate"
        }
        
        return descriptions.get(template_name, f"Template: {template_name}")
    
    async def validate_template(self, template_name: str) -> Dict[str, Any]:
        """Validate a template for syntax errors"""
        try:
            template_file = f"{template_name}.html"
            template = self.jinja_env.get_template(template_file)
            
            # Try rendering with sample data
            sample_data = await self._get_sample_template_data()
            rendered = template.render(sample_data)
            
            return {
                "valid": True,
                "template_name": template_name,
                "message": "Template is valid",
                "rendered_size": len(rendered)
            }
            
        except TemplateNotFound:
            return {
                "valid": False,
                "template_name": template_name,
                "error": "Template file not found",
                "message": f"Template {template_name}.html does not exist"
            }
            
        except Exception as e:
            return {
                "valid": False,
                "template_name": template_name,
                "error": str(e),
                "message": f"Template validation failed: {str(e)}"
            }
    
    async def _get_sample_template_data(self) -> Dict[str, Any]:
        """Get sample data for template validation"""
        from datetime import datetime
        
        return {
            "name": "John Doe",
            "email": "john.doe@example.com",
            "role": "employee",
            "formatted_role": "Employee",
            "department_name": "Engineering",
            "department_description": "Software Development",
            "phone_number": "+1 (555) 123-4567",
            "address": "123 Main St, City, State 12345",
            "joining_date": "January 15, 2023",
            "current_date": datetime.now().strftime("%B %d, %Y"),
            "company_name": "Rise Tech Village",
            "company_address": "123 Tech Street, Innovation City, IC 12345",
            "company_phone": "+1 (555) 123-4567",
            "company_email": "hr@risetechvillage.com",
            "letter_title": "Sample Letter",
            "subject": "Sample Subject",
            "service_duration": "2 years and 3 months",
            "confirmation_date": datetime.now().strftime("%B %d, %Y"),
            "probation_period": "3 months",
            "purpose": "Sample Purpose"
        }
    
    async def create_template_from_string(
        self, 
        template_name: str, 
        template_content: str
    ) -> Dict[str, Any]:
        """Create a new template from string content"""
        try:
            template_file = self.templates_dir / f"{template_name}.html"
            
            # Write template content to file
            with open(template_file, 'w', encoding='utf-8') as f:
                f.write(template_content)
            
            # Validate the new template
            validation_result = await self.validate_template(template_name)
            
            if validation_result["valid"]:
                logger.info(f"Template created successfully: {template_name}")
                return {
                    "success": True,
                    "template_name": template_name,
                    "file_path": str(template_file),
                    "message": "Template created and validated successfully"
                }
            else:
                # Remove invalid template file
                template_file.unlink(missing_ok=True)
                return {
                    "success": False,
                    "template_name": template_name,
                    "error": validation_result["error"],
                    "message": "Template creation failed validation"
                }
                
        except Exception as e:
            logger.error(f"Error creating template {template_name}: {str(e)}")
            return {
                "success": False,
                "template_name": template_name,
                "error": str(e),
                "message": f"Failed to create template: {str(e)}"
            }

# Singleton instance
template_service = TemplateService()