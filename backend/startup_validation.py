"""
Startup validation module for Docker container
Validates dependencies and creates necessary directories
"""

import os
import sys
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

def validate_weasyprint():
    """Validate WeasyPrint installation and dependencies"""
    try:
        import weasyprint
        logger.info("✅ WeasyPrint imported successfully")
        
        # Test basic functionality
        html_content = "<html><body><h1>Test</h1></body></html>"
        weasyprint.HTML(string=html_content)
        logger.info("✅ WeasyPrint basic functionality verified")
        
        return True
    except ImportError as e:
        logger.error(f"❌ WeasyPrint import failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ WeasyPrint functionality test failed: {str(e)}")
        return False

def validate_template_system():
    """Validate template system dependencies"""
    try:
        import jinja2
        logger.info("✅ Jinja2 template engine available")
        
        # Test basic template rendering
        template = jinja2.Template("Hello {{ name }}!")
        result = template.render(name="World")
        assert result == "Hello World!"
        logger.info("✅ Jinja2 template rendering verified")
        
        return True
    except ImportError as e:
        logger.error(f"❌ Jinja2 import failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Jinja2 functionality test failed: {str(e)}")
        return False

def create_required_directories():
    """Create all required directories for the application"""
    directories = [
        "storage/documents",
        "templates",
        "logs"
    ]
    
    success = True
    for directory in directories:
        try:
            dir_path = Path(directory)
            dir_path.mkdir(parents=True, exist_ok=True)
            
            # Set proper permissions
            os.chmod(str(dir_path), 0o755)
            logger.info(f"✅ Directory created/verified: {directory}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {str(e)}")
            success = False
    
    return success

def validate_environment_variables():
    """Validate required environment variables"""
    required_vars = [
        "OPENAI_API_KEY"
    ]
    
    optional_vars = [
        "OPENAI_MODEL",
        "OPENAI_TEMPERATURE",
        "LANGSMITH_TRACING"
    ]
    
    missing_required = []
    for var in required_vars:
        if not os.getenv(var):
            missing_required.append(var)
    
    if missing_required:
        logger.warning(f"⚠️ Missing required environment variables: {missing_required}")
        logger.warning("Document generation may not work without proper API configuration")
    else:
        logger.info("✅ All required environment variables present")
    
    # Log optional variables status
    for var in optional_vars:
        value = os.getenv(var)
        if value:
            # Don't log full API keys for security
            if "API_KEY" in var:
                logger.info(f"✅ {var} configured (***{value[-4:]})")
            else:
                logger.info(f"✅ {var} = {value}")
        else:
            logger.info(f"ℹ️ {var} not set (using default)")
    
    return len(missing_required) == 0

def validate_file_permissions():
    """Validate file system permissions"""
    test_file = Path("storage/documents/permission_test.txt")
    
    try:
        # Test write permission
        with open(test_file, "w") as f:
            f.write("permission test")
        
        # Test read permission
        with open(test_file, "r") as f:
            content = f.read()
        
        assert content == "permission test"
        
        # Clean up test file
        test_file.unlink()
        
        logger.info("✅ File system permissions verified")
        return True
        
    except Exception as e:
        logger.error(f"❌ File permission test failed: {str(e)}")
        return False

def validate_python_path():
    """Validate Python path and module imports"""
    try:
        # Test relative imports
        current_dir = Path.cwd()
        logger.info(f"Current working directory: {current_dir}")
        
        # Check if we can import our modules
        sys.path.insert(0, str(current_dir))
        
        # Test imports
        import database
        logger.info("✅ Database module import successful")
        
        import models
        logger.info("✅ Models module import successful")
        
        from services import document_service
        logger.info("✅ Services module import successful")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Module import failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"❌ Python path validation failed: {str(e)}")
        return False

def run_startup_validation():
    """Run all startup validations"""
    logger.info("🚀 Starting Docker container validation...")
    
    validations = [
        ("Environment Variables", validate_environment_variables),
        ("Directory Creation", create_required_directories), 
        ("File Permissions", validate_file_permissions),
        ("Python Path & Imports", validate_python_path),
        ("Template System", validate_template_system),
        ("WeasyPrint Dependencies", validate_weasyprint),
    ]
    
    results = {}
    for name, validation_func in validations:
        logger.info(f"Validating: {name}")
        try:
            result = validation_func()
            results[name] = result
        except Exception as e:
            logger.error(f"❌ Validation {name} failed with exception: {str(e)}")
            results[name] = False
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    logger.info("=" * 50)
    logger.info("🔍 STARTUP VALIDATION SUMMARY")
    logger.info("=" * 50)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {name}")
    
    logger.info("=" * 50)
    logger.info(f"Result: {passed}/{total} validations passed")
    
    if passed == total:
        logger.info("🎉 All validations passed! Container is ready.")
        return True
    else:
        logger.error(f"💥 {total - passed} validations failed. Check logs above.")
        return False

if __name__ == "__main__":
    success = run_startup_validation()
    sys.exit(0 if success else 1)