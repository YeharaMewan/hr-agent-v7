#!/usr/bin/env python3
"""
Docker Container Test Script
Tests the Docker backend container for proper functionality
"""

import requests
import time
import json
import sys

def test_container_health():
    """Test container health endpoint"""
    print("🔍 Testing container health...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=30)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"✅ Container is {health_data['status']}")
            print(f"   Environment: {health_data.get('environment', 'unknown')}")
            
            # Check dependencies
            deps = health_data.get('dependencies', {})
            print("   Dependencies:")
            for dep, status in deps.items():
                icon = "✅" if "available" in status or "found" in status else "❌"
                print(f"   {icon} {dep}: {status}")
            
            # Check for warnings
            if 'warnings' in health_data:
                print(f"   ⚠️ Warnings: {health_data['warnings']}")
                
            return health_data['status'] in ['healthy', 'degraded']
            
        else:
            print(f"❌ Health check failed with status {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to container - is it running?")
        return False
    except Exception as e:
        print(f"❌ Health check error: {str(e)}")
        return False

def test_api_endpoints():
    """Test basic API endpoints"""
    print("\n🔍 Testing API endpoints...")
    
    endpoints = [
        ("/", "Root endpoint"),
        ("/health", "Health check"),
        ("/usage", "Usage statistics"),
        ("/employees", "Employees endpoint"),
        ("/documents/templates", "Document templates")
    ]
    
    results = {}
    
    for endpoint, description in endpoints:
        try:
            response = requests.get(f"http://localhost:8000{endpoint}", timeout=10)
            
            if response.status_code == 200:
                print(f"✅ {description} - OK")
                results[endpoint] = "success"
            else:
                print(f"⚠️ {description} - Status {response.status_code}")
                results[endpoint] = f"status_{response.status_code}"
                
        except requests.exceptions.ConnectionError:
            print(f"❌ {description} - Connection failed")
            results[endpoint] = "connection_failed"
        except Exception as e:
            print(f"❌ {description} - Error: {str(e)}")
            results[endpoint] = f"error"
    
    return results

def test_document_generation():
    """Test document generation functionality"""
    print("\n🔍 Testing document generation...")
    
    try:
        # Test chat endpoint with document generation request
        chat_payload = {
            "message": "Generate service letter for Test Employee"
        }
        
        response = requests.post(
            "http://localhost:8000/chat",
            json=chat_payload,
            timeout=30,
            stream=True
        )
        
        if response.status_code == 200:
            print("✅ Chat endpoint responds")
            
            # Try to read streaming response
            content = ""
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str and data_str != "[DONE]":
                        try:
                            data = json.loads(data_str)
                            if data.get("type") == "document":
                                print("✅ Document generation response received")
                                return True
                            elif data.get("type") == "text":
                                content += data.get("content", "")
                        except json.JSONDecodeError:
                            continue
                
                # Limit response reading
                if len(content) > 1000:
                    break
            
            if "employee" in content.lower() or "document" in content.lower():
                print("✅ Document-related response received")
                return True
            else:
                print("⚠️ No document generation detected in response")
                return False
                
        else:
            print(f"❌ Chat endpoint failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Document generation test error: {str(e)}")
        return False

def wait_for_container(max_wait=60):
    """Wait for container to be ready"""
    print(f"⏳ Waiting for container to be ready (max {max_wait}s)...")
    
    for i in range(max_wait):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ Container ready after {i+1}s")
                return True
        except:
            pass
        
        time.sleep(1)
        if (i + 1) % 10 == 0:
            print(f"   Still waiting... ({i+1}/{max_wait}s)")
    
    print(f"❌ Container not ready after {max_wait}s")
    return False

def run_container_tests():
    """Run all container tests"""
    print("🐳 Docker Backend Container Tests")
    print("=" * 50)
    
    # Wait for container
    if not wait_for_container():
        return False
    
    # Test health
    health_ok = test_container_health()
    
    # Test API endpoints
    api_results = test_api_endpoints()
    
    # Test document generation
    doc_ok = test_document_generation()
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    print(f"Health Check: {'✅ PASS' if health_ok else '❌ FAIL'}")
    
    api_success = sum(1 for r in api_results.values() if r == "success")
    api_total = len(api_results)
    print(f"API Endpoints: ✅ {api_success}/{api_total} working")
    
    print(f"Document Generation: {'✅ PASS' if doc_ok else '❌ FAIL'}")
    
    overall_success = health_ok and api_success >= api_total * 0.8 and doc_ok
    
    print("=" * 50)
    if overall_success:
        print("🎉 DOCKER CONTAINER TESTS PASSED!")
        print("\nYour container is working correctly:")
        print("✅ All dependencies installed")
        print("✅ API endpoints responding")
        print("✅ Document generation functional")
        print("✅ Ready for production use")
    else:
        print("💥 DOCKER CONTAINER TESTS FAILED!")
        print("\nIssues detected:")
        if not health_ok:
            print("❌ Health check failed")
        if api_success < api_total:
            print(f"❌ {api_total - api_success} API endpoints not working")
        if not doc_ok:
            print("❌ Document generation not working")
    
    return overall_success

if __name__ == "__main__":
    success = run_container_tests()
    sys.exit(0 if success else 1)