#!/usr/bin/env python3
"""
Test script to verify document generation fixes
Tests that the database schema fix resolves the document generation issues
"""

import requests
import time
import json
import sys

def test_employee_lookup():
    """Test employee lookup functionality"""
    print("Testing employee lookup after database fix...")
    
    try:
        # Test with a simple employee query first
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": "show me employee Hasitha Pathum"},
            timeout=30,
            stream=True
        )
        
        if response.status_code == 200:
            content = ""
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str and data_str != "[DONE]":
                        try:
                            data = json.loads(data_str)
                            if data.get("type") == "text":
                                content += data.get("content", "")
                        except json.JSONDecodeError:
                            continue
                
                if len(content) > 500:  # Limit response reading
                    break
            
            if "hasitha pathum" in content.lower():
                if "not found" in content.lower() or "error" in content.lower():
                    print("FAIL: Employee lookup still failing")
                    return False
                else:
                    print("PASS: Employee lookup working")
                    return True
            else:
                print("WARN: Unclear employee lookup response")
                return False
                
        else:
            print(f"FAIL: Employee lookup request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR: Employee lookup test error: {str(e)}")
        return False

def test_service_letter_generation():
    """Test service letter generation"""
    print("\nTesting service letter generation...")
    
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": "generate service letter for Hasitha Pathum"},
            timeout=45,
            stream=True
        )
        
        if response.status_code == 200:
            content = ""
            document_found = False
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str and data_str != "[DONE]":
                        try:
                            data = json.loads(data_str)
                            if data.get("type") == "document":
                                print("SUCCESS: Service letter generation successful!")
                                document_info = data.get("document_info", {})
                                print(f"   Document ID: {document_info.get('document_id', 'N/A')}")
                                print(f"   Filename: {document_info.get('filename', 'N/A')}")
                                print(f"   Employee: {document_info.get('employee_name', 'N/A')}")
                                return True
                            elif data.get("type") == "text":
                                content += data.get("content", "")
                        except json.JSONDecodeError:
                            continue
                
                if len(content) > 1000:  # Limit response reading
                    break
            
            # Check if success message in text content
            if "service letter" in content.lower() and "generated" in content.lower():
                print("SUCCESS: Service letter generation appears successful")
                return True
            elif "not found" in content.lower() or "error" in content.lower():
                print("FAIL: Service letter generation failed - employee lookup issue")
                print(f"   Response: {content[:200]}...")
                return False
            else:
                print("WARN: Service letter generation unclear")
                print(f"   Response: {content[:200]}...")
                return False
                
        else:
            print(f"FAIL: Service letter request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR: Service letter test error: {str(e)}")
        return False

def test_confirmation_letter_generation():
    """Test confirmation letter generation"""
    print("\nTesting confirmation letter generation...")
    
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": "generate confirmation letter for Hasitha Pathum"},
            timeout=45,
            stream=True
        )
        
        if response.status_code == 200:
            content = ""
            document_found = False
            
            for line in response.iter_lines(decode_unicode=True):
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str and data_str != "[DONE]":
                        try:
                            data = json.loads(data_str)
                            if data.get("type") == "document":
                                print("SUCCESS: Confirmation letter generation successful!")
                                document_info = data.get("document_info", {})
                                print(f"   Document ID: {document_info.get('document_id', 'N/A')}")
                                print(f"   Filename: {document_info.get('filename', 'N/A')}")
                                print(f"   Employee: {document_info.get('employee_name', 'N/A')}")
                                return True
                            elif data.get("type") == "text":
                                content += data.get("content", "")
                        except json.JSONDecodeError:
                            continue
                
                if len(content) > 1000:  # Limit response reading
                    break
            
            # Check if success message in text content
            if "confirmation letter" in content.lower() and "generated" in content.lower():
                print("SUCCESS: Confirmation letter generation appears successful")
                return True
            elif "not found" in content.lower() or "error" in content.lower():
                print("FAIL: Confirmation letter generation failed - employee lookup issue")
                print(f"   Response: {content[:200]}...")
                return False
            else:
                print("WARN: Confirmation letter generation unclear")
                print(f"   Response: {content[:200]}...")
                return False
                
        else:
            print(f"FAIL: Confirmation letter request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR: Confirmation letter test error: {str(e)}")
        return False

def test_health_check():
    """Test health check shows dependencies are working"""
    print("\nTesting health check...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=10)
        
        if response.status_code == 200:
            health_data = response.json()
            print(f"PASS: Health status: {health_data.get('status', 'unknown')}")
            
            deps = health_data.get('dependencies', {})
            all_good = True
            for dep, status in deps.items():
                if "missing" in status.lower():
                    print(f"FAIL: {dep}: {status}")
                    all_good = False
                else:
                    print(f"PASS: {dep}: {status}")
            
            return all_good
        else:
            print(f"FAIL: Health check failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR: Health check error: {str(e)}")
        return False

def wait_for_container(max_wait=30):
    """Wait for container to be ready"""
    print(f"Waiting for container to be ready...")
    
    for i in range(max_wait):
        try:
            response = requests.get("http://localhost:8000/health", timeout=5)
            if response.status_code == 200:
                print(f"Container ready")
                return True
        except:
            pass
        
        time.sleep(1)
    
    print(f"ERROR: Container not ready after {max_wait}s")
    return False

def run_document_fix_tests():
    """Run all document generation fix tests"""
    print("Document Generation Fix Verification")
    print("=" * 50)
    
    # Wait for container
    if not wait_for_container():
        return False
    
    # Test health
    health_ok = test_health_check()
    
    # Test employee lookup
    lookup_ok = test_employee_lookup()
    
    # Test document generation
    service_ok = test_service_letter_generation()
    confirmation_ok = test_confirmation_letter_generation()
    
    # Summary
    print("\n" + "=" * 50)
    print("DOCUMENT GENERATION FIX TEST SUMMARY")
    print("=" * 50)
    
    print(f"Health Check: {'PASS' if health_ok else 'FAIL'}")
    print(f"Employee Lookup: {'PASS' if lookup_ok else 'FAIL'}")
    print(f"Service Letter: {'PASS' if service_ok else 'FAIL'}")
    print(f"Confirmation Letter: {'PASS' if confirmation_ok else 'FAIL'}")
    
    overall_success = health_ok and lookup_ok and service_ok and confirmation_ok
    
    print("=" * 50)
    if overall_success:
        print("ALL TESTS PASSED!")
        print("\nThe database schema fix successfully resolved the issues:")
        print("PASS: Employee lookup works correctly")
        print("PASS: Service letter generation functional")
        print("PASS: Confirmation letter generation functional")
        print("PASS: No more 'column does not exist' errors")
        print("\nDocument generation system is fully operational!")
    else:
        print("SOME TESTS FAILED!")
        print("\nRemaining issues:")
        if not health_ok:
            print("FAIL: Health check shows missing dependencies")
        if not lookup_ok:
            print("FAIL: Employee lookup still failing")
        if not service_ok:
            print("FAIL: Service letter generation not working")
        if not confirmation_ok:
            print("FAIL: Confirmation letter generation not working")
        
        print("\nAdditional fixes may be needed")
    
    return overall_success

if __name__ == "__main__":
    success = run_document_fix_tests()
    sys.exit(0 if success else 1)