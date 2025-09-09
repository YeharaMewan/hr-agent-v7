#!/usr/bin/env python3
"""
Quick test for document generation with existing employee
"""

import requests
import time
import json

def test_service_letter_generation():
    """Test service letter generation with an existing employee"""
    print("Testing service letter generation for Yehara...")
    
    try:
        response = requests.post(
            "http://localhost:8000/chat",
            json={"message": "generate service letter for Yehara"},
            timeout=30,
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
                
                # Limit content reading
                if len(content) > 1000:
                    break
            
            print(f"Response content: {content[:500]}...")
            
            # Check text content for success indicators
            if "service letter" in content.lower() and "generated" in content.lower():
                print("SUCCESS: Service letter generation appears successful")
                return True
            else:
                print("FAIL: Service letter generation unclear or failed")
                return False
                
        else:
            print(f"FAIL: Request failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR: Test failed: {str(e)}")
        return False

if __name__ == "__main__":
    result = test_service_letter_generation()
    print(f"Test result: {'PASS' if result else 'FAIL'}")