#!/usr/bin/env python3
"""
Simple test script for authentication endpoints
Run this after installing dependencies and starting the server
"""

import requests
import json
import random


BASE_URL = "http://localhost:8000"

num = f"{random.randint(0, 99999999):08d}"
user_data = {
    "username": f"testuser{num}",
    "email": f"test{num}@example.com",
    "password": "testpassword123",
    "first_name": "Test",
    "last_name": "User",
    "display_name": "Test User"
}

def test_registration():
    """Test user registration"""
    print("🧪 Testing user registration...")
    response = requests.post(f"{BASE_URL}/register", json=user_data)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Registration successful!")
        print(f"   User ID: {data['user']['id']}")
        print(f"   Username: {data['user']['username']}")
        print(f"   Token: {data['token']['access_token'][:20]}...")
        return data['token']['access_token']
    else:
        print(f"❌ Registration failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_login():
    """Test user login"""
    print("\n🧪 Testing user login...")
    
    login_data = {
        "username": user_data["username"],
        "password": user_data["password"]
    }
    
    response = requests.post(f"{BASE_URL}/login", json=login_data)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Login successful!")
        print(f"   User ID: {data['user']['id']}")
        print(f"   Username: {data['user']['username']}")
        print(f"   Token: {data['token']['access_token'][:20]}...")
        return data['token']['access_token']
    else:
        print(f"❌ Login failed: {response.status_code}")
        print(f"   Error: {response.text}")
        return None

def test_protected_endpoint(token):
    """Test accessing protected endpoint with token"""
    print("\n🧪 Testing protected endpoint...")
    
    headers = {
        "Authorization": f"Bearer {token}"
    }
    
    response = requests.get(f"{BASE_URL}/me", headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print("✅ Protected endpoint access successful!")
        print(f"   User: {data['username']} ({data['email']})")
        print(f"   Created: {data['created_at']}")
    else:
        print(f"❌ Protected endpoint access failed: {response.status_code}")
        print(f"   Error: {response.text}")

def test_invalid_login():
    """Test login with invalid credentials"""
    print("\n🧪 Testing invalid login...")

    login_data = {
        "username": user_data["username"],
        "password": "wrongpassword"
    }
    
    response = requests.post(f"{BASE_URL}/login", json=login_data)
    
    if response.status_code == 401:
        print("✅ Invalid login properly rejected!")
    else:
        print(f"❌ Invalid login should return 401, got {response.status_code}")

def main():
    """Run all tests"""
    print("🚀 Starting authentication tests...\n")
    
    try:
        # Test registration
        token = test_registration()
        
        if token:
            # Test protected endpoint
            test_protected_endpoint(token)
        
        # Test login (should work even if registration failed due to existing user)
        token = test_login()
        
        if token:
            # Test protected endpoint with login token
            test_protected_endpoint(token)
        
        # Test invalid login
        test_invalid_login()
        
        print("\n🎉 All tests completed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server. Make sure the FastAPI server is running on http://localhost:8000")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()