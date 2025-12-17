"""Test script to debug user registration"""
import sys
sys.path.insert(0, r'c:\Users\dat\Projects\RAG')

from src.auth import add_user
import yaml
from pathlib import Path

# Test adding a user
print("Testing add_user function...")
result = add_user('testuser', 'Test User', 'test@example.com', 'testpass123')
print(f"Registration result: {result}")

# Check the file
users_file = Path(r'c:\Users\dat\Projects\RAG\data\users.yaml')
print(f"\nFile exists: {users_file.exists()}")

if users_file.exists():
    with open(users_file, 'r') as f:
        content = yaml.safe_load(f)
    print(f"File contents: {content}")
    print(f"Number of users: {len(content.get('credentials', {}).get('usernames', {}))}")
