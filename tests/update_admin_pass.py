"""Update admin password hash in users.yaml"""
import streamlit_authenticator as stauth
import yaml
from pathlib import Path

# Generate new hash for admin123
password = "admin123"
hashed = stauth.Hasher.hash(password)

print(f"Generated hash for '{password}': {hashed}")

# Load current users.yaml
users_file = Path(r'c:\Users\dat\Projects\RAG\data\users.yaml')
with open(users_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Update admin password
config['credentials']['usernames']['admin']['password'] = hashed

# Save back
with open(users_file, 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

print(f"Updated admin password in {users_file}")
print("New password: admin123")
