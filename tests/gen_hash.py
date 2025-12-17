"""Generate password hash for admin user"""
import streamlit_authenticator as stauth

# Generate hash for admin123
password = "admin123"
hashed = stauth.Hasher.hash(password)
print("=" * 80)
print(f"Password: {password}")
print(f"Hashed: {hashed}")
print("=" * 80)
