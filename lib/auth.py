"""
Authentication utilities for TasteLens API
Handles password hashing, JWT token creation/verification
"""
import os

from datetime import datetime, timedelta, timezone
from typing import Optional, Union
import hashlib
import secrets

from passlib.context import CryptContext

from jose import JWTError, jwt
from fastapi import HTTPException, status

# Security configuration
SECRET_KEY = os.environ["SECRET_KEY"]
ALGORITHM = os.environ["HASH_ALGORITHM"]
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Password hashing context - fallback to PBKDF2 if bcrypt fails
pwd_context = CryptContext(
    schemes=["bcrypt"], 
    deprecated="auto",
    bcrypt__rounds=12
)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against its hash using SHA-256 pre-hashing + bcrypt
    
    Args:
        plain_password: The plain text password
        hashed_password: The hashed password from database
        
    Returns:
        True if password matches, False otherwise
    """
    # Pre-hash with SHA-256 to normalize length and entropy
    pre_hash = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
    
    return pwd_context.verify(pre_hash, hashed_password)

def get_password_hash(password: str) -> str:
    """
    Hash a password for storing in database using SHA-256 pre-hashing + bcrypt
    
    Args:
        password: Plain text password (no length limitations)
        
    Returns:
        Hashed password string
    """
    # Pre-hash with SHA-256 to normalize length and entropy
    # This ensures consistent input length for bcrypt and handles arbitrarily long passwords
    pre_hash = hashlib.sha256(password.encode('utf-8')).hexdigest()
    
    return pwd_context.hash(pre_hash)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create JWT access token
    
    Args:
        data: Data to encode in token (typically user info)
        expires_delta: Token expiration time, defaults to ACCESS_TOKEN_EXPIRE_MINUTES
        
    Returns:
        JWT token string
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def verify_token(token: str) -> Optional[dict]:
    """
    Verify and decode JWT token
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload if valid, None if invalid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        return None

def authenticate_user(username: str, password: str, get_user_func) -> Union[dict, bool]:
    """
    Authenticate a user with username and password
    
    Args:
        username: User's username
        password: Plain text password
        get_user_func: Function to get user from database
        
    Returns:
        User dict if authentication successful, False otherwise
    """
    user = get_user_func(username)
    if not user:
        return False
    if not verify_password(password, user.password_hash):
        return False
    return user

class AuthenticationError(HTTPException):
    """Custom authentication error"""
    def __init__(self, detail: str = "Could not validate credentials"):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )

class UserExistsError(HTTPException):
    """User already exists error"""
    def __init__(self, detail: str = "User already exists"):
        super().__init__(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )