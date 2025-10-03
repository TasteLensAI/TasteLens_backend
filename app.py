from fastapi import FastAPI, APIRouter, Query, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional
from datetime import timedelta

from lib.interface import (
    MoviesResponse, MovieLensInterface, GenresResponse,
    UserCreate, UserLogin, UserResponse, LoginResponse, User, Token
)
from lib.sql.agents import SQLiteAgent
from lib.auth import (
    get_password_hash, authenticate_user, create_access_token, 
    verify_token, AuthenticationError, UserExistsError,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

# Create FastAPI app
app = FastAPI(
    title="TasteLens API",
    description="A movie recommendation API",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

router = APIRouter()
security = HTTPBearer()

DATABASE_NAME = "tastelens"

# Dependency to get current user from token
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """
    Get current user from JWT token
    """
    token = credentials.credentials
    payload = verify_token(token)
    
    if payload is None:
        raise AuthenticationError()
    
    username = payload.get("sub")
    if username is None:
        raise AuthenticationError()
    
    # Get user from database
    with SQLiteAgent(source_args={"name": DATABASE_NAME}) as agent:
        interface = MovieLensInterface(agent=agent)
        user = interface.get_user_by_username(username)
        
        if user is None:
            raise AuthenticationError()
        
        return User(
            id=user.id,
            username=user.username,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            display_name=user.display_name,
            bio=user.bio,
            avatar_url=user.avatar_url,
            is_verified=user.is_verified,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login=user.last_login
        )

@router.get("/movies", response_model=MoviesResponse)
async def get_movies(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(10, ge=1, le=50, description="Movies per page"),
    genres: Optional[List[str]] = Query(None, description="List of genres (movies must have ALL selected genres)"),
    search: Optional[str] = None
):
    # Calculate offset
    offset = (page - 1) * limit
    
    with SQLiteAgent(source_args={"name": DATABASE_NAME}) as agent:
        interface = MovieLensInterface(agent=agent)

        movies = interface.get_movies(offset=offset, limit=limit, genres=genres, search=search)
        total = interface.count_movies(genres=genres, search=search)
    
    
    has_next = offset + limit < total
    
    return MoviesResponse(
        movies=movies,
        total=total,
        page=page,
        limit=limit,
        has_next=has_next
    )

@router.get("/genres", response_model=GenresResponse)
async def get_genres():
    """
    Get all unique genres with their movie counts and total movie count
    """
    with SQLiteAgent(source_args={"name": DATABASE_NAME}) as agent:
        interface = MovieLensInterface(agent=agent)
        stats = interface.get_genre_statistics()
    
    return GenresResponse(
        genres=stats["genres"],
        total_movies=stats["total_movies"]
    )

@router.post("/register", response_model=UserResponse)
async def register_user(user_data: UserCreate):
    """
    Register a new user
    """
    with SQLiteAgent(source_args={"name": DATABASE_NAME}) as agent:
        interface = MovieLensInterface(agent=agent)
        
        # Check if user already exists
        if interface.user_exists(username=user_data.username):
            raise UserExistsError("Username already exists")
        
        if interface.user_exists(email=user_data.email):
            raise UserExistsError("Email already exists")
        
        # Hash password and create user
        password_hash = get_password_hash(user_data.password)
        
        try:
            user = interface.create_user(user_data, password_hash)
            
            # Create access token
            access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = create_access_token(
                data={"sub": user.username}, 
                expires_delta=access_token_expires
            )
            
            token = Token(access_token=access_token, token_type="bearer")
            
            return UserResponse(user=user, token=token)
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create user: {str(e)}"
            )

@router.post("/login", response_model=LoginResponse)
async def login_user(login_data: UserLogin):
    """
    Login user and return access token
    """
    with SQLiteAgent(source_args={"name": DATABASE_NAME}) as agent:
        interface = MovieLensInterface(agent=agent)
        
        # Authenticate user
        user = authenticate_user(
            login_data.username, 
            login_data.password, 
            interface.get_user_by_username
        )
        
        if not user:
            raise AuthenticationError("Incorrect username or password")
        
        # Update last login
        interface.update_user_last_login(user.id)
        
        # Create access token
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, 
            expires_delta=access_token_expires
        )
        
        token = Token(access_token=access_token, token_type="bearer")
        
        # Convert to Pydantic User model
        user_model = User(
            id=user.id,
            username=user.username,
            email=user.email,
            first_name=user.first_name,
            last_name=user.last_name,
            display_name=user.display_name,
            bio=user.bio,
            avatar_url=user.avatar_url,
            is_verified=user.is_verified,
            created_at=user.created_at,
            updated_at=user.updated_at,
            last_login=user.last_login
        )
        
        return LoginResponse(user=user_model, token=token)

@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information (requires authentication)
    """
    return current_user

# Include the router in the app
app.include_router(router)