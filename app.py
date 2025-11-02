import os

from fastapi import FastAPI, APIRouter, Query, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import List, Optional
from datetime import timedelta

from lib.interface import (
    MoviesResponse, MovieLensInterface, GenresResponse,
    UserCreate, UserLogin, UserResponse, LoginResponse, User, Token
)
from lib.sql.agents import SQLiteAgent, PostgreSQLAgent
from lib.auth import (
    get_password_hash, authenticate_user, create_access_token, 
    verify_token, AuthenticationError, UserExistsError,
    ACCESS_TOKEN_EXPIRE_MINUTES
)

from dotenv import load_dotenv
load_dotenv()

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

# DATABASE_NAME = "tastelens"
# source_args = {
#     "name": "tastelens"
# }

source_args = {
    "host": os.getenv("POSTGRES_HOST"),
    "port": os.getenv("POSTGRES_PORT"),
    "user": os.getenv("POSTGRES_USER")
}

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
    with PostgreSQLAgent(source_args=source_args) as agent:
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
    
    with PostgreSQLAgent(source_args=source_args) as agent:
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
    with PostgreSQLAgent(source_args=source_args) as agent:
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
    with PostgreSQLAgent(source_args=source_args) as agent:
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
    with PostgreSQLAgent(source_args=source_args) as agent:
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

@router.post("/wishlist/add")
async def add_to_wishlist(
    movieId: int = Query(..., description="Movie ID to add to wishlist"),
    current_user: User = Depends(get_current_user)
):
    """
    Add a movie to user's wishlist
    """
    with PostgreSQLAgent(source_args=source_args) as agent:
        interface = MovieLensInterface(agent=agent)
        
        # Add to wishlist
        success = interface.add_to_wishlist(current_user.id, movieId)
        
        if success:
            return {"message": "Movie added to wishlist", "movieId": movieId}
        else:
            return {"message": "Movie already in wishlist", "movieId": movieId}

@router.post("/wishlist/remove")
async def remove_from_wishlist(
    movieId: int = Query(..., description="Movie ID to remove from wishlist"),
    current_user: User = Depends(get_current_user)
):
    """
    Remove a movie from user's wishlist
    """
    with PostgreSQLAgent(source_args=source_args) as agent:
        interface = MovieLensInterface(agent=agent)
        
        success = interface.remove_from_wishlist(current_user.id, movieId)
        
        if success:
            return {"message": "Movie removed from wishlist", "movieId": movieId}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Movie not found in wishlist"
            )

@router.get("/wishlist")
async def get_wishlist(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Movies per page"),
    current_user: User = Depends(get_current_user)
):
    """
    Get user's wishlist movies with full movie data
    """
    offset = (page - 1) * limit
    
    with PostgreSQLAgent(source_args=source_args) as agent:
        interface = MovieLensInterface(agent=agent)
        
        movies = interface.get_wishlist_movies(current_user.id, limit=limit, offset=offset)
        total = interface.count_wishlist_movies(current_user.id)

        has_next = offset + limit < total
        
        return {
            "movies": movies,
            "page": page,
            "limit": limit,
            "total": total,
            "has_next": has_next
        }

@router.get("/wishlist/check/{movieId}")
async def check_wishlist(
    movieId: int,
    current_user: User = Depends(get_current_user)
):
    """
    Check if a movie exists in user's wishlist
    """
    with PostgreSQLAgent(source_args=source_args) as agent:
        interface = MovieLensInterface(agent=agent)
        
        in_wishlist = interface.is_in_wishlist(current_user.id, movieId)
        
        return {
            "movieId": movieId,
            "inWishlist": in_wishlist
        }

@router.post("/watched/add")
async def add_to_watched(
    movieId: int = Query(..., description="Movie ID to mark as watched"),
    current_user: User = Depends(get_current_user)
):
    """
    Mark a movie as watched
    """
    with PostgreSQLAgent(source_args=source_args) as agent:
        interface = MovieLensInterface(agent=agent)
        
        # Add to watched list
        success = interface.add_to_watched(current_user.id, movieId)
        
        if success:
            return {"message": "Movie marked as watched", "movieId": movieId}
        else:
            return {"message": "Movie already in watched list", "movieId": movieId}

@router.post("/watched/remove")
async def remove_from_watched(
    movieId: int = Query(..., description="Movie ID to remove from watched"),
    current_user: User = Depends(get_current_user)
):
    """
    Remove a movie from user's watched list
    """
    with PostgreSQLAgent(source_args=source_args) as agent:
        interface = MovieLensInterface(agent=agent)
        
        success = interface.remove_from_watched(current_user.id, movieId)
        
        if success:
            return {"message": "Movie removed from watched list", "movieId": movieId}
        else:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Movie not found in watched list"
            )

@router.get("/watched")
async def get_watched(
    page: int = Query(1, ge=1, description="Page number"),
    limit: int = Query(20, ge=1, le=100, description="Movies per page"),
    current_user: User = Depends(get_current_user)
):
    """
    Get user's watched movies with full movie data
    """
    offset = (page - 1) * limit
    
    with PostgreSQLAgent(source_args=source_args) as agent:
        interface = MovieLensInterface(agent=agent)
        
        movies = interface.get_watched_movies(current_user.id, limit=limit, offset=offset)
        total = interface.count_watched_movies(current_user.id)

        has_next = offset + limit < total
        
        return {
            "movies": movies,
            "page": page,
            "limit": limit,
            "total": total,
            "has_next": has_next
        }

@router.get("/watched/check/{movieId}")
async def check_watched(
    movieId: int,
    current_user: User = Depends(get_current_user)
):
    """
    Check if a movie exists in user's watched list
    """
    with PostgreSQLAgent(source_args=source_args) as agent:
        interface = MovieLensInterface(agent=agent)
        
        is_watched = interface.is_watched(current_user.id, movieId)
        
        return {
            "movieId": movieId,
            "isWatched": is_watched
        }

@router.get("/me", response_model=User)
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current user information (requires authentication)
    """
    return current_user

@router.get("/recommendations/onboarding")
async def get_onboarding_recommendations(
    limit: int = Query(20, ge=1, le=50, description="Number of movies to return"),
    current_user: User = Depends(get_current_user)
):
    """
    Get recommendations for onboarding users
    - Returns random popular movies if user has < 20 total movies (watched + wishlist)
    - Returns personalized recommendations if user has >= 20 movies
    """
    with PostgreSQLAgent(source_args=source_args) as agent:
        interface = MovieLensInterface(agent=agent)
        
        # Get user's movie counts
        counts = interface.get_user_movie_counts(current_user.id)
        total_movies = counts["total_count"]
        
        if total_movies < 20:
            # Not enough data - return random popular movies
            movies = interface.get_random_movies_for_user(
                user_id=current_user.id,
                limit=limit,
                min_vote_count=100
            )
            
            return {
                "movies": movies,
                "count": len(movies),
                "type": "random",
                "message": "Add more movies to your lists to get personalized recommendations",
                "user_stats": counts
            }
        else:
            # Enough data - return personalized recommendations
            try:
                movies = interface.get_personalized_recommendations(
                    user_id=current_user.id,
                    limit=limit,
                    n_clusters=3,
                    min_vote_count=50
                )
                
                return {
                    "movies": movies,
                    "count": len(movies),
                    "type": "personalized",
                    "message": "Recommendations based on your taste",
                    "user_stats": counts
                }
            except Exception as e:
                # Fallback to random if personalization fails
                movies = interface.get_random_movies_for_user(
                    user_id=current_user.id,
                    limit=limit,
                    min_vote_count=100
                )
                
                return {
                    "movies": movies,
                    "count": len(movies),
                    "type": "random_fallback",
                    "message": f"Personalization unavailable: {str(e)}",
                    "user_stats": counts
                }

@router.get("/recommendations/personalized")
async def get_personalized_recommendations_endpoint(
    limit: int = Query(20, ge=1, le=50, description="Number of recommendations"),
    n_clusters: int = Query(3, ge=1, le=10, description="Number of taste clusters"),
    min_votes: int = Query(50, ge=0, description="Minimum vote count for quality"),
    current_user: User = Depends(get_current_user)
):
    """
    Get personalized movie recommendations based on user's watch history
    Requires at least some watched movies to work
    """
    with PostgreSQLAgent(source_args=source_args) as agent:
        interface = MovieLensInterface(agent=agent)
        
        # Check if user has enough data
        counts = interface.get_user_movie_counts(current_user.id)
        
        if counts["watched_count"] == 0:
            raise HTTPException(
                status_code=400,
                detail="No watched movies found. Please add some movies to your watched list first."
            )
        
        try:
            movies = interface.get_personalized_recommendations(
                user_id=current_user.id,
                limit=limit,
                n_clusters=n_clusters,
                min_vote_count=min_votes
            )

            
            return {
                "movies": movies,
                "count": len(movies),
                "type": "personalized",
                "user_stats": counts
            }
        except FileNotFoundError:
            raise HTTPException(
                status_code=503,
                detail="Recommendation system not available. Vector database not found."
            )
        
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail="Error: " + str(e)
            )

# Include the router in the app
app.include_router(router)