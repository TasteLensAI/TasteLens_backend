from typing import List, Optional
from pydantic import BaseModel, field_validator
from sqlalchemy import text, func
from datetime import datetime

from collections import Counter

from lib.sql.builders import BaseDatabaseReference

class Movie(BaseModel):
    movieId: int
    tmdbId: str
    title: str
    original_title: str
    genres: str
    tagline: str
    description: str
    year: int
    duration: int
    tmdbRating: float
    tmdbVoteCount: int
    poster_path: str

class MoviesResponse(BaseModel):
    movies: List[Movie]
    total: int
    page: int
    limit: int
    has_next: bool

class GenreStats(BaseModel):
    genre: str
    count: int

class GenresResponse(BaseModel):
    genres: List[GenreStats]
    total_movies: int

# Authentication Models
class UserCreate(BaseModel):
    username: str
    email: str  # Keep email but no verification required
    password: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    bio: Optional[str] = None
    
    @field_validator("password")
    @classmethod
    def validate_password(cls, v):
        if len(v) < 6:
            raise ValueError('Password must be at least 6 characters long')
        # No upper limit - SHA-256 pre-hashing handles any length safely
        return v
    
    @field_validator("username")
    @classmethod
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError('Username must be at least 3 characters long')
        if len(v) > 50:
            raise ValueError('Username cannot be longer than 50 characters')
        return v

class UserLogin(BaseModel):
    username: str
    password: str

class User(BaseModel):
    id: int
    username: str
    email: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    display_name: Optional[str] = None
    bio: Optional[str] = None
    avatar_url: Optional[str] = None
    is_verified: bool
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class UserResponse(BaseModel):
    user: User
    token: Token

class LoginResponse(BaseModel):
    user: User
    token: Token

class MovieLensInterface:
    def __init__(self, agent):
        self.agent = agent

    def get_movies(self, offset: int = 0, limit: int = 10, search: Optional[str] = None, genres: Optional[List[str]] = None) -> List[Movie]:
        """
        Get movies from the metadata table with pagination and optional filtering
        
        Args:
            offset: Number of records to skip
            limit: Maximum number of records to return
            search: Optional search term for title
            genres: Optional list of genres (movies must have ALL selected genres)
        
        Returns:
            List of Movie objects
        """
        
        query = """
        SELECT m.movieId, m.tmdbId, m.title, m.original_title, m.genres, m.tagline, m.description, 
               m.year, m.duration, m.tmdbRating, m.tmdbVoteCount, m.poster_path
        FROM metadata m
        WHERE 1=1 AND m.genres IS NOT NULL AND m.genres != ''
        """
        params = {}
        
        if search:
            query += " AND LOWER(m.title) LIKE :search"
            params['search'] = f"%{search.lower()}%"
        
        if genres and len(genres) > 0:
            # For each genre, add a condition that the movie's genres must contain it
            for i, genre in enumerate(genres):
                param_name = f"genre_{i}"
                query += f" AND (LOWER(m.genres) LIKE :{param_name})"
                params[param_name] = f"%{genre.lower()}%"
        
        query += " ORDER BY m.tmdbRating DESC NULLS LAST, m.title"
        query += " LIMIT :limit OFFSET :offset"
        params['limit'] = limit
        params['offset'] = offset
        
        result = self.agent.session.execute(text(query), params)
        rows = result.fetchall()
        
        movies = []
        for row in rows:
            movie = Movie(
                movieId=row[0],
                tmdbId=row[1] or "",
                title=row[2] or "Unknown Title",
                original_title=row[3] or "",
                genres=row[4] or "",
                tagline=row[5] or "",
                description=row[6] or "",
                year=row[7] or 0,
                duration=row[8] or 0,
                tmdbRating=row[9] or 0.0,
                tmdbVoteCount=row[10] or 0,
                poster_path=row[11] or ""
            )
            movies.append(movie)
        
        return movies

    def count_movies(self, search: Optional[str] = None, genres: Optional[List[str]] = None) -> int:
        """
        Count total movies in the metadata table with optional filtering
        
        Args:
            search: Optional search term for title
            genres: Optional list of genres (movies must have ALL selected genres)
        
        Returns:
            Total count of movies matching the criteria
        """
        query = "SELECT COUNT(*) FROM metadata m WHERE 1=1 AND m.genres IS NOT NULL AND m.genres != ''"
        params = {}
        
        if search:
            query += " AND LOWER(m.title) LIKE :search"
            params['search'] = f"%{search.lower()}%"
        
        if genres and len(genres) > 0:
            # For each genre, add a condition that the movie's genres must contain it
            for i, genre in enumerate(genres):
                param_name = f"genre_{i}"
                query += f" AND (LOWER(m.genres) LIKE :{param_name})"
                params[param_name] = f"%{genre.lower()}%"
        
        result = self.agent.session.execute(text(query), params)
        return result.scalar()
    
    def get_genre_statistics(self) -> dict:
        """
        Get statistics for all unique genres with their movie counts
        
        Returns:
            Dictionary with genre statistics and total count
        """
        # First get the total count of all movies
        total_query = "SELECT COUNT(*) FROM metadata WHERE genres IS NOT NULL AND genres != ''"
        total_result = self.agent.session.execute(text(total_query))
        total_movies = total_result.scalar()
        
        # Get all movies with genres and split them
        query = "SELECT genres FROM metadata WHERE genres IS NOT NULL AND genres != ''"
        result = self.agent.session.execute(text(query))
        rows = result.fetchall()
        
        # Count genres
        genre_counts = {}
        genres = [
            [
                genre.strip().lower() 
                for genre in row[0].split("|")
            ] 
            for row in rows if (row[0] and row[0] != "")
        ]
        genres = sum(genres, start=[])
        genre_counts = dict(Counter(genres))
        
        
        # Convert to list of GenreStats, sorted by count descending
        genre_stats = [
            GenreStats(genre=genre, count=count)
            for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {
            "genres": genre_stats,
            "total_movies": total_movies
        }
    
    # User Authentication Methods
    def create_user(self, user_data: UserCreate, password_hash: str) -> User:
        """
        Create a new user in the database
        
        Args:
            user_data: UserCreate model with user information
            password_hash: Hashed password
            
        Returns:
            Created User object
            
        Raises:
            Exception if user creation fails
        """        
        # Get the Users table reference
        dbref = BaseDatabaseReference()
        Users = dbref._references["users"]
        
        # Create new user instance
        new_user = Users(
            username=user_data.username,
            email=user_data.email,
            password_hash=password_hash,
            first_name=user_data.first_name,
            last_name=user_data.last_name,
            display_name=user_data.display_name or user_data.username,
            bio=user_data.bio,
            is_verified=False  # No email verification for now
        )
        
        # Add to session and commit
        self.agent.session.add(new_user)
        self.agent.session.commit()
        self.agent.session.refresh(new_user)
        
        # Convert to Pydantic model
        return User(
            id=new_user.id,
            username=new_user.username,
            email=new_user.email,
            first_name=new_user.first_name,
            last_name=new_user.last_name,
            display_name=new_user.display_name,
            bio=new_user.bio,
            avatar_url=new_user.avatar_url,
            is_verified=new_user.is_verified,
            created_at=new_user.created_at,
            updated_at=new_user.updated_at,
            last_login=new_user.last_login
        )
    
    def get_user_by_username(self, username: str):
        """
        Get user by username
        
        Args:
            username: Username to search for
            
        Returns:
            User SQLAlchemy object or None if not found
        """

        dbref = BaseDatabaseReference()
        Users = dbref._references["users"]
        
        return self.agent.session.query(Users).filter(Users.username == username).first()
    
    def get_user_by_email(self, email: str):
        """
        Get user by email
        
        Args:
            email: Email to search for
            
        Returns:
            User SQLAlchemy object or None if not found
        """

        dbref = BaseDatabaseReference()
        Users = dbref._references["users"]
        
        return self.agent.session.query(Users).filter(Users.email == email).first()
    
    def update_user_last_login(self, user_id: int):
        """
        Update user's last login timestamp
        
        Args:
            user_id: User ID to update
        """
        
        dbref = BaseDatabaseReference()
        Users = dbref._references["users"]
        
        self.agent.session.query(Users).filter(Users.id == user_id).update({
            "last_login": func.now()
        })
        self.agent.session.commit()
    
    def user_exists(self, username: str = None, email: str = None) -> bool:
        """
        Check if user exists by username or email
        
        Args:
            username: Username to check
            email: Email to check
            
        Returns:
            True if user exists, False otherwise
        """
        if username:
            return self.get_user_by_username(username) is not None
        if email:
            return self.get_user_by_email(email) is not None
        return False