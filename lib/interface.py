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

    def _get_movies_by_ids(self, movie_ids: List[int]) -> List[Movie]:
        """
        Helper function to get full movie data by movie IDs using SQLAlchemy ORM
        
        Args:
            movie_ids: List of movie IDs to fetch
            
        Returns:
            List of Movie objects
        """
        if not movie_ids:
            return []
        
        dbref = BaseDatabaseReference()
        Metadata = dbref._references["metadata"]
        
        # Use SQLAlchemy query with in_ filter
        results = self.agent.session.query(Metadata).filter(
            Metadata.movieId.in_(movie_ids)
        ).all()
        
        # Create a dict for quick lookup
        movies_dict = {}
        for row in results:
            movie = Movie(
                movieId=row.movieId,
                tmdbId=row.tmdbId or "",
                title=row.title or "Unknown Title",
                original_title=row.original_title or "",
                genres=row.genres or "",
                tagline=row.tagline or "",
                description=row.description or "",
                year=row.year or 0,
                duration=row.duration or 0,
                tmdbRating=row.tmdbRating or 0.0,
                tmdbVoteCount=row.tmdbVoteCount or 0,
                poster_path=row.poster_path or ""
            )
            movies_dict[row.movieId] = movie
        
        # Return movies in the same order as input IDs
        return [movies_dict[movie_id] for movie_id in movie_ids if movie_id in movies_dict]

    def get_movies(self, offset: int = 0, limit: int = 10, search: Optional[str] = None, genres: Optional[List[str]] = None) -> List[Movie]:
        """
        Get movies from the metadata table with pagination and optional filtering
        Uses SQLAlchemy ORM with weighted scoring to prevent low vote count movies from dominating
        
        Args:
            offset: Number of records to skip
            limit: Maximum number of records to return
            search: Optional search term for title
            genres: Optional list of genres (movies must have ALL selected genres)
        
        Returns:
            List of Movie objects ordered by weighted score (rating + vote count consideration)
        """
        dbref = BaseDatabaseReference()
        Metadata = dbref._references["metadata"]
        
        # Start with base query
        query = self.agent.session.query(Metadata).filter(
            Metadata.genres.isnot(None),
            Metadata.genres != ''
        )
        
        # Apply search filter
        if search:
            query = query.filter(
                func.lower(Metadata.title).like(f"%{search.lower()}%")
            )
        
        # Apply genre filters (movies must have ALL selected genres)
        if genres and len(genres) > 0:
            for genre in genres:
                query = query.filter(
                    func.lower(Metadata.genres).like(f"%{genre.lower()}%")
                )
        
        # Order by weighted score: combines rating with vote count consideration
        # Formula: (tmdbRating * tmdbVoteCount + 6.5 * 100) / (tmdbVoteCount + 100)
        # This is a Bayesian average with C=100 (confidence threshold) and m=6.5 (average rating)
        # Movies need ~100 votes to significantly impact their position
        # Then order by vote count as tiebreaker, then title
        query = query.order_by(
            ((Metadata.tmdbRating * Metadata.tmdbVoteCount + 6.5 * 100) / 
             (Metadata.tmdbVoteCount + 100)).desc(),
            Metadata.tmdbVoteCount.desc(),
            Metadata.title
        )
        
        # Apply pagination
        query = query.limit(limit).offset(offset)
        
        # Execute and convert to Movie objects
        results = query.all()
        
        movies = []
        for row in results:
            movie = Movie(
                movieId=row.movieId,
                tmdbId=row.tmdbId or "",
                title=row.title or "Unknown Title",
                original_title=row.original_title or "",
                genres=row.genres or "",
                tagline=row.tagline or "",
                description=row.description or "",
                year=row.year or 0,
                duration=row.duration or 0,
                tmdbRating=row.tmdbRating or 0.0,
                tmdbVoteCount=row.tmdbVoteCount or 0,
                poster_path=row.poster_path or ""
            )
            movies.append(movie)
        
        return movies

    def count_movies(self, search: Optional[str] = None, genres: Optional[List[str]] = None) -> int:
        """
        Count total movies in the metadata table with optional filtering
        Uses SQLAlchemy ORM
        
        Args:
            search: Optional search term for title
            genres: Optional list of genres (movies must have ALL selected genres)
        
        Returns:
            Total count of movies matching the criteria
        """
        dbref = BaseDatabaseReference()
        Metadata = dbref._references["metadata"]
        
        # Start with base query
        query = self.agent.session.query(Metadata).filter(
            Metadata.genres.isnot(None),
            Metadata.genres != ''
        )
        
        # Apply search filter
        if search:
            query = query.filter(
                func.lower(Metadata.title).like(f"%{search.lower()}%")
            )
        
        # Apply genre filters (movies must have ALL selected genres)
        if genres and len(genres) > 0:
            for genre in genres:
                query = query.filter(
                    func.lower(Metadata.genres).like(f"%{genre.lower()}%")
                )
        
        return query.count()
    
    def get_genre_statistics(self) -> dict:
        """
        Get statistics for all unique genres with their movie counts.
        Uses the pre-built movie_genres table for fast performance.
        
        Returns:
            Dictionary with genre statistics and total count
        """
        # Count total unique movies with genres
        total_query = "SELECT COUNT(DISTINCT movieId) FROM moviegenres"
        total_result = self.agent.session.execute(text(total_query))
        total_movies = total_result.scalar()
        
        # Get genre counts directly from the table
        stats_query = """
        SELECT genre, COUNT(*) as count
        FROM moviegenres
        GROUP BY genre
        ORDER BY count DESC
        """
        result = self.agent.session.execute(text(stats_query))
        rows = result.fetchall()
        
        # Convert to list of GenreStats
        genre_stats = [
            GenreStats(genre=row[0], count=row[1])
            for row in rows
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
    
    # Watched Movies Methods
    def add_to_watched(self, user_id: int, movie_id: int, watched_at: Optional[datetime] = None) -> bool:
        """
        Add a movie to user's watched list
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            watched_at: Optional timestamp when movie was watched
            
        Returns:
            True if added successfully, False if already exists
        """
        dbref = BaseDatabaseReference()
        Watched = dbref._references["watched"]
        
        # Check if already watched
        existing = self.agent.session.query(Watched).filter(
            Watched.userId == user_id,
            Watched.movieId == movie_id
        ).first()
        
        if existing:
            return False
        
        # Add to watched
        watched_entry = Watched(
            userId=user_id,
            movieId=movie_id,
            watched_at=watched_at or func.now()
        )
        
        self.agent.session.add(watched_entry)
        self.agent.session.commit()
        return True
    
    def remove_from_watched(self, user_id: int, movie_id: int) -> bool:
        """
        Remove a movie from user's watched list
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            True if removed, False if not found
        """
        dbref = BaseDatabaseReference()
        Watched = dbref._references["watched"]
        
        deleted = self.agent.session.query(Watched).filter(
            Watched.userId == user_id,
            Watched.movieId == movie_id
        ).delete()
        
        self.agent.session.commit()
        return deleted > 0
    
    def get_watched_movies(self, user_id: int, limit: int = 50, offset: int = 0) -> List[Movie]:
        """
        Get user's watched movies with full movie data
        
        Args:
            user_id: User ID
            limit: Maximum number of results
            offset: Number of records to skip
            
        Returns:
            List of Movie objects
        """
        dbref = BaseDatabaseReference()
        Watched = dbref._references["watched"]
        
        watched = self.agent.session.query(Watched).filter(
            Watched.userId == user_id
        ).order_by(Watched.watched_at.desc()).limit(limit).offset(offset).all()
        
        # Get movie IDs
        movie_ids = [w.movieId for w in watched]
        
        # Fetch full movie data
        return self._get_movies_by_ids(movie_ids)
    
    def count_watched_movies(self, user_id: int) -> int:
        """
        Count total watched movies for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Total count of watched movies
        """
        dbref = BaseDatabaseReference()
        Watched = dbref._references["watched"]
        
        count = self.agent.session.query(Watched).filter(
            Watched.userId == user_id
        ).count()
        
        return count
    
    def is_watched(self, user_id: int, movie_id: int) -> bool:
        """
        Check if user has watched a movie
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            True if watched, False otherwise
        """
        dbref = BaseDatabaseReference()
        Watched = dbref._references["watched"]
        
        return self.agent.session.query(Watched).filter(
            Watched.userId == user_id,
            Watched.movieId == movie_id
        ).first() is not None
    
    # Wishlist Methods
    def add_to_wishlist(self, user_id: int, movie_id: int) -> bool:
        """
        Add a movie to user's wishlist
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            True if added successfully, False if already exists
        """
        dbref = BaseDatabaseReference()
        Wishlist = dbref._references["wishlist"]
        
        # Check if already in wishlist
        existing = self.agent.session.query(Wishlist).filter(
            Wishlist.userId == user_id,
            Wishlist.movieId == movie_id
        ).first()
        
        if existing:
            return False
        
        # Add to wishlist
        wishlist_entry = Wishlist(
            userId=user_id,
            movieId=movie_id
        )
        
        self.agent.session.add(wishlist_entry)
        self.agent.session.commit()
        return True
    
    def remove_from_wishlist(self, user_id: int, movie_id: int) -> bool:
        """
        Remove a movie from user's wishlist
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            True if removed, False if not found
        """
        dbref = BaseDatabaseReference()
        Wishlist = dbref._references["wishlist"]
        
        deleted = self.agent.session.query(Wishlist).filter(
            Wishlist.userId == user_id,
            Wishlist.movieId == movie_id
        ).delete()
        
        self.agent.session.commit()
        return deleted > 0
    
    def get_wishlist_movies(self, user_id: int, limit: int = 50, offset: int = 0) -> List[Movie]:
        """
        Get user's wishlist movies with full movie data
        
        Args:
            user_id: User ID
            limit: Maximum number of results
            offset: Number of records to skip
            
        Returns:
            List of Movie objects
        """
        dbref = BaseDatabaseReference()
        Wishlist = dbref._references["wishlist"]
        
        wishlist = self.agent.session.query(Wishlist).filter(
            Wishlist.userId == user_id
        ).order_by(Wishlist.added_at.desc()).limit(limit).offset(offset).all()
        
        # Get movie IDs
        movie_ids = [w.movieId for w in wishlist]
        
        # Fetch full movie data
        return self._get_movies_by_ids(movie_ids)
    
    def count_wishlist_movies(self, user_id: int) -> int:
        """
        Count total watched movies for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Total count of watched movies
        """
        dbref = BaseDatabaseReference()
        Wishlist = dbref._references["wishlist"]
        
        count = self.agent.session.query(Wishlist).filter(
            Wishlist.userId == user_id
        ).count()
        
        return count
    
    def is_in_wishlist(self, user_id: int, movie_id: int) -> bool:
        """
        Check if movie is in user's wishlist
        
        Args:
            user_id: User ID
            movie_id: Movie ID
            
        Returns:
            True if in wishlist, False otherwise
        """
        dbref = BaseDatabaseReference()
        Wishlist = dbref._references["wishlist"]
        
        return self.agent.session.query(Wishlist).filter(
            Wishlist.userId == user_id,
            Wishlist.movieId == movie_id
        ).first() is not None