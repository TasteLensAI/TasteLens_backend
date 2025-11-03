import os
import json
import faiss
import numpy as np

from sklearn.cluster import KMeans
from typing import List, Optional
from pydantic import BaseModel, field_validator
from sqlalchemy import text, func
from datetime import datetime
from pathlib import Path

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

    @classmethod
    def from_db_row(cls, row):
        return cls(
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

class VectorDBInterface:
    def __init__(self, path, use_quantized=True):
        """
        Initialize Vector Database Interface
        
        Args:
            path: Path to vector database directory
            use_quantized: If True, use quantized index (smaller memory footprint)
        """
        self.vector_db_path = Path(path)
        self.use_quantized = use_quantized
        self._vector_index = None
        self._idx_to_id_mapping = None
        self._id_to_idx_mapping = None
        self._load_vector_db()
    
    def _load_vector_db(self):
        if self._vector_index is None:
            # Try to load quantized index first if requested
            if self.use_quantized:
                quantized_path = self.vector_db_path / "movie_vectors_quantized.index"
                if quantized_path.exists():
                    print(f"Loading quantized vector database from {quantized_path}")
                    self._vector_index = faiss.read_index(str(quantized_path))
                    # Set search parameters for better accuracy with quantized index
                    self._vector_index.nprobe = 10  # Number of clusters to search
                else:
                    print(f"Quantized index not found at {quantized_path}, falling back to normal index")
                    self.use_quantized = False
            
            # Fall back to normal index if quantized not found or not requested
            if not self.use_quantized:
                index_path = self.vector_db_path / "movie_vectors.index"
                if not index_path.exists():
                    raise FileNotFoundError(f"Vector database not found at {index_path}")
                
                print(f"Loading normal vector database from {index_path}")
                self._vector_index = faiss.read_index(str(index_path))

            # Load ID mappings from separate files (lighter than queries.json)
            idx_to_id_path = self.vector_db_path / "idx_to_id_mapping.json"
            id_to_idx_path = self.vector_db_path / "id_to_idx_mapping.json"
            
            if idx_to_id_path.exists() and id_to_idx_path.exists():
                # Use pre-computed mapping files
                with open(idx_to_id_path, 'r') as f:
                    # Keys are strings in JSON, convert to int
                    self._idx_to_id_mapping = {int(k): v for k, v in json.load(f).items()}
                
                with open(id_to_idx_path, 'r') as f:
                    # Values need to be int, keys are already int movie IDs
                    self._id_to_idx_mapping = {int(k): v for k, v in json.load(f).items()}
                
                print(f"Loaded ID mappings for {len(self._idx_to_id_mapping)} movies")
            else:
                # Fallback to queries.json if mapping files don't exist
                print("ID mapping files not found, loading from queries.json (legacy)")
                queries_path = self.vector_db_path / "queries.json"
                with open(queries_path, 'r') as f:
                    queries = json.load(f)

                self._idx_to_id_mapping = {i: int(movie_id) for i, movie_id in enumerate(queries.keys())}
                self._id_to_idx_mapping = {movid:idx for idx, movid in self._idx_to_id_mapping.items()}

    def _get_clustered_profiling_vectors(self, movie_lists_map, n_clusters = 3, min_movies_per_cluster=5):
        movie_lists = [item["list"] for item in movie_lists_map]
        weights = [item["weight"] for item in movie_lists_map]

        all_vectors = []
        all_weights = []
        for i, movie_id_list in enumerate(movie_lists):
            weight = weights[i]
            
            # Convert movie IDs to indices, filtering out missing ones
            indices = [self._id_to_idx_mapping[movie_id] for movie_id in movie_id_list 
                    if movie_id in self._id_to_idx_mapping]
            
            if not indices:
                continue
            
            # Reconstruct vectors for this list
            # Use reconstruct() for single vector or reconstruct_batch() for multiple
            indices_array = np.array(indices, dtype=np.int64)
            vectors = np.vstack([self._vector_index.reconstruct(int(idx)) for idx in indices_array])
            
            # Average vectors in this list and apply weight
            all_vectors.append(vectors)
            all_weights.extend([weight] * len(vectors))
        
        if not all_vectors:
            return None

        combined_vectors = np.vstack(all_vectors)
        all_weights = np.array(all_weights)

        if len(combined_vectors) < min_movies_per_cluster:
            # Fall back to simple average if not enough movies
            weighted_avg = np.average(combined_vectors, axis=0, weights=all_weights, keepdims=True)
            weighted_avg = weighted_avg.astype('float32')
            faiss.normalize_L2(weighted_avg)
            return [(weighted_avg, 1.0)]
        
        max_clusters = min(n_clusters, len(combined_vectors) // min_movies_per_cluster)
        max_clusters = max(2, max_clusters)  # At least 2 clusters

        kmeans = KMeans(n_clusters=max_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(combined_vectors)

        profile_vectors = []
        for cluster_id in range(max_clusters):
            # Get vectors belonging to this cluster
            cluster_mask = cluster_labels == cluster_id
            cluster_vectors = combined_vectors[cluster_mask]
            cluster_weights = all_weights[cluster_mask]
            
            if len(cluster_vectors) == 0:
                continue
            
            # Weighted average within cluster
            cluster_profile = np.average(cluster_vectors, axis=0, weights=cluster_weights, keepdims=True)
            cluster_profile = cluster_profile.astype('float32')
            faiss.normalize_L2(cluster_profile)

            # Weight is proportional to cluster size (bigger clusters = more important)
            cluster_weight = len(cluster_vectors) / len(combined_vectors)

            profile_vectors.append((cluster_profile, cluster_weight))
        
        return profile_vectors if profile_vectors else None
    
    def get_recommendations(
        self,
        watched_ids,
        wishlist_ids=None,
        k=20,
        n_clusters=3,
        watched_weight=1.0,
        wishlist_weight=0.4
    ):
        movie_lists_map = [
            {"list": watched_ids, "weight": watched_weight}
        ]
        
        if wishlist_ids:
            movie_lists_map.append(
                {"list": wishlist_ids, "weight": wishlist_weight}
            )

        # Get clustered profile vectors
        profile_vectors = self._get_clustered_profiling_vectors(
            movie_lists_map, 
            n_clusters=n_clusters
        )

        if profile_vectors is None:
            return []
        
        exclude_ids = set(watched_ids)
        if wishlist_ids:
            exclude_ids.update(wishlist_ids)

        all_recommendations = []
        for cluster_id, (profile_vector, cluster_weight) in enumerate(profile_vectors):
            cluster_k = int(k * cluster_weight)
            search_k = cluster_k * 3
            
            distances, indices = self._vector_index.search(profile_vector, search_k)
            
            cluster_recs = []
            for idx, dist in zip(indices[0], distances[0]):
                movie_id = self._idx_to_id_mapping[idx]
                if movie_id not in exclude_ids:
                    cluster_recs.append((movie_id, float(dist), cluster_id))
                    exclude_ids.add(movie_id)

            all_recommendations.extend(cluster_recs)
        
        all_recommendations.sort(key=lambda x: x[1])
        return all_recommendations[:k]

class MovieLensInterface:
    def __init__(self, agent):
        self.agent = agent
        self.vector_db = None

    def _init_vector_db(self):
        if self.vector_db == None:
            # Get vector DB path from environment or use default
            vector_db_path = os.getenv(
                "VECTOR_DB_PATH",
                os.path.join(os.path.dirname(__file__), "..", "data", "vector-db")
            )
            # Use quantized index by default (set to False to use normal index)
            use_quantized = os.getenv("USE_QUANTIZED_INDEX", "true").lower() == "true"
            self.vector_db = VectorDBInterface(path=vector_db_path, use_quantized=use_quantized)

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
        return [Movie.from_db_row(row) for row in results]

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
        Uses SQLAlchemy ORM for type safety and database portability.
        
        Returns:
            Dictionary with genre statistics and total count
        """
        dbref = BaseDatabaseReference()
        MovieGenres = dbref._references["moviegenres"]
        
        # Count total unique movies with genres using SQLAlchemy
        total_movies = self.agent.session.query(
            func.count(func.distinct(MovieGenres.movieId))
        ).scalar()
        
        # Get genre counts directly from the table using SQLAlchemy
        # Query returns tuples of (genre, count) ordered by count descending
        genre_counts = self.agent.session.query(
            MovieGenres.genre,
            func.count(MovieGenres.genre).label('count')
        ).group_by(
            MovieGenres.genre
        ).order_by(
            func.count(MovieGenres.genre).desc()
        ).all()
        
        # Convert to list of GenreStats
        genre_stats = [
            GenreStats(genre=row.genre, count=row.count)
            for row in genre_counts
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

    # Recommendation Methods
    def get_random_movies_for_user(
        self, 
        user_id: int, 
        limit: int = 20,
        min_vote_count: int = 100,
        min_rating: float = 6.5
    ) -> List[Movie]:
        """
        Get random popular movies that are NOT in user's watched or wishlist
        Useful for user onboarding and discovery
        
        Args:
            user_id: User ID to exclude their movies
            limit: Number of random movies to return
            min_vote_count: Minimum vote count to ensure quality movies
            min_rating: Minimum TMDB rating (default 6.5)
            
        Returns:
            List of Movie objects
        """
        self._init_vector_db()

        dbref = BaseDatabaseReference()
        Metadata = dbref._references["metadata"]
        Watched = dbref._references["watched"]
        Wishlist = dbref._references["wishlist"]

        watched_subquery = self.agent.session.query(Watched.movieId).filter(
            Watched.userId == user_id
        )

        wishlist_subquery = self.agent.session.query(Wishlist.movieId).filter(
            Wishlist.userId == user_id
        )

        query = self.agent.session.query(Metadata).filter(
            Metadata.genres.isnot(None),
            Metadata.genres != '',
            Metadata.tmdbVoteCount >= min_vote_count,
            Metadata.tmdbRating >= min_rating,
            ~Metadata.movieId.in_(watched_subquery),
            ~Metadata.movieId.in_(wishlist_subquery)
        ).order_by(func.random()).limit(limit)

        return [Movie.from_db_row(row) for row in query.all()]

    def get_user_movie_counts(self, user_id: int) -> dict:
        """
        Get counts of movies in user's watched and wishlist
        Useful for determining if user needs onboarding
        
        Args:
            user_id: User ID
            
        Returns:
            Dictionary with watched_count and wishlist_count
        """
        watched_count = self.count_watched_movies(user_id)
        wishlist_count = self.count_wishlist_movies(user_id)

        return {
            "watched_count": watched_count,
            "wishlist_count": wishlist_count,
            "total_count": watched_count + wishlist_count,
        }

    def get_personalized_recommendations(
        self,
        user_id: int,
        limit: int = 20,
        n_clusters: int = 3,
        watched_weight: float = 1.0,
        wishlist_weight: float = 0.4,
        min_vote_count: int = 50
    ) -> List[Movie]:
        """
        Get personalized movie recommendations using vector similarity
        Uses clustering to capture different aspects of user's taste
        
        Args:
            user_id: User ID
            limit: Number of recommendations to return
            n_clusters: Number of taste clusters to create
            watched_weight: Weight for watched movies (default 1.0)
            wishlist_weight: Weight for wishlist movies (default 0.4)
            min_vote_count: Minimum vote count for quality filtering
            
        Returns:
            List of recommended Movie objects
        """
        self._init_vector_db()

        # Get user's watched and wishlist movies
        watched_movies = self.get_watched_movies(user_id, limit=200)
        wishlist_movies = self.get_wishlist_movies(user_id, limit=100)

        num_movies = len(watched_movies) + len(wishlist_movies)
        if num_movies < 5:
            # User has no data, return random popular movies
            return self.get_random_movies_for_user(user_id, limit, min_vote_count)
        
        # Extract movie IDs
        watched_ids = [m.movieId for m in watched_movies]
        wishlist_ids = [m.movieId for m in wishlist_movies] if wishlist_movies else None
        
        # Get recommendations from vector database
        recommendations = self.vector_db.get_recommendations(
            watched_ids=watched_ids,
            wishlist_ids=wishlist_ids,
            k=limit * 2,  # Get more for filtering
            n_clusters=n_clusters,
            watched_weight=watched_weight,
            wishlist_weight=wishlist_weight
        )
        
        if not recommendations:
            raise
        
        # Extract movie IDs from recommendations (ignore distance and cluster_id)
        recommended_movie_ids = [movie_id for movie_id, _, _ in recommendations]
        
        # Get full movie data
        movies = self._get_movies_by_ids(recommended_movie_ids)
        
        # Filter by vote count for quality
        filtered_movies = [m for m in movies if m.tmdbVoteCount >= min_vote_count]
        
        # Return requested limit
        return filtered_movies[:limit]
    
    def get_similar_movies(
        self,
        movie_id: int,
        limit: int = 10,
        user_id: Optional[int] = None,
        min_vote_count: int = 50
    ) -> List[Movie]:
        """
        Get movies similar to a specific movie using vector similarity
        
        Args:
            movie_id: Movie ID to find similar movies for
            limit: Number of similar movies to return
            user_id: Optional user ID to exclude their watched/wishlist
            min_vote_count: Minimum vote count for quality filtering
            
        Returns:
            List of similar Movie objects
        """
        self._init_vector_db()

        try:
            # Get exclusion lists if user_id provided
            exclude_ids = set()
            if user_id:
                watched = self.get_watched_movies(user_id, limit=1000)
                exclude_ids.update(m.movieId for m in watched)
                wishlist = self.get_wishlist_movies(user_id, limit=1000)
                exclude_ids.update(m.movieId for m in wishlist)
            
            # Add the source movie to exclusions
            exclude_ids.add(movie_id)
            
            # Create a simple recommendation based on single movie
            recommendations = self.vector_db.get_recommendations(
                watched_ids=[movie_id],
                wishlist_ids=None,
                k=limit * 2,  # Get more for filtering
                n_clusters=1,  # No clustering for single movie
                watched_weight=1.0,
                wishlist_weight=0.0
            )
            
            if not recommendations:
                return []
            
            # Filter out excluded movies
            filtered_recs = [(mid, dist, cid) for mid, dist, cid in recommendations 
                           if mid not in exclude_ids]
            
            # Get movie IDs
            recommended_movie_ids = [movie_id for movie_id, _, _ in filtered_recs]
            
            # Get full movie data
            movies = self._get_movies_by_ids(recommended_movie_ids)
            
            # Filter by vote count
            filtered_movies = [m for m in movies if m.tmdbVoteCount >= min_vote_count]
            
            return filtered_movies[:limit]
            
        except Exception as e:
            print(f"Error getting similar movies: {e}")
            return []
    
    def get_recommendation_insights(
        self,
        user_id: int,
        n_clusters: int = 3
    ) -> dict:
        """
        Get insights about user's taste clusters
        Useful for explaining recommendations or showing taste profiles
        
        Args:
            user_id: User ID
            n_clusters: Number of taste clusters to analyze
            
        Returns:
            Dictionary with cluster information and movie samples
        """
        self._init_vector_db()

        # Get user's movies
        watched_movies = self.get_watched_movies(user_id, limit=200)
        wishlist_movies = self.get_wishlist_movies(user_id, limit=100)
        
        if not watched_movies:
            return {
                "error": "User has no watched movies",
                "total_movies": 0
            }
        
        watched_ids = [m.movieId for m in watched_movies]
        wishlist_ids = [m.movieId for m in wishlist_movies] if wishlist_movies else []
        
        try:
            # Build movie lists map
            movie_lists_map = [{"list": watched_ids, "weight": 1.0}]
            if wishlist_ids:
                movie_lists_map.append({"list": wishlist_ids, "weight": 0.4})
            
            # Get clustered profile vectors
            profile_vectors = self.vector_db._get_clustered_profiling_vectors(
                movie_lists_map,
                n_clusters=n_clusters
            )
            
            if not profile_vectors:
                return {
                    "error": "Could not create taste clusters",
                    "total_movies": len(watched_ids) + len(wishlist_ids)
                }
            
            # Get sample recommendations from each cluster
            clusters = []
            for cluster_id, (profile_vector, cluster_weight) in enumerate(profile_vectors):
                # Get top 5 recommendations from this cluster
                distances, indices = self.vector_db._vector_index.search(profile_vector, 5)
                
                sample_movie_ids = [
                    self.vector_db._idx_to_id_mapping[idx] 
                    for idx in indices[0]
                ]
                
                sample_movies = self._get_movies_by_ids(sample_movie_ids)
                
                clusters.append({
                    "cluster_id": cluster_id,
                    "weight": round(cluster_weight * 100, 1),  # As percentage
                    "sample_movies": sample_movies
                })
            
            return {
                "total_movies": len(watched_ids) + len(wishlist_ids),
                "watched_count": len(watched_ids),
                "wishlist_count": len(wishlist_ids),
                "n_clusters": len(profile_vectors),
                "clusters": clusters
            }
            
        except Exception as e:
            return {
                "error": f"Failed to analyze taste clusters: {str(e)}",
                "total_movies": len(watched_ids) + len(wishlist_ids)
            }