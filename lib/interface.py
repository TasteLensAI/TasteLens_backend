from typing import List, Optional
from pydantic import BaseModel
from sqlalchemy import text

from collections import Counter

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