from fastapi import FastAPI, APIRouter, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional

from lib.interface import MoviesResponse, MovieLensInterface, GenresResponse
from lib.sql.agents import SQLiteAgent

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

DATABASE_NAME = "tastelens"

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

# Include the router in the app
app.include_router(router)