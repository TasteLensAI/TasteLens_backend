from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, SmallInteger
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func
import pandas as pd
from pathlib import Path
import json

class BaseDatabaseReference:
    def __init__(self):
        self.base = declarative_base()
        self._define_tables()

    def _define_tables(self):
        """Define all database tables"""
        
        class Movies(self.base):
            __tablename__ = "movies"
            __table_args__ = {'extend_existing': True}
            movieId = Column(Integer, primary_key=True)
            title = Column(String(255))
            genres = Column(String(500))

        class Links(self.base):
            __tablename__ = "links"
            __table_args__ = {'extend_existing': True}
            movieId = Column(Integer, primary_key=True)
            imdbId = Column(String)
            tmdbId = Column(String)
        
        class Metadata(self.base):
            __tablename__ = "metadata"
            __table_args__ = {'extend_existing': True}
            metadataId = Column(Integer, primary_key=True)
            movieId = Column(Integer)
            tmdbId = Column(String)
            title = Column(String)
            original_title = Column(String)
            genres = Column(String)
            tagline = Column(String)
            description = Column(String)
            year = Column(Integer)
            duration = Column(Integer)
            tmdbRating = Column(Float)
            tmdbVoteCount = Column(Integer)
            poster_path = Column(String)

        class MovieActors(self.base):
            __tablename__ = "movieactors"
            __table_args__ = {'extend_existing': True}
            movieActorId = Column(Integer, primary_key=True)
            movieId = Column(Integer)
            name = Column(String)
            popularity = Column(Float)

        class MovieDirectors(self.base):
            __tablename__ = "moviedirectors"
            __table_args__ = {'extend_existing': True}
            movieDirectorId = Column(Integer, primary_key=True)
            movieId = Column(Integer)
            name = Column(String)
            popularity = Column(Float)

        class MovieGenres(self.base):
            __tablename__ = "moviegenres"
            __table_args__ = {'extend_existing': True}
            id = Column(Integer, primary_key=True, autoincrement=True)
            movieId = Column(Integer, index=True)
            genre = Column(String(100), index=True)

        class Watched(self.base):
            __tablename__ = "watched"
            __table_args__ = {'extend_existing': True}
            id = Column(Integer, primary_key=True, autoincrement=True)
            userId = Column(Integer, nullable=False, index=True)
            movieId = Column(Integer, nullable=False, index=True)
            watched_at = Column(DateTime(timezone=True), server_default=func.now())
            
            def __repr__(self):
                return f"<Watched(userId={self.userId}, movieId={self.movieId}, watched_at={self.watched_at})>"

        class Wishlist(self.base):
            __tablename__ = "wishlist"
            __table_args__ = {'extend_existing': True}
            id = Column(Integer, primary_key=True, autoincrement=True)
            userId = Column(Integer, nullable=False, index=True)
            movieId = Column(Integer, nullable=False, index=True)
            added_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
            
            def __repr__(self):
                return f"<Wishlist(userId={self.userId}, movieId={self.movieId})>"

        class Users(self.base):
            __tablename__ = "users"
            __table_args__ = {'extend_existing': True}
            
            # Primary key
            id = Column(Integer, primary_key=True, autoincrement=True)
            
            # Basic user info
            username = Column(String(50), unique=True, nullable=False, index=True)
            email = Column(String(255), unique=True, nullable=False, index=True)
            password_hash = Column(String(255), nullable=False)  # Store hashed passwords only
            
            # Profile info
            first_name = Column(String(100))
            last_name = Column(String(100))
            display_name = Column(String(100))
            bio = Column(Text)
            avatar_url = Column(Text)  # URL or path to profile picture
            
            # Account status
            # is_active = Column(Boolean, default=True, nullable=False)
            is_verified = Column(Boolean, default=False, nullable=False)
            # is_admin = Column(Boolean, default=False, nullable=False)
            
            # Timestamps
            created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
            updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
            last_login = Column(DateTime(timezone=True))
            
            def __repr__(self):
                return f"<User(id={self.id}, username='{self.username}', email='{self.email}')>"
        
        # Store references for external access
        self._references = {
            "movies": Movies,
            "links": Links,
            "users": Users,
            "metadata": Metadata,
            "movieactors": MovieActors,
            "moviedirectors": MovieDirectors,
            "moviegenres": MovieGenres,
            "watched": Watched,
            "wishlist": Wishlist
        }

class BaseDatabaseBuilder():
    def __init__(self, agent, db_ref):
        self.db_ref = db_ref
        self.agent = agent

    def create_tables(self):
        engine = (
            self.agent._engine if hasattr(self.agent, "_engine") 
            else self.agent.session.get_bind()
        )
        
        self.db_ref.base.metadata.create_all(engine)  # checkfirst=True by default

class PandasDatabaseBuilder(BaseDatabaseBuilder):
    def __init__(self, agent, db_ref, data_dir):
        super().__init__(agent, db_ref)
        
        # Set up data paths
        self.data_dir = Path(data_dir)
        self.ml_latest_dir = self.data_dir / "ml-latest"
        self.processed_dir = self.data_dir / "processed"
        
        # Initialize dataframes
        self.dataframes = {}
        self.initialize_dataframes()

    def build(self):
        self.create_tables()
        self.populate_existing_tables()

    def initialize_dataframes(self):
        """Load CSV files into pandas DataFrames"""
        print("Loading CSV files...")
        
        # Load movies data
        movies_path = self.ml_latest_dir / "movies.csv"
        if movies_path.exists():
            self.dataframes['movies'] = pd.read_csv(movies_path)
            
        # Load links data
        links_path = self.ml_latest_dir / "links.csv"
        if links_path.exists():
            self.dataframes['links'] = pd.read_csv(links_path)
            
        # Load metadata
        metadata_path = self.processed_dir / "movie_metadata_latest.csv"
        if metadata_path.exists():
            self.dataframes['metadata'] = pd.read_csv(metadata_path)

    def populate_existing_tables(self):
        """Populate database tables from loaded DataFrames"""
        print("Populating database tables...")
        
        if self.agent.session is None:
            self.agent.session = self.agent._Session()
        
        # Populate each table
        if 'movies' in self.dataframes:
            self._populate_movies()
            
        if 'links' in self.dataframes:
            self._populate_links()
            
        if 'metadata' in self.dataframes:
            self._populate_metadata()
            self._populate_actors()
            self._populate_directors()
            self._populate_movie_genres()  # Add this
    
    def _populate_movies(self):
        movies_df = self.dataframes['movies'].copy()
        chunk_size = 1000
        
        for i in range(0, len(movies_df), chunk_size):
            chunk = movies_df.iloc[i:i+chunk_size]
            records = chunk.to_dict('records')
            self.agent.session.bulk_insert_mappings(self.db_ref._references['movies'], records)
        
        self.agent.session.commit()
    
    def _populate_links(self):
        links_df = self.dataframes['links'].copy().fillna('')
        chunk_size = 1000
        
        for i in range(0, len(links_df), chunk_size):
            chunk = links_df.iloc[i:i+chunk_size]
            records = chunk.to_dict('records')
            self.agent.session.bulk_insert_mappings(self.db_ref._references['links'], records)
        
        self.agent.session.commit()
    
    def _populate_metadata(self):
        metadata_df = self.dataframes['metadata'].copy()
        metadata_df['metadataId'] = range(1, len(metadata_df) + 1)
        
        # Select relevant columns
        columns = ['metadataId', 'movieId', 'tmdbId', 'title', 'original_title', 'genres', 'tagline',
                  'description', 'year', 'duration', 'tmdbRating', 'tmdbVoteCount', 'poster_path']
        available_columns = [col for col in columns if col in metadata_df.columns]
        filtered_df = metadata_df[available_columns].fillna('')
        
        chunk_size = 1000
        for i in range(0, len(filtered_df), chunk_size):
            chunk = filtered_df.iloc[i:i+chunk_size]
            records = chunk.to_dict('records')
            self.agent.session.bulk_insert_mappings(self.db_ref._references['metadata'], records)
        
        self.agent.session.commit()
    
    def _populate_actors(self):
        if 'actors' not in self.dataframes['metadata'].columns:
            return
            
        metadata_df = self.dataframes['metadata']
        actor_records = []
        actor_id = 1
        
        for _, row in metadata_df.iterrows():
            if pd.isna(row['actors']) or row['actors'] == '' or row['actors'] == []:
                continue
                
            actors_data = json.loads(row['actors']) if isinstance(row['actors'], str) else row['actors']
            
            for actor in actors_data:
                actor_records.append({
                    'movieActorId': actor_id,
                    'movieId': row['movieId'],
                    'name': actor['name'],
                    'popularity': actor.get('popularity', 0.0)
                })
                actor_id += 1
        
        if actor_records:
            chunk_size = 1000
            for i in range(0, len(actor_records), chunk_size):
                chunk = actor_records[i:i+chunk_size]
                self.agent.session.bulk_insert_mappings(self.db_ref._references['movieactors'], chunk)
            
            self.agent.session.commit()
    
    def _populate_directors(self):
        if 'directors' not in self.dataframes['metadata'].columns:
            return
            
        metadata_df = self.dataframes['metadata']
        director_records = []
        director_id = 1
        
        for _, row in metadata_df.iterrows():
            if pd.isna(row['directors']) or row['directors'] == '' or row['directors'] == []:
                continue

            directors_data = json.loads(row['directors']) if isinstance(row['directors'], str) else row['directors']
            
            for director in directors_data:
                director_records.append({
                    'movieDirectorId': director_id,
                    'movieId': row['movieId'],
                    'name': director['name'],
                    'popularity': director.get('popularity', 0.0)
                })
                director_id += 1

        
        if director_records:
            chunk_size = 1000
            for i in range(0, len(director_records), chunk_size):
                chunk = director_records[i:i+chunk_size]
                self.agent.session.bulk_insert_mappings(self.db_ref._references['moviedirectors'], chunk)
            
            self.agent.session.commit()
    
    def _populate_movie_genres(self):
        """
        Create movie_genres table with individual (movieId, genre) pairs.
        Splits pipe-separated genres into separate rows.
        """
        print("Populating movie_genres table...")
        
        metadata_df = self.dataframes['metadata']
        genre_records = []
        
        for _, row in metadata_df.iterrows():
            if pd.isna(row['genres']) or row['genres'] == '':
                continue
            
            # Split pipe-separated genres
            genres = row['genres'].split('|')
            
            for genre in genres:
                genre_clean = genre.strip().lower()
                if genre_clean:  # Skip empty strings
                    genre_records.append({
                        'movieId': row['movieId'],
                        'genre': genre_clean
                    })
        
        print(f"Inserting {len(genre_records)} movie-genre pairs...")
        
        if genre_records:
            chunk_size = 1000
            for i in range(0, len(genre_records), chunk_size):
                chunk = genre_records[i:i+chunk_size]
                self.agent.session.bulk_insert_mappings(self.db_ref._references['moviegenres'], chunk)
            
            self.agent.session.commit()
            print(f"Successfully inserted {len(genre_records)} genre records")