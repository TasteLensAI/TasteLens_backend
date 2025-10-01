from sqlalchemy import Column, Integer, String, DateTime, Boolean, Text, Float, SmallInteger
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

class BaseDatasetBuilder:
    def __init__(self, agent):
        self.agent = agent
        self.base = declarative_base()
        self._define_tables()

    def _define_tables(self):
        """Define all database tables"""
        
        class Movies(self.base):
            __tablename__ = "movies"
            movieId = Column(Integer, primary_key=True)
            title = Column(String(255))
            genres = Column(String(500))

        class Links(self.base):
            __tablename__ = "links"
            movieId = Column(Integer, primary_key=True)
            imdbId = Column(String)
            tmdbId = Column(String)

        class Users(self.base):
            __tablename__ = "users"
            
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
            # is_verified = Column(Boolean, default=False, nullable=False)
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
            "users": Users
        }

    def create_tables(self):
        engine = (
            self.agent._engine if hasattr(self.agent, "_engine") 
            else self.agent.session.get_bind()
        )
        
        self.base.metadata.create_all(engine)  # checkfirst=True by default

class PandasDatasetBuilder(BaseDatasetBuilder):
    def __init__(self, agent):
        super().__init__(agent)

        self.initialize_dataframes()
        self.populate_existing_tables()

    def initialize_dataframes(self):
        pass

    def populate_existing_tables(self):
        pass