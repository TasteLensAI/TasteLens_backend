import pandas as pd
import os

from functools import wraps

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

# from utils import TYPE_PASSWORD_SOURCE, read_password

class BaseSQLAgent():
    def __init__(self, source_args = {}, echo = False, connect_args = {}):
        self.source_args = source_args
        self.echo = echo
        self.connect_args = connect_args

        self.url = self.construct_url_from_source()

        self._engine = create_engine(self.url, echo=self.echo, connect_args=self.connect_args)
        self._Session = sessionmaker(bind=self._engine, expire_on_commit=False, class_=Session)

        self.session = None

    def construct_url_from_source(self):
        return None
    
    def dispose(self):
        if self.session:
            self.session.close()

        self._engine.dispose()
    
    def __enter__(self):
        self.session = self._Session()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        try:
            if exc_value:
                self.session.rollback()
        finally:
            self.dispose()

        return False
    
    @staticmethod
    def if_session_alive(func):
        @wraps(func)
        def wrap(self, *args, **kwargs):
            assert self._engine is not None, "Engine is not initialized."
            
            if self.session is not None:
                self.session = self._Session()

            return func(self, *args, **kwargs)

        return wrap
    
class SQLiteAgent(BaseSQLAgent):
    def __init__(self, source_args={}, echo=False, connect_args={}):
        super().__init__(source_args, echo, connect_args)

    def construct_url_from_source(self):
        assert "name" in self.source_args, "Argument 'name' must be present in the source arguments."
        self.name = self.source_args["name"]

        return f"sqlite:///{self.name}.db"
    
class PostgreSQLAgent(BaseSQLAgent):
    def __init__(self, source_args={}, echo=False, connect_args={}):
        # Default connect_args for Supabase connection pooler
        if not connect_args:
            connect_args = {
                "sslmode": "require",
                "connect_timeout": 10
            }
        super().__init__(source_args, echo, connect_args)

    def construct_url_from_source(self):
        assert "host" in self.source_args, "Argument 'host' must be present in the source arguments."
        assert "port" in self.source_args, "Argument 'port' must be present in the source arguments."
        assert "user" in self.source_args, "Argument 'user' must be present in the source arguments."
        
        self.host = self.source_args["host"]
        self.port = self.source_args["port"]
        self.user = self.source_args["user"]
        
        # Get password from environment variable
        password = os.getenv("POSTGRES_PASSWORD")
        if not password:
            raise ValueError("POSTGRES_PASSWORD environment variable not set")
        
        # Get database name (default to postgres)
        database = self.source_args.get("database", "postgres")

        return f"postgresql://{self.user}:{password}@{self.host}:{self.port}/{database}"