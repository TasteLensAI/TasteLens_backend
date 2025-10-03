# Test SQLite database creation with BaseDatasetBuilder using real SQLiteAgent
import sys
import os
import argparse

# Import your real implementations
from lib.sql.builders import BaseDatabaseReference, PandasDatabaseBuilder
from lib.sql.agents import SQLiteAgent

# Test database creation using your SQLiteAgent
def main(args):
    try:
        # Create SQLite agent with your implementation
        source_args = {"name": args.db_name}  # Will create test_tastelens.db
        agent = SQLiteAgent(source_args=source_args, echo=args.echo)
        
        # Use context manager for proper session handling
        with agent as db_agent:
            # Debug: Check if session was created properly
            print(f"Session type: {type(db_agent.session)}")
            print(f"Session: {db_agent.session}")
            
            # Create database builder with your real agent
            db_ref = BaseDatabaseReference()
            builder = PandasDatabaseBuilder(db_agent, db_ref, data_dir=args.data_dir)
            
            # Create all tables
            print("Creating tables...")
            builder.build()
            print("✅ Tables created successfully!")
            
            # Verify tables exist by querying metadata
            print("\n📋 Created tables:")
            for table_name in db_ref.base.metadata.tables.keys():
                print(f"  - {table_name}")
            
            # Get table references
            Movies = db_ref._references["movies"]
            Users = db_ref._references["users"]
            
            # Query back to verify
            movie_count = db_agent.session.query(Movies).count()
            user_count = db_agent.session.query(Users).count()
            
            print(f"\n📊 Database verification:")
            print(f"  - Movies: {movie_count} record(s)")
            print(f"  - Users: {user_count} record(s)")
            
            # Query the inserted movie
            inserted_movie = db_agent.session.query(Movies).filter(Movies.movieId == 1).first()
            if inserted_movie:
                print(f"  - Retrieved movie: '{inserted_movie.title}'")
            
            print("\n🎉 Database creation test completed successfully!")
            print(f"📁 Database file created: {agent.name}.db")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    """Main function with command line argument support"""
    parser = argparse.ArgumentParser(description="Construct the sqlite database with respect to csv files.")
    parser.add_argument("--db-name", type=str, default="tastelens", help="")
    parser.add_argument("--data-dir", type=str, default="./data", help="")
    parser.add_argument("--echo", action="store_true")

    args = parser.parse_args()
    main(args)
