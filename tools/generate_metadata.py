#!/usr/bin/env python3
"""
MovieLens TMDB Metadata Extractor

This script reads the MovieLens links.csv file and uses the TMDB API to fetch
detailed metadata for each movie, then saves it to a CSV file for database import.
"""

import sys
import os
import pandas as pd
import time
import json
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from lib.tmdb.metadata_extractor import TMDBMetadataExtractor

class RequestWindowStatistics:
    def __init__(self, num_requests, duration):
        self.num_requests = num_requests
        self.duration = duration
        self.window = None

    @property
    def seconds_passed(self):
        return (self.window["last_tick_time"] - self.window["start_time"]).total_seconds()

    @property
    def is_current_window_expired(self):
        return self.seconds_passed >= self.duration
    
    @property
    def is_rate_limit_reached(self):
        return self.window["n_ticks"] >= self.num_requests
    
    @property
    def seconds_to_expire(self):
        return self.duration - self.seconds_passed

    def init_current_window(self):
        now = datetime.now()
        self.window = {
            "start_time": now,
            "n_ticks": 0,
            "last_tick_time": now
        }

    def reset_current_window(self):
        return self.init_current_window()

    def update_current_window(self, request_date):
        self.window["last_tick_time"] = request_date
        self.window["n_ticks"] += 1

        if self.is_current_window_expired:
            self.reset_current_window()

class MovieLensMetadataGenerator:
    def __init__(self, data_dir="data/ml-latest", output_dir="data/processed"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.progress_dir = self.output_dir / "checkpoints"
        
        self.output_dir.mkdir(exist_ok=True)
        self.progress_dir.mkdir(exist_ok=True)
        
        # Initialize TMDB extractor
        self.tmdb_extractor = TMDBMetadataExtractor()
        
        # Rate limiting (50 requests per second range)
        self.requests_per_window = 300  # Be conservative
        self.window_duration = 10  # seconds
        self.window_stats = RequestWindowStatistics(
            num_requests=self.requests_per_window,
            duration=self.window_duration
        )
        
    def load_links_data(self):
        """Load MovieLens links.csv file"""
        links_file = self.data_dir / "links.csv"
        
        if not links_file.exists():
            raise FileNotFoundError(f"not found: {links_file}")
        
        print(f"📁 Loading links data from {links_file}")
        df = pd.read_csv(links_file)
        n_total = len(df)
        
        # Filter out movies without TMDB IDs
        df = df.dropna(subset=['tmdbId'])
        n_filtered = len(df)
        
        df['tmdbId'] = df['tmdbId'].astype(int)
        percentage = (n_filtered / n_total) * 100
        
        print(f"📊 Out of {n_total} entries, found {n_filtered} movies with TMDB IDs ({percentage:.2f}%)")
        return df

    def check_rate_limit(func):
        def wrapper(self, *args, **kwargs):
            if self.window_stats.is_rate_limit_reached:
                freeze_duration = self.window_stats.seconds_to_expire + 0.1
                if freeze_duration > 0:
                    print(f"⏱️  Rate limiting: sleeping for {freeze_duration:.1f} seconds")
                    import time; time.sleep(freeze_duration)
                    self.window_stats.reset_current_window()
            return func(self, *args, **kwargs)
        return wrapper
    
    @check_rate_limit
    def fetch_movie_metadata(self, tmdb_id, movie_id):
        """Fetch metadata for a single movie"""
        try:
            # Get metadata with credits for directors and actors
            raw_response, metadata = self.tmdb_extractor.get_metadata(
                tmdb_id, 
                append_to_response=["credits"]
            )
            request_exit_date = datetime.now()
            self.window_stats.update_current_window(request_exit_date)

            
            # Add MovieLens ID to the metadata
            metadata['movieId'] = movie_id
            metadata['tmdbId'] = tmdb_id
            
            # Extract additional fields that match your database schema
            enhanced_metadata = {
                'movieId': movie_id,
                'tmdbId': tmdb_id,
                'title': metadata.get('title', ''),
                'original_title': metadata.get('original_title', ''),
                'tagline': metadata.get('title', ''),  # Note: this seems to be overwritten in your extractor
                'description': metadata.get('overview', ''),
                'year': metadata.get('year'),
                'duration': metadata.get('duration'),  # runtime in minutes
                'tmdbRating': metadata.get('tmdbRating'),
                'tmdbVoteCount': metadata.get('tmdbVoteCount'),
                'genres': '|'.join(metadata.get('genres', [])),  # Join genres like MovieLens format
                'poster_path': metadata.get('coverImage', ''),
                'directors': json.dumps(metadata.get('directors', [])),  # Store as JSON
                'actors': json.dumps(metadata.get('actors', [])),  # Store as JSON
                'fetched_at': datetime.now().isoformat()
            }
            
            return enhanced_metadata
            
        except Exception as e:
            print(f"❌ Error fetching metadata for TMDB ID {tmdb_id} (MovieLens ID {movie_id}): {e}")
            return None
    
    def generate_metadata_csv(self, max_movies=None, save_progress_every = 100, start_from=0):
        """Generate metadata CSV for all movies"""
        # Load links data
        links_df = self.load_links_data()
        
        if max_movies:
            links_df = links_df.iloc[start_from:start_from + max_movies]
        else:
            links_df = links_df.iloc[start_from:]
        
        print(f"🎬 Processing {len(links_df)} movies (starting from index {start_from})")
        
        # Initialize results list
        metadata_list = []
        failed_movies = []
        
        # Process each movie with progress bar
        self.window_stats.init_current_window()
        for idx, row in tqdm(links_df.iterrows(), total=len(links_df), desc="Fetching metadata"):
            movie_id = row['movieId']
            tmdb_id = row['tmdbId']
            
            metadata = self.fetch_movie_metadata(tmdb_id, movie_id)
            
            if metadata:
                metadata_list.append(metadata)
            else:
                failed_movies.append({'movieId': movie_id, 'tmdbId': tmdb_id})
            
            # Save progress every 100 movies
            if len(metadata_list) % save_progress_every == 0 and len(metadata_list) > 0:
                self._save_progress(metadata_list, failed_movies, start_from)
        
        # Save final results
        self._save_final_results(metadata_list, failed_movies, start_from)
        
        return metadata_list, failed_movies
    
    def _save_progress(self, metadata_list, failed_movies, start_from):
        """Save progress to temporary files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save successful metadata
        if metadata_list:
            df = pd.DataFrame(metadata_list)
            progress_file = self.progress_dir / f"movie_metadata_progress_{start_from}_{timestamp}.csv"
            df.to_csv(progress_file, index=False)
            print(f"💾 Progress saved: {len(metadata_list)} movies to {progress_file}")
        
        # Save failed movies
        if failed_movies:
            failed_df = pd.DataFrame(failed_movies)
            failed_file = self.progress_dir / f"failed_movies_{start_from}_{timestamp}.csv"
            failed_df.to_csv(failed_file, index=False)
    
    def _save_final_results(self, metadata_list, failed_movies, start_from):
        """Save final results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n📊 Final Results:")
        print(f"  ✅ Successfully fetched: {len(metadata_list)} movies")
        print(f"  ❌ Failed: {len(failed_movies)} movies")
        
        # Save successful metadata
        if metadata_list:
            df = pd.DataFrame(metadata_list)
            output_file = self.output_dir / f"movie_metadata_{start_from}_{timestamp}.csv"
            df.to_csv(output_file, index=False)
            print(f"💾 Metadata saved to: {output_file}")
            
            # Also save as the latest version
            latest_file = self.output_dir / "movie_metadata_latest.csv"
            df.to_csv(latest_file, index=False)
            print(f"💾 Latest version saved to: {latest_file}")
        
        # Save failed movies for retry
        if failed_movies:
            failed_df = pd.DataFrame(failed_movies)
            failed_file = self.output_dir / f"failed_movies_{start_from}_{timestamp}.csv"
            failed_df.to_csv(failed_file, index=False)
            print(f"📝 Failed movies saved to: {failed_file}")


def main():
    """Main function with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate TMDB metadata for MovieLens movies")
    parser.add_argument("--max-movies", type=int, help="Maximum number of movies to process")
    parser.add_argument("--save-progress-every", type=int, default=1000, help="Period of checkpoints for intermediate results")
    parser.add_argument("--start-from", type=int, default=0, help="Start processing from this index")
    parser.add_argument("--data-dir", default="data/ml-latest", help="MovieLens data directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    
    args = parser.parse_args()
    
    print("🎬 MovieLens TMDB Metadata Generator")
    print("=" * 50)
    
    try:
        generator = MovieLensMetadataGenerator(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )
        
        metadata_list, failed_movies = generator.generate_metadata_csv(
            max_movies=args.max_movies,
            save_progress_every=args.save_progress_every,
            start_from=args.start_from
        )
        
        print(f"\n🎉 Process completed!")
        print(f"   Successfully processed: {len(metadata_list)} movies")
        print(f"   Failed: {len(failed_movies)} movies")
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
