#!/usr/bin/env python3
"""
MovieLens TMDB Metadata Extractor

This script reads the MovieLens links.csv file and uses the TMDB API to fetch
detailed metadata for each movie, then saves it to a CSV file for database import.
"""

import sys
import os
import logging
import pandas as pd
import time
import json
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Optional, Dict, Any

from lib.tmdb.metadata_extractor import TMDBMetadataExtractor

@dataclass
class MovieRequest:
    """Data structure for movie metadata requests"""
    movie_id: int
    tmdb_id: int

@dataclass
class MovieResult:
    """Data structure for raw TMDB API responses"""
    movie_id: int
    tmdb_id: int
    raw_response: Optional[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]]
    error: Optional[str] = None
    fetched_at: datetime = None

    def __post_init__(self):
        if self.fetched_at is None:
            self.fetched_at = datetime.now()

@dataclass
class ProcessedMovie:
    """Data structure for processed/enhanced metadata"""
    enhanced_metadata: Dict[str, Any]
    original_result: MovieResult

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
    def __init__(self, data_dir="data/ml-latest", output_dir="data/processed", 
                 fetch_threads=5, process_threads=3, queue_size=100):
        
        self.logger = logging.getLogger("GenerateMetadata")
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.progress_dir = self.output_dir / "checkpoints"
        
        self.output_dir.mkdir(exist_ok=True)
        self.progress_dir.mkdir(exist_ok=True)
        
        # Threading configuration
        self.fetch_threads = fetch_threads
        self.process_threads = process_threads
        self.queue_size = queue_size
        
        # Shared queues for producer-consumer pattern
        self.request_queue = queue.Queue(maxsize=queue_size)
        self.result_queue = queue.Queue()
        self.processed_queue = queue.Queue()
        
        # Thread synchronization
        self.shutdown_event = threading.Event()
        self.fetch_complete = threading.Event()
        self.process_complete = threading.Event()
        
        # Progress tracking
        self.fetch_progress = {'completed': 0, 'failed': 0}
        self.process_progress = {'completed': 0, 'failed': 0}
        self.progress_lock = threading.Lock()
        
        # Store failed movies for reporting
        self.failed_movies_list = []
        
        # Progress bars (will be initialized in generate_metadata_csv)
        self.fetch_pbar = None
        self.process_pbar = None
        
        # Rate limiting (50 requests per second range)
        self.requests_per_window = 400  # Be conservative
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

    def fetch_worker(self, worker_id: int):
        """Worker thread for Operation 1: Fetch TMDB metadata"""
        tmdb_extractor = TMDBMetadataExtractor()
        header = f"[FETCH   WORKER {str(worker_id).zfill(2)}]:\t"
        
        get_log_msg = lambda x: header + x
        printi = lambda x: self.logger.info(get_log_msg(x))
        printw = lambda x: self.logger.warning(get_log_msg(x))
        printe = lambda x: self.logger.error(get_log_msg(x))

        printi("Started")
                
        while not self.shutdown_event.is_set():
            request = self._get_request_from_queue()
            if request is None:
                continue
            
            if self._is_poison_pill(request):
                break
                
            self._handle_rate_limiting(printi)
            self._process_fetch_request(request, tmdb_extractor, printw)
            self.request_queue.task_done()
        
        printi("Stopped")
    
    def _get_request_from_queue(self):
        """Get request from queue with timeout, return None if empty or error"""
        try:
            return self.request_queue.get(timeout=1.0)
        except queue.Empty:
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error getting request: {e}")
            return None
    
    def _is_poison_pill(self, request):
        """Check if request is poison pill (shutdown signal)"""
        return request is None
    
    def _handle_rate_limiting(self, printi):
        """Handle rate limiting with early return if no action needed"""
        with self._rate_limit_lock:
            if not self.window_stats.is_rate_limit_reached:
                return
                
            freeze_duration = self.window_stats.seconds_to_expire + 0.1
            if freeze_duration <= 0:
                return
                
            printi(f"Rate limiting - sleeping for {freeze_duration:.1f}s")
            time.sleep(freeze_duration)
            self.window_stats.reset_current_window()
    
    def _process_fetch_request(self, request, tmdb_extractor, printw):
        """Process a single fetch request"""
        try:
            result = self._fetch_metadata_for_request(request, tmdb_extractor)
            self._update_fetch_success()
        except Exception as e:
            result = self._create_error_result(request, e)
            self._update_fetch_failure()
            printw(f"Error fetching TMDB ID {request.tmdb_id}: {e}")
        
        self.result_queue.put(result)
    
    def _fetch_metadata_for_request(self, request, tmdb_extractor):
        """Fetch metadata and create result object"""
        raw_response, metadata = tmdb_extractor.get_metadata(
            request.tmdb_id, 
            append_to_response=["credits"]
        )
        
        # Update rate limiting window
        with self._rate_limit_lock:
            self.window_stats.update_current_window(datetime.now())
        
        return MovieResult(
            movie_id=request.movie_id,
            tmdb_id=request.tmdb_id,
            raw_response=raw_response,
            metadata=metadata
        )
    
    def _create_error_result(self, request, error):
        """Create error result object"""
        return MovieResult(
            movie_id=request.movie_id,
            tmdb_id=request.tmdb_id,
            raw_response=None,
            metadata=None,
            error=str(error)
        )
    
    def _update_fetch_success(self):
        """Update fetch success progress"""
        with self.progress_lock:
            self.fetch_progress['completed'] += 1
    
    def _update_fetch_failure(self):
        """Update fetch failure progress"""
        with self.progress_lock:
            self.fetch_progress['failed'] += 1

    def process_worker(self, worker_id: int):
        """Worker thread for Operation 2: Process metadata into CSV format"""
        header = f"[PROCESS WORKER {str(worker_id).zfill(2)}]:\t"
        
        get_log_msg = lambda x: header + x
        printi = lambda x: self.logger.info(get_log_msg(x))
        printw = lambda x: self.logger.warning(get_log_msg(x))
        printe = lambda x: self.logger.error(get_log_msg(x))

        printi("Started")
        
        while not self.shutdown_event.is_set() or not self.result_queue.empty():
            result = self._get_result_from_queue()
            if result is None:
                continue
                
            if self._is_poison_pill(result):
                break
                
            self._process_single_result(result, printw, printe)
            self.result_queue.task_done()
        
        printi("Stopped")
    
    def _get_result_from_queue(self):
        """Get result from queue with timeout, return None if empty or error"""
        try:
            return self.result_queue.get(timeout=1.0)
        except queue.Empty:
            return None
        except Exception as e:
            self.logger.error(f"Unexpected error getting result: {e}")
            return None
    
    def _process_single_result(self, result, printw, printe):
        """Process a single result - either success or failure"""
        try:
            if self._is_successful_result(result):
                self._process_successful_result(result)
            else:
                self._process_failed_result(result, printw)
        except Exception as e:
            self._handle_processing_error(result, e, printe)
    
    def _is_successful_result(self, result):
        """Check if result represents a successful fetch"""
        return result.error is None and result.metadata
    
    def _process_successful_result(self, result):
        """Process a successful result into enhanced metadata"""
        enhanced_metadata = self._create_enhanced_metadata(result)
        processed_movie = ProcessedMovie(
            enhanced_metadata=enhanced_metadata,
            original_result=result
        )
        
        self.processed_queue.put(processed_movie)
        self._update_process_success()
    
    def _process_failed_result(self, result, printw):
        """Process a failed result"""
        printw(f"Skipping failed movie {result.movie_id} (TMDB: {result.tmdb_id})")
        
        failed_movie_info = {
            'movieId': result.movie_id,
            'tmdbId': result.tmdb_id,
            'error': result.error or 'Unknown error'
        }
        self.failed_movies_list.append(failed_movie_info)
        self._update_process_failure()
    
    def _handle_processing_error(self, result, error, printe):
        """Handle unexpected processing errors"""
        printe(f"Error processing movie {result.movie_id}: {error}")
        
        failed_movie_info = {
            'movieId': result.movie_id,
            'tmdbId': result.tmdb_id,
            'error': f"Processing error: {str(error)}"
        }
        self.failed_movies_list.append(failed_movie_info)
        self._update_process_failure()
    
    def _create_enhanced_metadata(self, result):
        """Create enhanced metadata dictionary from result"""
        metadata = result.metadata
        metadata['movieId'] = result.movie_id
        metadata['tmdbId'] = result.tmdb_id
        
        return {
            'movieId': result.movie_id,
            'tmdbId': result.tmdb_id,
            'title': metadata.get('title', ''),
            'original_title': metadata.get('original_title', ''),
            'tagline': metadata.get('tagline', ''),
            'description': metadata.get('overview', ''),
            'year': metadata.get('year'),
            'duration': metadata.get('duration'),
            'tmdbRating': metadata.get('tmdbRating'),
            'tmdbVoteCount': metadata.get('tmdbVoteCount'),
            'genres': '|'.join(metadata.get('genres', [])),
            'poster_path': metadata.get('coverImage', ''),
            'directors': json.dumps(metadata.get('directors', [])),
            'actors': json.dumps(metadata.get('actors', [])),
            'fetched_at': result.fetched_at.isoformat()
        }
    
    def _update_process_success(self):
        """Update process success progress"""
        with self.progress_lock:
            self.process_progress['completed'] += 1
    
    def _update_process_failure(self):
        """Update process failure progress"""
        with self.progress_lock:
            self.process_progress['failed'] += 1

    def generate_metadata_csv(self, max_movies=None, save_progress_every=100, start_from=0):
        """Generate metadata CSV for all movies using multi-threading"""
        # Load links data
        links_df = self.load_links_data()
        
        if max_movies:
            links_df = links_df.iloc[start_from:start_from + max_movies]
        else:
            links_df = links_df.iloc[start_from:]
        
        print(f"🎬 Processing {len(links_df)} movies (starting from index {start_from})")
        print(f"🧵 Using {self.fetch_threads} fetch threads and {self.process_threads} process threads")
        
        # Initialize rate limiting
        self.window_stats.init_current_window()
        self._rate_limit_lock = threading.Lock()
        
        # Start worker threads
        fetch_threads = []
        process_threads = []
        
        # Start fetch workers
        for i in range(self.fetch_threads):
            thread = threading.Thread(target=self.fetch_worker, args=(i,))
            thread.daemon = True
            thread.start()
            fetch_threads.append(thread)
        
        # Start process workers
        for i in range(self.process_threads):
            thread = threading.Thread(target=self.process_worker, args=(i,))
            thread.daemon = True
            thread.start()
            process_threads.append(thread)
        
        # Progress tracking setup
        metadata_list = []
        total_movies = len(links_df)
        
        # Reset failed movies list for this run
        self.failed_movies_list = []
        
        # Initialize tqdm progress bars
        self.fetch_pbar = tqdm(
            total=total_movies, 
            desc="🔄 Fetching", 
            unit="movies",
            position=0,
            leave=True
        )
        self.process_pbar = tqdm(
            total=total_movies, 
            desc="⚙️  Processing", 
            unit="movies",
            position=1,
            leave=True
        )
        
        # Start progress monitoring thread
        progress_thread = threading.Thread(target=self._progress_monitor, args=(total_movies,))
        progress_thread.daemon = True
        progress_thread.start()
        
        # Producer: Add all requests to the queue
        print("📥 Adding movie requests to queue...")
        for idx, row in links_df.iterrows():
            request = MovieRequest(
                movie_id=row['movieId'],
                tmdb_id=row['tmdbId']
            )
            self.request_queue.put(request)
        
        print(f"✅ Added {len(links_df)} requests to queue")
        
        # Consumer: Collect results
        metadata_list = []
        
        try:
            while not self._is_all_work_complete(len(links_df)):
                processed_movie = self._get_processed_movie(save_progress_every, metadata_list, start_from)
                if processed_movie is None:
                    continue
                    
                metadata_list.append(processed_movie.enhanced_metadata)
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user, shutting down...")
            self.shutdown_event.set()
        
        self._cleanup_workers(fetch_threads, process_threads)
        self._log_and_save_results(metadata_list, start_from)
        
        return metadata_list, self.failed_movies_list
    
    def _is_all_work_complete(self, total_movies):
        """Check if all work is complete using progress tracking"""
        with self.progress_lock:
            total_fetched = self.fetch_progress['completed'] + self.fetch_progress['failed']
            total_processed = self.process_progress['completed'] + self.process_progress['failed']
        
        if total_fetched >= total_movies and total_processed >= total_fetched:
            print(f"✅ All work completed: Fetched {total_fetched}, Processed {total_processed}")
            return True
        
        if total_fetched >= total_movies and total_processed < total_fetched:
            print(f"⏳ Waiting for processing to complete: {total_processed}/{total_fetched}")
            time.sleep(1)
        
        return False
    
    def _get_processed_movie(self, save_progress_every, metadata_list, start_from):
        """Get a processed movie from queue with timeout"""
        try:
            processed_movie = self.processed_queue.get(timeout=2.0)
            self.processed_queue.task_done()
            
            # Save progress periodically
            if self._should_save_progress(save_progress_every, metadata_list):
                self._save_progress(metadata_list, self.failed_movies_list, start_from)
            
            return processed_movie
        except queue.Empty:
            return None
    
    def _should_save_progress(self, save_progress_every, metadata_list):
        """Check if progress should be saved"""
        return (save_progress_every > 0 and 
                len(metadata_list) % save_progress_every == 0 and 
                len(metadata_list) > 0)
    
    def _cleanup_workers(self, fetch_threads, process_threads):
        """Signal shutdown and cleanup workers"""
        print("🧹 Shutting down workers...")
        self.shutdown_event.set()
        
        # Add poison pills to queues
        for _ in range(self.fetch_threads):
            self.request_queue.put(None)
        for _ in range(self.process_threads):
            self.result_queue.put(None)
        
        # Wait for all threads to complete
        for thread in fetch_threads + process_threads:
            thread.join(timeout=5.0)
        
        # Close progress bars
        if self.fetch_pbar:
            self.fetch_pbar.close()
        if self.process_pbar:
            self.process_pbar.close()
    
    def _log_and_save_results(self, metadata_list, start_from):
        """Log final summary and save results"""
        final_msg = (f"FINAL SUMMARY: Successfully processed {len(metadata_list)} movies, "
                    f"Failed: {len(self.failed_movies_list)} movies")
        self.logger.info(final_msg)
        print(f"\n{final_msg}")
        
        self._save_final_results(metadata_list, self.failed_movies_list, start_from)
    
    def _progress_monitor(self, total_movies):
        """Monitor and display progress using tqdm and logging"""
        last_fetch_completed = 0
        last_fetch_failed = 0
        last_process_completed = 0
        last_process_failed = 0
        
        while not self.shutdown_event.is_set():
            with self.progress_lock:
                fetch_completed = self.fetch_progress['completed']
                fetch_failed = self.fetch_progress['failed']
                process_completed = self.process_progress['completed']
                process_failed = self.process_progress['failed']
            
            current_fetch_total = fetch_completed + fetch_failed
            current_process_total = process_completed + process_failed
            
            # Update fetch progress bar
            if current_fetch_total != (last_fetch_completed + last_fetch_failed):
                fetch_delta = current_fetch_total - (last_fetch_completed + last_fetch_failed)
                self.fetch_pbar.update(fetch_delta)
                
                # Update description with success/failure counts
                self.fetch_pbar.set_postfix({
                    'Success': fetch_completed,
                    'Failed': fetch_failed,
                    'Rate': f"{self.fetch_pbar.format_dict.get('rate', 0):.1f}/s"
                })
            
            # Update process progress bar
            if current_process_total != (last_process_completed + last_process_failed):
                process_delta = current_process_total - (last_process_completed + last_process_failed)
                self.process_pbar.update(process_delta)
                
                self.process_pbar.set_postfix({
                    'Success': process_completed,
                    'Failed': process_failed,
                    'Rate': f"{self.process_pbar.format_dict.get('rate', 0):.1f}/s"
                })
            
            # Log progress every 1000 items or when significant changes occur
            if (current_fetch_total % 1000 == 0 and current_fetch_total != (last_fetch_completed + last_fetch_failed)) or \
               (current_process_total % 1000 == 0 and current_process_total != (last_process_completed + last_process_failed)):
                progress_msg = (f"Progress Update: Fetched {current_fetch_total}/{total_movies} "
                               f"(Success: {fetch_completed}, Failed: {fetch_failed}), "
                               f"Processed: {current_process_total}/{total_movies} "
                               f"(Success: {process_completed}, Failed: {process_failed})")
                self.logger.info(progress_msg)
            
            # Update last values
            last_fetch_completed = fetch_completed
            last_fetch_failed = fetch_failed
            last_process_completed = process_completed
            last_process_failed = process_failed
            
            if current_fetch_total >= total_movies and current_process_total >= current_fetch_total:
                break
                
            time.sleep(1)  # Update every 1 second for more responsive UI
    
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
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # Console output
            logging.FileHandler('./logs/metadata_generation.log')  # File output
        ]
    )
    
    parser = argparse.ArgumentParser(description="Generate TMDB metadata for MovieLens movies")
    parser.add_argument("--max-movies", type=int, help="Maximum number of movies to process")
    parser.add_argument("--save-progress-every", type=int, default=10000, help="Period of checkpoints for intermediate results")
    parser.add_argument("--queue_size", type=int, default=1000, help="Queue size")
    parser.add_argument("--start-from", type=int, default=0, help="Start processing from this index")
    parser.add_argument("--data-dir", default="data/ml-latest", help="MovieLens data directory")
    parser.add_argument("--output-dir", default="data/processed", help="Output directory")
    parser.add_argument("--fetch-threads", type=int, default=4, help="Number of threads for fetching TMDB data")
    parser.add_argument("--process-threads", type=int, default=4, help="Number of threads for processing data")
    
    args = parser.parse_args()
    
    print("🎬 MovieLens TMDB Metadata Generator")
    print("=" * 50)
    
    try:
        generator = MovieLensMetadataGenerator(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            fetch_threads=args.fetch_threads,
            process_threads=args.process_threads,
            queue_size=args.queue_size
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
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
