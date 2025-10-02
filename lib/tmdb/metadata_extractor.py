import os
import requests

from dotenv import load_dotenv
from datetime import datetime

class TMDBMetadataExtractor():
    def __init__(self):
        self.api_base = "https://api.themoviedb.org/3"
        self.config = None
        load_dotenv()

    @property
    def api_key(self):
        return os.environ['TMDB_API_KEY']
    
    @property
    def api_configuration(self):
        if self.config:
            return self.config
        
        configuration_endpoint = f"{self.api_base}/configuration"
        r = requests.get(configuration_endpoint, params={
            "api_key": self.api_key
        })

        r.raise_for_status()
        
        self.config = r.json()
        return r.json()
    
    def get_metadata(self, tmdbId, append_to_response = []):#
        assert isinstance(append_to_response, list)

        api_endpoint = f"{self.api_base}/movie/{tmdbId}"

        r = None
        if len(append_to_response) > 0:
            r = requests.get(api_endpoint, params={
                "api_key": self.api_key,
                "append_to_response": ",".join([x.lower() for x in append_to_response])
            })
        else:
            r = requests.get(api_endpoint, params={
                "api_key": self.api_key
            })

        r.raise_for_status()

        r = r.json()
        metadata = {
            "title": r.get("title", ""),
            "original_title": r.get("original_title", ""),
            "tagline": r.get("tagline", ""),
            "coverImage": r.get("poster_path", ""),
            "tmdbRating": r.get("vote_average", None),
            "tmdbVoteCount": r.get("vote_count", None),
            "genres": [x["name"] for x in r.get("genres", [])],
            "year": datetime.strptime(r["release_date"], "%Y-%m-%d").year if "release_date" in r else None,
            "directors": (
                [
                    {
                        "name": x["name"], 
                        "popularity": x["popularity"]
                    } 
                    for x in r["credits"]["crew"] if x["job"].lower() == "director"
                ]
                if "credits" in r else []
            ),
            "actors": (
                [
                    {
                        "name": x["name"], 
                        "popularity": x["popularity"]
                    } 
                    for x in r["credits"]["cast"] if "acting" in x["known_for_department"].lower()
                ]
                if "credits" in r else []
            ),
            "duration": r.get("runtime", None),
            "overview": r.get("overview", ""),
            "imdbId": r.get("imdb_id", None)
        }

        return r, metadata