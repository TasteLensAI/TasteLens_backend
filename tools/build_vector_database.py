import faiss
import argparse
import json
import numpy as np
import pandas as pd

from pathlib import Path
from sqlalchemy import func, desc
from lib.sql.builders import BaseDatabaseReference
from lib.sql.agents import SQLiteAgent
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def get_best_actors_subquery(session, ref):
    Actors = ref._references["movieactors"]
    ranked_actors = (
        session.query(
            Actors.movieId,
            Actors.name.label("actor_name"),
            Actors.popularity.label("actor_popularity"),
            func.row_number()
                .over(
                    partition_by=Actors.movieId,
                    order_by=Actors.popularity.desc()
                )
                .label("rank")
        )
        .filter(Actors.popularity.isnot(None))
    ).subquery()
    return ranked_actors

def get_best_directors_subquery(session, ref):
    Directors = ref._references["moviedirectors"]
    ranked_directors = (
        session.query(
            Directors.movieId,
            Directors.name.label("director_name"),
            Directors.popularity.label("director_popularity"),
            func.row_number()
                .over(
                    partition_by=Directors.movieId,
                    order_by=Directors.popularity.desc()
                )
                .label("rank")
        )
        .filter(Directors.popularity.isnot(None))
    ).subquery()
    return ranked_directors

def get_movies_with_top_cast(limit=1000, rank=5):
    """
    Get movies with metadata and top 5 actors/directors by popularity
    """
    ref = BaseDatabaseReference()
    Movies = ref._references["movies"]
    Metadata = ref._references["metadata"]

    with SQLiteAgent({"name": "tastelens"}) as agent:
        session = agent.session

        ranked_actors = get_best_actors_subquery(session, ref)
        ranked_directors = get_best_directors_subquery(session, ref)

        query = (
            session.query(
                Movies.movieId,
                Metadata.title,
                Metadata.genres,
                Metadata.description,
                Metadata.duration,
                Metadata.year,
                Metadata.tmdbRating,
                Metadata.tmdbVoteCount,
                Metadata.tagline,
                func.group_concat(ranked_actors.c.actor_name, ", ").label(f"top_{rank}_actors"),
                func.group_concat(ranked_directors.c.director_name, ", ").label(f"top_{rank}_directors"),
            )
            .join(Metadata, Movies.movieId == Metadata.movieId)
            .outerjoin(ranked_actors, Movies.movieId == ranked_actors.c.movieId)
            .outerjoin(ranked_directors, Movies.movieId == ranked_directors.c.movieId)
            .filter(
                Metadata.title.isnot(None),
                Metadata.description.isnot(None),
                Metadata.genres.isnot(None),
                Metadata.genres != "",
                Metadata.duration > 0,
                ((ranked_actors.c.rank <= rank) | (ranked_actors.c.rank.is_(None))),
                ((ranked_directors.c.rank <= rank) | (ranked_directors.c.rank.is_(None))),
            )
            .group_by(Movies.movieId)
        )

        if limit > 0:
            query = query.limit(limit)

        return query.all()

def string_to_list(x):
    # Get unique elemets without breaking the order
    raw_list = [x.strip() for x in x.split(',')]
    unique_list = []
    for item in raw_list:
        if item not in unique_list:
            unique_list.append(item)

    return unique_list

def get_relevant_tags(genome_df, movie_id, threshold=0.7, top_k=25):
    movie_tags = genome_df[genome_df.movieId == movie_id]
    filtered = movie_tags[movie_tags.relevance >= threshold]
    filtered = filtered.sort_values("relevance", ascending=False)
    return filtered.tag.tolist()[:top_k]

def get_queries(movies_df):
    template = (
"""Movie recommendation: This {year} {genre_phrase} film lasts {duration} minutes. 
{description_sentence}
Starring: {actor_phrase}. Directed by {director_phrase}.
Tagline: "{tagline}".
Rated {tmdbRating:.1f}/10 on TMDB with {tmdbVoteCount} votes.
Tags: {tags_phrase}.
"""
    )


    movie_queries = []
    for i, row in tqdm(movies_df.iterrows()):
        genre_phrase = row.genres.lower().replace('|', ', ')
        actor_phrase = ", ".join(row.actors) if row.actors else "unknown"
        director_phrase = row.directors[0] if row.directors else "unknown"
        tags_phrase = row.relevant_tags

        lm_query = template.format(
            year=row.year if row.year else "",
            genre_phrase=genre_phrase,
            duration=row.duration,
            description_sentence=row.description if row.description else "",
            actor_phrase=actor_phrase,
            director_phrase=director_phrase,
            tagline = row.tagline if row.tagline else "",
            tmdbRating=row.tmdbRating if row.tmdbRating else 0.0,
            tmdbVoteCount = row.tmdbVoteCount if row.tmdbVoteCount else 0.0,
            tags_phrase = tags_phrase
        )

        movie_queries.append(
            (row.movieId, lm_query)
        )

    return movie_queries

def get_embeddings(model, movie_texts):
    movie_embeddings = model.encode(movie_texts, batch_size=32, show_progress_bar=True)
    return movie_embeddings

def init_vector_db(movie_embeddings):
    d = movie_embeddings.shape[1] 
    index = faiss.IndexFlatL2(d)   # build the index
    index.add(movie_embeddings)

    return index

def save_vector_db(vector_db, output_dir):
    faiss.write_index(vector_db, str(output_dir / "movie_vectors.index"))

    print(f"Saved vector database to {output_dir}")
    print(f"Total movies: {vector_db.ntotal}")

def main(args):
    print("Processing input and output paths...")
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Reading movies with metadata, actors, and directors information...")
    movies_df = pd.DataFrame(get_movies_with_top_cast(limit=-1))
    print(f"Loaded {len(movies_df)} movies")
        
    movies_df.top_5_directors = movies_df.top_5_directors.fillna("")
    movies_df.top_5_actors = movies_df.top_5_actors.fillna("")

    movies_df["directors"] = movies_df.top_5_directors.apply(lambda x: string_to_list(x))
    movies_df["actors"] = movies_df.top_5_actors.apply(lambda x: string_to_list(x))

    print("Reading genome tags from the input path...")
    genome_scores_df = pd.read_csv(data_dir / "genome-scores.csv")
    genome_tags_df = pd.read_csv(data_dir / "genome-tags.csv")

    genome_df = pd.merge(
        left=genome_scores_df,
        right=genome_tags_df,
        on="tagId",
    )

    genome_df = (
        genome_df[genome_df["relevance"] >= args.relevance]
        .sort_values(["movieId", "relevance"], ascending=[True, False])
    )

    print(f"Calculating relevant tasks that have relevance >= {args.relevance}...")
    relevant_tags = (
        genome_df.groupby("movieId")
        .agg({"tag": lambda x: ", ".join(x)})
        .rename(columns={"tag": "relevant_tags"})
        .reset_index()
    )

    movies_df = movies_df.merge(relevant_tags, on="movieId", how="left")
    movies_df["relevant_tags"] = movies_df["relevant_tags"].fillna("")

    print("Creating movie_queries...")
    movie_queries = get_queries(movies_df=movies_df)
    movie_texts = [query[1] for query in movie_queries]
    movie_ids = [query[0] for query in movie_queries]
    
    if len(movie_queries) > 0:
        print("Sample query:")
        print(movie_texts[0])
    
    print("Creating movie embeddings...")
    model = SentenceTransformer('all-mpnet-base-v2', device="cuda:0") 
    movie_embeddings = get_embeddings(model, movie_texts)
    print(f"Done: Creating movie embeddings with dimensionality = {movie_embeddings.shape[1]}")

    print("Creating the vector database...")
    vector_db = init_vector_db(movie_embeddings)

    print("Saving results...")
    save_vector_db(vector_db, output_dir)

    movies_df.to_csv(output_dir / "movies_df.csv", index=False)
    
    with open(output_dir / "queries.json", "w") as f:
        json.dump(dict(movie_queries), f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the vector database (with FAISS) for MovieLens movies"
    )

    parser.add_argument("--data-dir", default="data/ml-latest", help="MovieLens data directory")
    parser.add_argument("--output-dir", default="data/vector-db", help="Output directory")
    parser.add_argument("--relevance", type=float, default=0.7, help="")

    args = parser.parse_args()
    main(args)