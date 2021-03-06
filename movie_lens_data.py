import pandas as pd
import os


MOVIE_LENS_DIR = 'movie_lens'
ML_DIR = {
        '100k': 'ml-100k',
        '1m': 'ml-1m',
        '10m': 'ml-10m',
        '20m': 'ml-20m',
        '25m': 'ml-25m',
        }
ML_SEP = {
        '100k': ',',
        '1m': '::',
        '10m': '::',
        '20m': ',',
        '25m': ',',
        }
ML_EXT = {
        '100k': '.csv',
        '1m': '.dat',
        '10m': '.dat',
        '20m': '.csv',
        '25m': '.csv',
        }
ML_FILES = {
        '100k': ['ratings', 'movies', 'tags', 'links'],
        '1m': ['ratings', 'movies', 'users'],
        '10m': ['ratings', 'movies', 'tags'],
        '20m': ['ratings', 'movies', 'tags', 'links'],
        '25m': ['ratings', 'movies', 'tags', 'links', 'genome-scores', 'genome-tags'],
        }

def _get_movie_lens_dict(ml_dir, files, ext, sep):
    engine = 'c' if len(sep) == 1 else 'python'
    data = {
            f: pd.read_csv(
                os.path.join(ml_dir, f + ext),
                sep=sep,
                engine=engine,
                )
            for f in files}
    if len(files) == 1:
        return data['ratings']
    return data

def get_movie_lens(ver, only_ratings=True):
    ml_dir = os.path.join(MOVIE_LENS_DIR, ML_DIR[ver])
    files = ['ratings'] if only_ratings else ML_FILES[ver]
    return _get_movie_lens_dict(ml_dir, files, ML_EXT[ver], ML_SEP[ver])

