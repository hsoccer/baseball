import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
INTEGRATED_DATA_DIR = os.path.join(DATA_DIR, "integrated")
DETAIL_DATA_DIR = os.path.join(DATA_DIR, "game_detail_data_npb")
SCORE_DATA_DIR = os.path.join(DATA_DIR, "game_score_data_npb")
DETAIL_DATA_DIR_MLB = os.path.join(DATA_DIR, "game_detail_data_mlb")
SCORE_DATA_DIR_MLB = os.path.join(DATA_DIR, "game_score_data_mlb")