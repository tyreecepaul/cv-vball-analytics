import os

MODEL_PATHS = {
    'volleyball': 'data/models/volleyball_yolov8.pt',
    'players': 'data/models/players_yolov8.pt', 
    'actions': 'data/models/actions_yolov8.pt',
    'court': 'data/models/court_yolov8_seg.pt'
}

ROBOFLOW_CONFIG = {
    'api_key': 'g7xKIGOSwZGBz8SlVM3G',
    'workspace': 'shukur-sabzaliev-bh7pq',
    'projects': {
        'volleyball': 'volleyball-tracking',
        'players': 'players-dataset', 
        'actions': 'volleyball-actions',
        'court': 'court-segmented'
    }
}

BYTETRACK_CONFIG = {
    'track_thresh': 0.5,
    'track_buffer': 30,
    'match_thresh': 0.8,
    'frame_rate': 30
}