import numpy as np

def calculate_distance_travelled(positions):
    dist = 0.0
    for i in range(1, len(positions)):
        dist += np.linalg.norm(np.array(positions[i]) - np.array(positions[i-1]))
    return dist

def calculate_average_speed(positions, fps):
    dist = calculate_distance_travelled(positions)  # pixels or meters
    time_s = len(positions) / fps
    return dist / time_s if time_s > 0 else 0
