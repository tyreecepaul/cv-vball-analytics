from volleyball_analysis_service import VolleyVisionService
from config import MODEL_PATHS

# Initialize service
service = VolleyVisionService(MODEL_PATHS, use_roboflow=True)

# Process a video
results = service.process_video('raw/match.mp4', output_dir='analysis_output')

# Generate statistics
stats = service.generate_statistics(results)
print(f"Ball detection rate: {stats['ball_detection_rate']:.2%}")