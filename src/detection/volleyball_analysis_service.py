import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import torch
from ultralytics import YOLO
import requests
from PIL import Image, ImageDraw

# You'll need to install these packages:
# pip install ultralytics opencv-python numpy matplotlib pillow
# pip install git+https://github.com/ifzhang/ByteTrack.git

@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    confidence: float
    class_id: int
    track_id: Optional[int] = None

@dataclass
class CourtCoordinates:
    corners: List[Tuple[int, int]]  # Court corner coordinates
    net_line: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None  # Net line coordinates

class VolleyVisionService:
    """
    Comprehensive volleyball analysis service combining VolleyVision models with ByteTrack
    """
    
    def __init__(self, model_paths: Dict[str, str], use_roboflow: bool = True, tracker_type: str = "bytetrack"):
        """
        Initialize the volleyball analysis service
        
        Args:
            model_paths: Dictionary with paths to different models
            use_roboflow: Whether to use RoboFlow models (more accurate) or local models (faster)
            tracker_type: Type of tracker to use ("bytetrack", "botsort", "ultralytics")
        """
        self.model_paths = model_paths
        self.use_roboflow = use_roboflow
        self.tracker_type = tracker_type
        
        # Initialize models
        self.volleyball_detector = self._load_volleyball_model()
        self.player_detector = self._load_player_model()
        self.action_detector = self._load_action_model()
        self.court_detector = self._load_court_model()
        
        # Initialize tracker
        self.tracker = self._init_tracker()
        self.track_history = {}  # For storing track histories
        
        # Class mappings
        self.action_classes = {
            0: 'block', 1: 'defense', 2: 'serve', 3: 'set', 4: 'spike'
        }
        
        # Court template for visualization
        self.court_template = self._create_court_template()
        
        # Storage for analysis results
        self.frame_data = []
        
    def _load_volleyball_model(self):
        """Load volleyball detection model"""
        if self.use_roboflow:
            # Use RoboFlow API for volleyball detection
            return self._init_roboflow_model("volleyball-tracking")
        else:
            # Load local YOLOv8 model
            return YOLO(self.model_paths.get('volleyball', 'yolov8n.pt'))
    
    def _load_player_model(self):
        """Load player detection model"""
        if self.use_roboflow:
            return self._init_roboflow_model("players-dataset")
        else:
            return YOLO(self.model_paths.get('players', 'yolov8n.pt'))
    
    def _load_action_model(self):
        """Load action recognition model"""
        if self.use_roboflow:
            return self._init_roboflow_model("volleyball-actions")
        else:
            return YOLO(self.model_paths.get('actions', 'yolov8n.pt'))
    
    def _load_court_model(self):
        """Load court segmentation model"""
        if self.use_roboflow:
            return self._init_roboflow_model("court-segmented")
        else:
            return YOLO(self.model_paths.get('court', 'yolov8n-seg.pt'))
    
    def _init_roboflow_model(self, model_name: str):
        """Initialize RoboFlow model"""
        # Placeholder for RoboFlow API integration
        # You'll need to replace this with actual RoboFlow API calls
        class RoboFlowModel:
            def __init__(self, model_name):
                self.model_name = model_name
                self.api_key = "YOUR_ROBOFLOW_API_KEY"  # Replace with your API key
            
            def predict(self, image):
                # Implement RoboFlow API call here
                # This is a placeholder implementation
                return []
        
        return RoboFlowModel(model_name)
    
    def _init_tracker(self):
        """Initialize tracker based on specified type"""
        if self.tracker_type == "ultralytics":
            # Use YOLOv8's built-in tracking (includes ByteTrack and BoTSORT)
            return "ultralytics_builtin"
        else:
            # Try to initialize standalone ByteTracker
            return self._init_bytetrack()
    def _init_bytetrack(self):
        """Initialize ByteTrack tracker"""
        try:
            # Try different import paths for ByteTracker
            try:
                from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
            except ImportError:
                try:
                    from ByteTrack.yolox.tracker.byte_tracker import BYTETracker
                except ImportError:
                    # Use Ultralytics built-in ByteTracker (available in newer versions)
                    from ultralytics.trackers.byte_tracker import BYTETracker
            
            from types import SimpleNamespace
            
            # ByteTrack configuration
            args = SimpleNamespace()
            args.track_thresh = 0.5
            args.track_buffer = 30
            args.match_thresh = 0.8
            args.mot20 = False
            
            return BYTETracker(frame_rate=30, args=args)
        except ImportError:
            print("ByteTracker not found. Falling back to Ultralytics built-in tracking.")
            return "ultralytics_builtin"
    
    def detect_players_with_tracking(self, frame: np.ndarray) -> List[Detection]:
        """Detect and track players using YOLOv8's built-in tracking"""
        if isinstance(self.player_detector, YOLO):
            # Use YOLOv8's track method which includes ByteTrack
            tracker_config = "bytetrack.yaml" if self.tracker_type == "bytetrack" else "botsort.yaml"
            results = self.player_detector.track(frame, tracker=tracker_config, persist=True)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        track_id = box.id[0].cpu().numpy() if box.id is not None else None
                        
                        detections.append(Detection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=float(conf),
                            class_id=1,
                            track_id=int(track_id) if track_id is not None else None
                        ))
            return detections
        else:
            return []
    
    def _create_court_template(self) -> np.ndarray:
        """Create a volleyball court template for visualization"""
        court = np.zeros((400, 600, 3), dtype=np.uint8)
        court[:] = (34, 139, 34)  # Green background
        
        # Court boundaries (simplified)
        cv2.rectangle(court, (50, 50), (550, 350), (255, 255, 255), 2)
        
        # Center line and net
        cv2.line(court, (300, 50), (300, 350), (255, 255, 255), 3)
        cv2.line(court, (300, 45), (300, 355), (255, 0, 0), 5)  # Net in red
        
        # Attack lines
        cv2.line(court, (200, 50), (200, 350), (255, 255, 255), 1)
        cv2.line(court, (400, 50), (400, 350), (255, 255, 255), 1)
        
        return court
    
    def detect_volleyball(self, frame: np.ndarray) -> List[Detection]:
        """Detect volleyball in frame"""
        if isinstance(self.volleyball_detector, YOLO):
            results = self.volleyball_detector(frame)
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        detections.append(Detection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=float(conf),
                            class_id=0
                        ))
            return detections
        else:
            # RoboFlow API call would go here
            return []
    
    def detect_players(self, frame: np.ndarray) -> List[Detection]:
        """Detect players in frame"""
        if isinstance(self.player_detector, YOLO):
            results = self.player_detector(frame)
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = box.conf[0].cpu().numpy()
                        detections.append(Detection(
                            bbox=(int(x1), int(y1), int(x2), int(y2)),
                            confidence=float(conf),
                            class_id=1
                        ))
            return detections
        else:
            return []
    
    def detect_actions(self, frame: np.ndarray, player_bboxes: List[Detection]) -> Dict[int, str]:
        """Detect actions for each player"""
        actions = {}
        for i, player_det in enumerate(player_bboxes):
            x1, y1, x2, y2 = player_det.bbox
            player_crop = frame[y1:y2, x1:x2]
            
            if isinstance(self.action_detector, YOLO):
                results = self.action_detector(player_crop)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None and len(boxes) > 0:
                        cls_id = int(boxes[0].cls[0].cpu().numpy())
                        actions[i] = self.action_classes.get(cls_id, 'unknown')
                    else:
                        actions[i] = 'standing'
            else:
                actions[i] = 'unknown'
        
        return actions
    
    def detect_court(self, frame: np.ndarray) -> Optional[CourtCoordinates]:
        """Detect court boundaries and extract coordinates"""
        if isinstance(self.court_detector, YOLO):
            results = self.court_detector(frame)
            for result in results:
                masks = result.masks
                if masks is not None:
                    # Extract court polygon from segmentation mask
                    mask = masks.data[0].cpu().numpy()
                    mask = (mask * 255).astype(np.uint8)
                    
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        largest_contour = max(contours, key=cv2.contourArea)
                        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
                        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
                        
                        corners = [(int(point[0][0]), int(point[0][1])) for point in approx]
                        return CourtCoordinates(corners=corners)
        
        return None
    
    def update_tracker(self, detections: List[Detection], frame_id: int) -> List[Detection]:
        """Update tracker with new detections"""
        if self.tracker == "ultralytics_builtin" or self.tracker is None:
            # Tracking is handled in detect_players_with_tracking method
            return detections
        
        # Use standalone ByteTracker
        # Convert detections to ByteTrack format
        det_array = np.array([[d.bbox[0], d.bbox[1], d.bbox[2], d.bbox[3], d.confidence] 
                             for d in detections])
        
        if len(det_array) == 0:
            return detections
        
        # Update tracker
        tracks = self.tracker.update(det_array, frame_id)
        
        # Update detection objects with track IDs
        tracked_detections = []
        for track in tracks:
            x1, y1, x2, y2, track_id = track[:5]
            tracked_detections.append(Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)),
                confidence=track[4] if len(track) > 5 else 0.5,
                class_id=1,  # Assuming player class
                track_id=int(track_id)
            ))
        
        return tracked_detections
    
    def map_to_court_coordinates(self, bbox: Tuple[int, int, int, int], 
                                court_coords: CourtCoordinates, 
                                frame_shape: Tuple[int, int]) -> Tuple[float, float]:
        """Map bounding box center to court coordinates"""
        if not court_coords or len(court_coords.corners) < 4:
            return (0.0, 0.0)
        
        # Get center of bounding box
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        
        # Normalize to frame coordinates
        norm_x = center_x / frame_shape[1]
        norm_y = center_y / frame_shape[0]
        
        # Map to court template coordinates (simplified homography)
        court_x = norm_x * 500 + 50  # Map to court template width
        court_y = norm_y * 300 + 50  # Map to court template height
        
        return (court_x, court_y)
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> Dict:
        """Process a single frame and extract all information"""
        frame_data = {
            'frame_id': frame_id,
            'volleyball': [],
            'players': [],
            'actions': {},
            'court': None,
            'mapped_positions': {}
        }
        
        # Detect court first
        court_coords = self.detect_court(frame)
        frame_data['court'] = court_coords
        
        # Detect volleyball
        volleyball_detections = self.detect_volleyball(frame)
        frame_data['volleyball'] = volleyball_detections
        
        # Detect players with tracking
        if self.tracker == "ultralytics_builtin" or isinstance(self.tracker, str):
            # Use YOLOv8's built-in tracking
            tracked_players = self.detect_players_with_tracking(frame)
        else:
            # Use standalone ByteTracker
            player_detections = self.detect_players(frame)
            tracked_players = self.update_tracker(player_detections, frame_id)
        
        frame_data['players'] = tracked_players
        
        # Detect actions for tracked players
        actions = self.detect_actions(frame, tracked_players)
        frame_data['actions'] = actions
        
        # Map positions to court coordinates
        if court_coords:
            for i, player in enumerate(tracked_players):
                court_pos = self.map_to_court_coordinates(
                    player.bbox, court_coords, frame.shape[:2]
                )
                frame_data['mapped_positions'][player.track_id or i] = court_pos
        
        return frame_data
    
    def visualize_frame(self, frame: np.ndarray, frame_data: Dict) -> np.ndarray:
        """Visualize detections on frame"""
        vis_frame = frame.copy()
        
        # Draw volleyball detections
        for volleyball in frame_data['volleyball']:
            x1, y1, x2, y2 = volleyball.bbox
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(vis_frame, f'Ball {volleyball.confidence:.2f}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Draw player detections with track IDs and actions
        for i, player in enumerate(frame_data['players']):
            x1, y1, x2, y2 = player.bbox
            track_id = player.track_id or i
            action = frame_data['actions'].get(i, 'unknown')
            
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(vis_frame, f'Player {track_id}: {action}', 
                       (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw court boundaries if detected
        if frame_data['court'] and frame_data['court'].corners:
            corners = np.array(frame_data['court'].corners, np.int32)
            cv2.polylines(vis_frame, [corners], True, (0, 255, 0), 2)
        
        return vis_frame
    
    def create_court_visualization(self, frame_data: Dict) -> np.ndarray:
        """Create top-down court view with player positions"""
        court_vis = self.court_template.copy()
        
        # Plot mapped positions
        for track_id, (x, y) in frame_data['mapped_positions'].items():
            cv2.circle(court_vis, (int(x), int(y)), 8, (255, 0, 0), -1)
            cv2.putText(court_vis, str(track_id), (int(x)-5, int(y)+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return court_vis
    
    def process_video(self, video_path: str, output_dir: str = "output"):
        """Process entire video and save results"""
        cap = cv2.VideoCapture(video_path)
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Output video writers
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out_original = cv2.VideoWriter(str(output_path / 'analysis.mp4'), 
                                     fourcc, fps, (width, height))
        out_court = cv2.VideoWriter(str(output_path / 'court_view.mp4'), 
                                  fourcc, fps, (600, 400))
        
        frame_id = 0
        all_frame_data = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            frame_data = self.process_frame(frame, frame_id)
            all_frame_data.append(frame_data)
            
            # Create visualizations
            vis_frame = self.visualize_frame(frame, frame_data)
            court_vis = self.create_court_visualization(frame_data)
            
            # Write frames
            out_original.write(vis_frame)
            out_court.write(court_vis)
            
            frame_id += 1
            if frame_id % 30 == 0:  # Progress update every second
                print(f"Processed {frame_id} frames")
        
        # Cleanup
        cap.release()
        out_original.release()
        out_court.release()
        
        # Save analysis data
        with open(output_path / 'analysis_data.json', 'w') as f:
            # Convert non-serializable objects to serializable format
            serializable_data = []
            for frame_data in all_frame_data:
                serializable_frame = {
                    'frame_id': frame_data['frame_id'],
                    'volleyball': [{'bbox': v.bbox, 'confidence': v.confidence} 
                                 for v in frame_data['volleyball']],
                    'players': [{'bbox': p.bbox, 'track_id': p.track_id, 'confidence': p.confidence} 
                               for p in frame_data['players']],
                    'actions': frame_data['actions'],
                    'mapped_positions': frame_data['mapped_positions']
                }
                serializable_data.append(serializable_frame)
            
            json.dump(serializable_data, f, indent=2)
        
        print(f"Analysis complete! Results saved to {output_path}")
        return all_frame_data
    
    def generate_statistics(self, frame_data_list: List[Dict]) -> Dict:
        """Generate game statistics from processed frames"""
        stats = {
            'total_frames': len(frame_data_list),
            'player_actions': {action: 0 for action in self.action_classes.values()},
            'ball_detection_rate': 0,
            'player_trajectories': {},
            'action_heatmap': {}
        }
        
        ball_detections = 0
        
        for frame_data in frame_data_list:
            # Count ball detections
            if frame_data['volleyball']:
                ball_detections += 1
            
            # Count actions
            for action in frame_data['actions'].values():
                if action in stats['player_actions']:
                    stats['player_actions'][action] += 1
            
            # Track player trajectories
            for track_id, position in frame_data['mapped_positions'].items():
                if track_id not in stats['player_trajectories']:
                    stats['player_trajectories'][track_id] = []
                stats['player_trajectories'][track_id].append(position)
        
        stats['ball_detection_rate'] = ball_detections / len(frame_data_list) if frame_data_list else 0
        
        return stats

# Example usage and configuration
def main():
    """Example usage of the VolleyVision service"""
    
    # Model paths - replace with your actual model paths
    model_paths = {
        'volleyball': 'volleyball_yolov8.pt',  # Path to volleyball detection model
        'players': 'players_yolov8.pt',        # Path to player detection model
        'actions': 'actions_yolov8.pt',        # Path to action recognition model
        'court': 'court_yolov8_seg.pt'         # Path to court segmentation model
    }
    
    # Initialize service with different tracker options:
    # Option 1: Use YOLOv8's built-in ByteTrack (Recommended)
    service = VolleyVisionService(model_paths, use_roboflow=False, tracker_type="ultralytics")
    
    # Option 2: Use YOLOv8's built-in BoTSORT
    # service = VolleyVisionService(model_paths, use_roboflow=False, tracker_type="botsort")
    
    # Option 3: Try standalone ByteTracker (may require additional setup)
    # service = VolleyVisionService(model_paths, use_roboflow=False, tracker_type="bytetrack")
    
    # Process video
    video_path = "volleyball_match.mp4"  # Replace with your video path
    results = service.process_video(video_path)
    
    # Generate statistics
    stats = service.generate_statistics(results)
    print("Game Statistics:")
    print(f"Ball detection rate: {stats['ball_detection_rate']:.2%}")
    print(f"Action counts: {stats['player_actions']}")
    
    return service, results, stats

if __name__ == "__main__":
    service, results, stats = main()