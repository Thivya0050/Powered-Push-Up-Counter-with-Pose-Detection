"""
Pose Detector Module
Handles MediaPipe pose detection, angle calculation, and body alignment analysis
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, List, Dict
import time

from config import (
    POSE_DETECTION_CONFIG, 
    LandmarkIndices,
    ANGLE_SMOOTHING_WINDOW,
    UP_ANGLE_THRESHOLD,
    DOWN_ANGLE_THRESHOLD
)


class PoseDetector:
    """Handles all pose detection and analysis using MediaPipe"""
    
    def __init__(self):
        """Initialize MediaPipe Pose and tracking variables"""
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detector
        self.pose = self.mp_pose.Pose(
            model_complexity=POSE_DETECTION_CONFIG['model_complexity'],
            min_detection_confidence=POSE_DETECTION_CONFIG['min_detection_confidence'],
            min_tracking_confidence=POSE_DETECTION_CONFIG['min_tracking_confidence'],
            enable_segmentation=POSE_DETECTION_CONFIG['enable_segmentation'],
            smooth_landmarks=POSE_DETECTION_CONFIG['smooth_landmarks']
        )
        
        # Angle smoothing
        self.angle_history = []
        self.smoothing_window = ANGLE_SMOOTHING_WINDOW
        
        # Calibration variables
        self.calibrated = False
        self.tracking_side = 'auto'  # 'left', 'right', or 'auto'
        self.preferred_arm = None  # Will be set during calibration
        
        # Performance tracking
        self.last_landmarks = None
        self.detection_failures = 0
        
    def detect_pose(self, frame: np.ndarray) -> Tuple[Optional[object], np.ndarray]:
        """
        Detect pose in frame and return landmarks
        
        Args:
            frame: Input BGR image from camera
            
        Returns:
            Tuple of (landmarks object or None, processed RGB frame)
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        
        # Process frame
        results = self.pose.process(frame_rgb)
        
        frame_rgb.flags.writeable = True
        
        # Track detection success
        if results.pose_landmarks:
            self.last_landmarks = results.pose_landmarks
            self.detection_failures = 0
        else:
            self.detection_failures += 1
            
        return results.pose_landmarks, frame_rgb
    
    def calculate_angle(self, a: tuple, b: tuple, c: tuple, smooth: bool = True) -> float:
        """
        Calculate angle at point b formed by points a, b, c
        
        Args:
            a: First point (x, y)
            b: Middle point (vertex) (x, y)
            c: Third point (x, y)
            smooth: Whether to apply smoothing
            
        Returns:
            Angle in degrees (0-180)
        """
        # Convert to numpy arrays
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle using dot product
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        
        # Clamp to valid range to avoid numerical errors
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.arccos(cosine_angle)
        angle_degrees = np.degrees(angle)
        
        # Apply smoothing if enabled
        if smooth:
            self.angle_history.append(angle_degrees)
            if len(self.angle_history) > self.smoothing_window:
                self.angle_history.pop(0)
            angle_degrees = np.mean(self.angle_history)
        
        return angle_degrees
    
    def get_landmark_coords(self, landmarks, landmark_idx: int, 
                           frame_width: int, frame_height: int) -> Tuple[int, int, float]:
        """
        Get pixel coordinates and visibility for a specific landmark
        
        Args:
            landmarks: MediaPipe landmarks object
            landmark_idx: Index of the landmark
            frame_width: Width of the frame
            frame_height: Height of the frame
            
        Returns:
            Tuple of (x, y, visibility)
        """
        landmark = landmarks.landmark[landmark_idx]
        x = int(landmark.x * frame_width)
        y = int(landmark.y * frame_height)
        visibility = landmark.visibility
        
        return x, y, visibility
    
    def get_elbow_angle(self, landmarks, frame_width: int, frame_height: int, 
                       side: str = 'auto') -> Tuple[float, str]:
        """
        Calculate elbow angle for push-up tracking
        
        Args:
            landmarks: MediaPipe landmarks object
            frame_width: Width of the frame
            frame_height: Height of the frame
            side: 'left', 'right', or 'auto'
            
        Returns:
            Tuple of (angle, side_used)
        """
        if side == 'auto':
            side = self._determine_tracking_side(landmarks)
        
        # Get appropriate landmarks based on side
        if side == 'left':
            shoulder_idx = LandmarkIndices.LEFT_SHOULDER
            elbow_idx = LandmarkIndices.LEFT_ELBOW
            wrist_idx = LandmarkIndices.LEFT_WRIST
        else:  # right
            shoulder_idx = LandmarkIndices.RIGHT_SHOULDER
            elbow_idx = LandmarkIndices.RIGHT_ELBOW
            wrist_idx = LandmarkIndices.RIGHT_WRIST
        
        # Get coordinates
        shoulder = landmarks.landmark[shoulder_idx]
        elbow = landmarks.landmark[elbow_idx]
        wrist = landmarks.landmark[wrist_idx]
        
        # Calculate angle
        angle = self.calculate_angle(
            (shoulder.x, shoulder.y),
            (elbow.x, elbow.y),
            (wrist.x, wrist.y)
        )
        
        return angle, side
    
    def _determine_tracking_side(self, landmarks) -> str:
        """
        Automatically determine which arm to track based on visibility
        
        Args:
            landmarks: MediaPipe landmarks object
            
        Returns:
            'left' or 'right'
        """
        if self.preferred_arm:
            return self.preferred_arm
        
        # Check visibility of both arms
        left_elbow_vis = landmarks.landmark[LandmarkIndices.LEFT_ELBOW].visibility
        right_elbow_vis = landmarks.landmark[LandmarkIndices.RIGHT_ELBOW].visibility
        
        left_wrist_vis = landmarks.landmark[LandmarkIndices.LEFT_WRIST].visibility
        right_wrist_vis = landmarks.landmark[LandmarkIndices.RIGHT_WRIST].visibility
        
        # Calculate average visibility for each side
        left_avg = (left_elbow_vis + left_wrist_vis) / 2
        right_avg = (right_elbow_vis + right_wrist_vis) / 2
        
        # Choose side with better visibility
        preferred = 'left' if left_avg > right_avg else 'right'
        
        # Cache the preference for consistency
        self.preferred_arm = preferred
        
        return preferred
    
    def get_body_alignment(self, landmarks) -> Tuple[float, Dict[str, any]]:
        """
        Calculate body alignment score (straight line from shoulders to ankles)
        
        Args:
            landmarks: MediaPipe landmarks object
            
        Returns:
            Tuple of (alignment_score 0-100, alignment_details dict)
        """
        # Get key points
        left_shoulder = landmarks.landmark[LandmarkIndices.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[LandmarkIndices.RIGHT_SHOULDER]
        left_hip = landmarks.landmark[LandmarkIndices.LEFT_HIP]
        right_hip = landmarks.landmark[LandmarkIndices.RIGHT_HIP]
        left_ankle = landmarks.landmark[LandmarkIndices.LEFT_ANKLE]
        right_ankle = landmarks.landmark[LandmarkIndices.RIGHT_ANKLE]
        
        # Calculate midpoints
        shoulder_mid = ((left_shoulder.x + right_shoulder.x) / 2,
                       (left_shoulder.y + right_shoulder.y) / 2)
        hip_mid = ((left_hip.x + right_hip.x) / 2,
                  (left_hip.y + right_hip.y) / 2)
        ankle_mid = ((left_ankle.x + right_ankle.x) / 2,
                    (left_ankle.y + right_ankle.y) / 2)
        
        # Calculate expected hip position if body were perfectly straight
        # Using linear interpolation
        t = 0.5  # Hip is roughly midway between shoulder and ankle
        expected_hip_x = shoulder_mid[0] + t * (ankle_mid[0] - shoulder_mid[0])
        expected_hip_y = shoulder_mid[1] + t * (ankle_mid[1] - shoulder_mid[1])
        
        # Calculate deviation
        deviation_x = abs(hip_mid[0] - expected_hip_x)
        deviation_y = abs(hip_mid[1] - expected_hip_y)
        
        # Total deviation (normalized)
        total_deviation = np.sqrt(deviation_x**2 + deviation_y**2)
        
        # Convert to score (0-100)
        # A deviation of 0.1 (10% of frame) should be considered poor
        score = max(0, 100 - (total_deviation * 1000))
        
        # Determine alignment issue
        issue = None
        if deviation_y > 0.05:
            if hip_mid[1] < expected_hip_y:
                issue = "hips_too_high"
            else:
                issue = "hips_sagging"
        
        details = {
            'score': score,
            'deviation': total_deviation,
            'issue': issue,
            'hip_position': hip_mid,
            'expected_hip_position': (expected_hip_x, expected_hip_y)
        }
        
        return score, details
    
    def detect_side_view(self, landmarks) -> Tuple[bool, float]:
        """
        Detect if user is in optimal side-view position
        
        Args:
            landmarks: MediaPipe landmarks object
            
        Returns:
            Tuple of (is_side_view bool, confidence score 0-1)
        """
        # Check shoulder visibility
        left_shoulder = landmarks.landmark[LandmarkIndices.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[LandmarkIndices.RIGHT_SHOULDER]
        
        # In side view, one shoulder should be more visible than the other
        visibility_diff = abs(left_shoulder.visibility - right_shoulder.visibility)
        
        # Check if shoulders are aligned horizontally (not side view)
        # In side view, shoulders should be at similar Y coordinates
        y_diff = abs(left_shoulder.y - right_shoulder.y)
        
        # Calculate confidence
        # Good side view: high visibility difference, low Y difference
        confidence = visibility_diff * (1 - y_diff * 2)
        confidence = np.clip(confidence, 0, 1)
        
        is_side_view = confidence > 0.5
        
        return is_side_view, confidence
    
    def calibrate(self, landmarks_list: List, duration: float = 3.0) -> Dict:
        """
        Auto-calibrate tracking based on user's range of motion
        
        Args:
            landmarks_list: List of landmark objects collected over time
            duration: Calibration duration in seconds
            
        Returns:
            Calibration results dictionary
        """
        if not landmarks_list:
            return {'success': False, 'message': 'No landmarks provided'}
        
        angles = []
        alignments = []
        
        # Analyze collected data
        for landmarks in landmarks_list:
            angle, side = self.get_elbow_angle(landmarks, 640, 480)
            alignment, _ = self.get_body_alignment(landmarks)
            
            angles.append(angle)
            alignments.append(alignment)
        
        # Calculate statistics
        min_angle = min(angles)
        max_angle = max(angles)
        avg_alignment = np.mean(alignments)
        
        # Set personalized thresholds (with safety margins)
        up_threshold = max_angle * 0.95  # 95% of max extension
        down_threshold = min_angle * 1.1  # 110% of min angle (more lenient)
        
        # Ensure thresholds are reasonable
        up_threshold = max(150, min(up_threshold, 175))
        down_threshold = max(70, min(down_threshold, 100))
        
        self.calibrated = True
        
        results = {
            'success': True,
            'preferred_arm': self.preferred_arm or 'right',
            'angle_range': (min_angle, max_angle),
            'suggested_up_threshold': up_threshold,
            'suggested_down_threshold': down_threshold,
            'average_alignment': avg_alignment,
            'message': f'Calibration complete! Tracking {self.preferred_arm} arm.'
        }
        
        return results
    
    def draw_skeleton(self, frame: np.ndarray, landmarks, 
                     color_good: tuple = (0, 255, 0),
                     color_bad: tuple = (0, 0, 255),
                     current_angle: float = None) -> np.ndarray:
        """
        Draw pose skeleton on frame with color coding based on form
        
        Args:
            frame: Frame to draw on
            landmarks: MediaPipe landmarks object
            color_good: Color for good form
            color_bad: Color for bad form
            current_angle: Current elbow angle (for color coding)
            
        Returns:
            Frame with skeleton drawn
        """
        if landmarks is None:
            return frame
        
        # Determine color based on angle
        if current_angle is not None:
            if UP_ANGLE_THRESHOLD - 10 <= current_angle <= 180 or \
               DOWN_ANGLE_THRESHOLD <= current_angle <= DOWN_ANGLE_THRESHOLD + 20:
                color = color_good
            else:
                color = color_bad
        else:
            color = color_good
        
        # Draw landmarks
        self.mp_drawing.draw_landmarks(
            frame,
            landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                color=color, thickness=2, circle_radius=3
            ),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(
                color=color, thickness=2, circle_radius=2
            )
        )
        
        return frame
    
    def draw_angle_arc(self, frame: np.ndarray, landmarks, 
                      frame_width: int, frame_height: int,
                      side: str = 'right') -> np.ndarray:
        """
        Draw angle arc at elbow joint
        
        Args:
            frame: Frame to draw on
            landmarks: MediaPipe landmarks object
            frame_width: Frame width
            frame_height: Frame height
            side: Which arm to draw ('left' or 'right')
            
        Returns:
            Frame with angle arc drawn
        """
        if side == 'left':
            shoulder_idx = LandmarkIndices.LEFT_SHOULDER
            elbow_idx = LandmarkIndices.LEFT_ELBOW
            wrist_idx = LandmarkIndices.LEFT_WRIST
        else:
            shoulder_idx = LandmarkIndices.RIGHT_SHOULDER
            elbow_idx = LandmarkIndices.RIGHT_ELBOW
            wrist_idx = LandmarkIndices.RIGHT_WRIST
        
        # Get coordinates
        shoulder_x, shoulder_y, _ = self.get_landmark_coords(
            landmarks, shoulder_idx, frame_width, frame_height
        )
        elbow_x, elbow_y, _ = self.get_landmark_coords(
            landmarks, elbow_idx, frame_width, frame_height
        )
        wrist_x, wrist_y, _ = self.get_landmark_coords(
            landmarks, wrist_idx, frame_width, frame_height
        )
        
        # Calculate angle
        angle = self.calculate_angle(
            (shoulder_x, shoulder_y),
            (elbow_x, elbow_y),
            (wrist_x, wrist_y),
            smooth=False
        )
        
        # Draw angle arc
        radius = 40
        cv2.ellipse(frame, (elbow_x, elbow_y), (radius, radius), 
                   0, 0, int(angle), (255, 255, 0), 2)
        
        # Draw angle text
        text_x = elbow_x + radius + 10
        text_y = elbow_y
        cv2.putText(frame, f"{int(angle)}°", (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def is_pose_visible(self, landmarks, min_visibility: float = 0.5) -> bool:
        """
        Check if key landmarks are sufficiently visible
        
        Args:
            landmarks: MediaPipe landmarks object
            min_visibility: Minimum visibility threshold
            
        Returns:
            True if pose is sufficiently visible
        """
        if landmarks is None:
            return False
        
        # Check key landmarks
        key_landmarks = [
            LandmarkIndices.LEFT_SHOULDER,
            LandmarkIndices.RIGHT_SHOULDER,
            LandmarkIndices.LEFT_ELBOW,
            LandmarkIndices.RIGHT_ELBOW,
            LandmarkIndices.LEFT_HIP,
            LandmarkIndices.RIGHT_HIP,
        ]
        
        visible_count = 0
        for idx in key_landmarks:
            if landmarks.landmark[idx].visibility >= min_visibility:
                visible_count += 1
        
        # At least 4 out of 6 key landmarks should be visible
        return visible_count >= 4
    
    def reset_smoothing(self):
        """Reset angle smoothing history"""
        self.angle_history = []
    
    def __del__(self):
        """Cleanup resources"""
        if hasattr(self, 'pose'):
            self.pose.close()





