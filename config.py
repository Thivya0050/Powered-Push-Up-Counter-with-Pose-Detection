"""
Configuration Module for AI-Powered Push-Up Counter
Contains all configurable parameters and constants
"""

# ==================== POSE DETECTION SETTINGS ====================
POSE_DETECTION_CONFIG = {
    'model_complexity': 1,  # 0, 1, or 2 (higher = more accurate but slower)
    'min_detection_confidence': 0.7,  # 0.5-0.9
    'min_tracking_confidence': 0.7,  # 0.5-0.9
    'enable_segmentation': False,
    'smooth_landmarks': True,
    'min_visibility_threshold': 0.5  # Minimum visibility for landmarks
}

# ==================== PUSH-UP PARAMETERS ====================
# Angle thresholds (in degrees)
UP_ANGLE_THRESHOLD = 160  # Extended arms position
DOWN_ANGLE_THRESHOLD = 90  # Lowered position

# Timing parameters (in seconds)
MIN_TIME_BETWEEN_REPS = 0.5  # Prevent false counts
HOLD_TIME_REQUIREMENT = 0.3  # Time at bottom position
MIN_REP_DURATION = 1.0  # Minimum time for a complete rep
MAX_REP_DURATION = 5.0  # Maximum time before considering rep abandoned

# Angle smoothing
ANGLE_SMOOTHING_WINDOW = 5  # Number of frames to average

# ==================== FORM SCORING WEIGHTS ====================
FORM_WEIGHTS = {
    'body_alignment': 0.30,  # 30%
    'elbow_consistency': 0.25,  # 25%
    'depth_consistency': 0.25,  # 25%
    'speed_control': 0.20  # 20%
}

# Form quality thresholds
FORM_QUALITY_THRESHOLDS = {
    'perfect': 90,  # 90-100
    'good': 75,  # 75-89
    'fair': 60,  # 60-74
    'poor': 0  # 0-59
}

# Body alignment tolerance (deviation from straight line)
MAX_BODY_ALIGNMENT_DEVIATION = 0.05  # 5% of body length

# Elbow angle from body (for flared elbow detection)
MAX_ELBOW_ANGLE_FROM_BODY = 45  # degrees

# ==================== AUDIO SETTINGS ====================
AUDIO_CONFIG = {
    'voice_feedback_enabled': True,
    'count_announcement_interval': 5,  # Announce every N reps
    'milestone_announcements': [10, 25, 50, 75, 100],
    'form_alert_volume': 80,  # 0-100
    'tts_rate': 175,  # Words per minute
    'tts_volume': 0.9,  # 0.0-1.0
    'alert_cooldown': 5.0,  # Seconds between same alert
}

# Motivational phrases
MOTIVATIONAL_PHRASES = {
    'milestone': ["Great job!", "Excellent work!", "You're crushing it!", "Keep it up!"],
    'perfect_form': ["Perfect form!", "Flawless!", "That's how it's done!"],
    'form_warning': ["Keep your hips aligned", "Go deeper", "Slow down", "Elbows in", "Straight back"],
    'workout_complete': ["Workout complete!", "Amazing job!", "Well done!"],
    'new_record': ["New personal record!", "You've outdone yourself!", "Record broken!"]
}

# ==================== UI SETTINGS ====================
UI_CONFIG = {
    'window_name': 'AI Push-Up Counter',
    'window_width': 1280,
    'window_height': 720,
    'fps_display': True,
    'show_skeleton': True,
    'show_angles': True,
    'show_form_score': True,
    'show_realtime_graphs': True,
    'graph_update_interval': 10,  # Update every N frames
}

# Color scheme (BGR format for OpenCV)
COLORS = {
    'good': (0, 255, 0),  # Green
    'warning': (0, 255, 255),  # Yellow
    'error': (0, 0, 255),  # Red
    'text': (255, 255, 255),  # White
    'background': (40, 40, 40),  # Dark gray
    'panel_bg': (20, 20, 20, 180),  # Semi-transparent dark
    'skeleton_good': (0, 255, 0),  # Green
    'skeleton_bad': (0, 0, 255),  # Red
    'ui_primary': (255, 200, 100),  # Light blue
    'ui_secondary': (200, 200, 200),  # Light gray
}

# Font settings
FONTS = {
    'main': 0,  # cv2.FONT_HERSHEY_SIMPLEX
    'bold': 2,  # cv2.FONT_HERSHEY_SIMPLEX
    'size_large': 2.0,
    'size_medium': 1.0,
    'size_small': 0.6,
    'thickness_large': 3,
    'thickness_medium': 2,
    'thickness_small': 1,
}

# ==================== MEDIAPIPE LANDMARK INDICES ====================
class LandmarkIndices:
    """MediaPipe Pose Landmark Indices"""
    # Upper body
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    
    # Lower body
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

# ==================== WORKOUT MODES ====================
WORKOUT_MODES = {
    'FREE': {
        'name': 'Free Mode',
        'description': 'Unlimited counting',
        'has_goal': False,
        'has_timer': False,
    },
    'GOAL': {
        'name': 'Goal Mode',
        'description': 'Set target reps',
        'has_goal': True,
        'has_timer': False,
        'default_goal': 20,
    },
    'TIMED': {
        'name': 'Timed Mode',
        'description': 'AMRAP - As Many Reps As Possible',
        'has_goal': False,
        'has_timer': True,
        'default_duration': 60,  # seconds
    },
    'INTERVAL': {
        'name': 'Interval Mode',
        'description': '20 sec work, 10 sec rest',
        'has_goal': False,
        'has_timer': True,
        'work_duration': 20,
        'rest_duration': 10,
        'rounds': 5,
    },
    'CHALLENGE': {
        'name': 'Challenge Mode',
        'description': 'Maintain perfect form',
        'has_goal': True,
        'has_timer': False,
        'min_form_score': 85,
        'default_goal': 10,
    }
}

# ==================== CALORIE CALCULATION ====================
# Calories per push-up based on body weight (rough estimates)
CALORIES_PER_PUSHUP_BASE = 0.32  # Average
CALORIES_PER_PUSHUP_MODIFIER = {
    'slow': 1.2,  # Slower reps burn more
    'normal': 1.0,
    'fast': 0.8,
}

# ==================== DATABASE SETTINGS ====================
DATABASE_CONFIG = {
    'db_path': 'data/sessions.db',
    'backup_enabled': True,
    'backup_interval_days': 7,
    'max_history_days': 365,  # Keep 1 year of history
}

# ==================== ANALYTICS SETTINGS ====================
ANALYTICS_CONFIG = {
    'show_realtime_graphs': True,
    'graph_history_points': 100,  # Number of points to show in live graphs
    'export_format': 'csv',  # 'csv' or 'json'
}

# ==================== ACHIEVEMENT THRESHOLDS ====================
ACHIEVEMENTS = {
    'first_pushup': {'name': 'First Push-Up', 'requirement': 1},
    'beginner': {'name': 'Getting Started', 'requirement': 10},
    'intermediate': {'name': 'Building Strength', 'requirement': 50},
    'advanced': {'name': 'Strong!', 'requirement': 100},
    'elite': {'name': 'Elite Athlete', 'requirement': 250},
    'master': {'name': 'Push-Up Master', 'requirement': 500},
    'legend': {'name': 'Legendary', 'requirement': 1000},
    'perfect_10': {'name': 'Perfect 10', 'description': '10 consecutive perfect form reps'},
    'century': {'name': 'Century Club', 'description': '100 reps in one session'},
    'consistency': {'name': 'Consistent', 'description': '7 days streak'},
}

# ==================== CAMERA SETTINGS ====================
CAMERA_CONFIG = {
    'camera_index': 0,  # Default webcam
    'frame_width': 1280,
    'frame_height': 720,
    'fps': 30,
    'processing_width': 640,  # Resize for faster processing
    'processing_height': 480,
}

# ==================== PERFORMANCE SETTINGS ====================
PERFORMANCE_CONFIG = {
    'process_every_n_frames': 1,  # Process every frame (1) or skip frames (2, 3, etc.)
    'enable_gpu': False,  # Use GPU acceleration if available
    'max_fps': 30,
    'reduce_quality_on_low_fps': True,
    'low_fps_threshold': 15,
}

# ==================== STATE MACHINE STATES ====================
class WorkoutState:
    """Workout state machine states"""
    IDLE = 'idle'
    CALIBRATING = 'calibrating'
    READY = 'ready'
    COUNTDOWN = 'countdown'
    ACTIVE = 'active'
    PAUSED = 'paused'
    RESTING = 'resting'
    COMPLETED = 'completed'

class RepState:
    """Rep counting state machine states"""
    IDLE = 'idle'
    UP = 'up'
    TRANSITIONING_DOWN = 'transitioning_down'
    DOWN = 'down'
    TRANSITIONING_UP = 'transitioning_up'
    COMPLETED = 'completed'

# ==================== ERROR MESSAGES ====================
ERROR_MESSAGES = {
    'no_camera': 'Error: Unable to access camera. Please check your webcam connection.',
    'no_pose_detected': 'No pose detected. Please ensure your full body is visible.',
    'partial_visibility': 'Partial body visibility. Adjust camera angle for better tracking.',
    'database_error': 'Database error occurred. Your workout data may not be saved.',
    'audio_error': 'Audio system unavailable. Continuing without voice feedback.',
}

# ==================== FEEDBACK VERBOSITY LEVELS ====================
class FeedbackLevel:
    SILENT = 0  # No audio feedback
    MINIMAL = 1  # Only count milestones
    MODERATE = 2  # Counts + important alerts
    VERBOSE = 3  # All feedback including form corrections

DEFAULT_FEEDBACK_LEVEL = FeedbackLevel.MODERATE

# ==================== EXPORT SETTINGS ====================
EXPORT_CONFIG = {
    'include_graphs': True,
    'graph_dpi': 150,
    'report_format': 'html',  # 'html', 'pdf', 'txt'
}



