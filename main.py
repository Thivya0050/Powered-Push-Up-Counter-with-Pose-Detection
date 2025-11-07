"""
Main Application - AI-Powered Push-Up Counter with Pose Detection
Advanced real-time push-up counting with form analysis and comprehensive feedback
"""

import cv2
import numpy as np
import time
from datetime import datetime
from typing import Optional, Tuple, Dict
import sys

# Import custom modules
from config import *
from pose_detector import PoseDetector
from form_analyzer import FormAnalyzer
from audio_feedback import get_audio_feedback, FeedbackLevel
from analytics_dashboard import AnalyticsDashboard
from database import WorkoutDatabase


class PushUpCounter:
    """Main application class for push-up counting"""
    
    def __init__(self, workout_mode: str = 'FREE', goal: int = 0):
        """
        Initialize push-up counter
        
        Args:
            workout_mode: Workout mode ('FREE', 'GOAL', 'TIMED', etc.)
            goal: Target reps (for GOAL mode)
        """
        # Initialize components
        print("Initializing AI Push-Up Counter...")
        
        self.pose_detector = PoseDetector()
        self.form_analyzer = FormAnalyzer()
        self.audio = get_audio_feedback(enabled=True, verbosity=FeedbackLevel.MODERATE)
        self.dashboard = AnalyticsDashboard()
        self.database = WorkoutDatabase()
        
        # Workout settings
        self.workout_mode = workout_mode
        self.goal = goal
        
        # State machine
        self.workout_state = WorkoutState.IDLE
        self.rep_state = RepState.IDLE
        
        # Counting variables
        self.rep_count = 0
        self.last_count_time = 0
        self.current_angle = 0
        self.current_form_score = 0
        
        # Rep tracking
        self.in_down_position = False
        self.down_position_time = None
        self.hold_satisfied = False
        
        # Camera
        self.cap = None
        self.frame_width = CAMERA_CONFIG['frame_width']
        self.frame_height = CAMERA_CONFIG['frame_height']
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = time.time()
        
        # UI state
        self.show_graphs = UI_CONFIG['show_realtime_graphs']
        self.show_skeleton = UI_CONFIG['show_skeleton']
        self.paused = False
        
        # Workout data
        self.workout_start_time = None
        self.workout_data = []
        
        print("Initialization complete!")
    
    def initialize_camera(self) -> bool:
        """
        Initialize camera capture
        
        Returns:
            True if successful
        """
        print("Initializing camera...")
        
        try:
            self.cap = cv2.VideoCapture(CAMERA_CONFIG['camera_index'])
            
            if not self.cap.isOpened():
                print(ERROR_MESSAGES['no_camera'])
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, CAMERA_CONFIG['fps'])
            
            # Read actual dimensions
            self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            print(f"Camera initialized: {self.frame_width}x{self.frame_height}")
            return True
            
        except Exception as e:
            print(f"Camera initialization error: {e}")
            return False
    
    def start_workout(self):
        """Start the workout session"""
        self.workout_state = WorkoutState.ACTIVE
        self.workout_start_time = time.time()
        self.dashboard.start_workout()
        self.form_analyzer.reset()
        self.rep_count = 0
        
        # Audio feedback
        self.audio.announce_workout_start()
        
        print("Workout started!")
    
    def update_rep_counting(self, angle: float, timestamp: float) -> bool:
        """
        Update rep counting state machine
        
        Args:
            angle: Current elbow angle
            timestamp: Current timestamp
            
        Returns:
            True if a new rep was counted
        """
        new_rep = False
        
        # State: UP (arms extended)
        if self.rep_state == RepState.IDLE or self.rep_state == RepState.UP:
            if angle >= UP_ANGLE_THRESHOLD:
                self.rep_state = RepState.UP
                self.in_down_position = False
                self.hold_satisfied = False
            elif angle < DOWN_ANGLE_THRESHOLD + 20:
                self.rep_state = RepState.TRANSITIONING_DOWN
                self.down_position_time = timestamp
                self.form_analyzer.update_rep_phase('descending', timestamp)
        
        # State: TRANSITIONING DOWN
        elif self.rep_state == RepState.TRANSITIONING_DOWN:
            if angle <= DOWN_ANGLE_THRESHOLD:
                # Check hold time
                if self.down_position_time:
                    hold_duration = timestamp - self.down_position_time
                    if hold_duration >= HOLD_TIME_REQUIREMENT:
                        self.rep_state = RepState.DOWN
                        self.in_down_position = True
                        self.hold_satisfied = True
                        self.form_analyzer.update_rep_phase('down', timestamp)
            elif angle >= UP_ANGLE_THRESHOLD - 10:
                # Returned to up without proper depth
                self.rep_state = RepState.UP
                self.down_position_time = None
        
        # State: DOWN (bottom position)
        elif self.rep_state == RepState.DOWN:
            if angle > DOWN_ANGLE_THRESHOLD + 10:
                self.rep_state = RepState.TRANSITIONING_UP
                self.form_analyzer.update_rep_phase('ascending', timestamp)
        
        # State: TRANSITIONING UP
        elif self.rep_state == RepState.TRANSITIONING_UP:
            if angle >= UP_ANGLE_THRESHOLD:
                # Check if can count rep
                time_since_last = timestamp - self.last_count_time
                
                if self.hold_satisfied and time_since_last >= MIN_TIME_BETWEEN_REPS:
                    # Count the rep!
                    self.rep_count += 1
                    self.last_count_time = timestamp
                    new_rep = True
                    
                    # Record rep data
                    rep_data = self.form_analyzer.end_rep(timestamp, self.rep_count)
                    self.dashboard.record_rep(rep_data)
                    self.workout_data.append(rep_data)
                    
                    # Audio feedback
                    self.audio.announce_count(self.rep_count)
                    
                    # Check for perfect form
                    if rep_data.get('quality_grade') == 'Perfect':
                        if self.dashboard.perfect_streak == 1:  # Just started streak
                            self.audio.announce_perfect_form()
                    
                    # Check for goal
                    if self.workout_mode == 'GOAL' and self.rep_count >= self.goal:
                        self.audio.announce_goal_reached(self.goal)
                        self.complete_workout()
                    
                    # Check streak milestones
                    self.audio.announce_streak(self.dashboard.perfect_streak)
                    
                    # Start tracking next rep
                    self.form_analyzer.start_rep(timestamp)
                
                # Reset state
                self.rep_state = RepState.UP
                self.in_down_position = False
                self.hold_satisfied = False
                self.down_position_time = None
                self.form_analyzer.update_rep_phase('up', timestamp)
        
        return new_rep
    
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Process a single frame
        
        Args:
            frame: Input frame from camera
            
        Returns:
            Tuple of (processed frame, pose_detected)
        """
        timestamp = time.time()
        
        # Detect pose
        landmarks, frame_rgb = self.pose_detector.detect_pose(frame)
        
        if landmarks is None or not self.pose_detector.is_pose_visible(landmarks):
            # No pose detected
            cv2.putText(frame, ERROR_MESSAGES['no_pose_detected'],
                       (50, self.frame_height // 2),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['error'], 2)
            return frame, False
        
        # Get elbow angle
        angle, tracking_side = self.pose_detector.get_elbow_angle(
            landmarks, self.frame_width, self.frame_height
        )
        self.current_angle = angle
        
        # Get body alignment
        alignment_score, alignment_details = self.pose_detector.get_body_alignment(landmarks)
        
        # Calculate form score
        form_score, form_components = self.form_analyzer.calculate_form_score(
            landmarks, angle, alignment_score
        )
        self.current_form_score = form_score
        
        # Detect mistakes
        mistakes = self.form_analyzer.detect_mistakes(
            landmarks, angle, alignment_details
        )
        
        # Announce form corrections (if enabled)
        if mistakes and self.audio.verbosity >= FeedbackLevel.VERBOSE:
            for mistake in mistakes[:1]:  # Only announce top mistake
                self.audio.announce_form_correction(
                    mistake['type'], mistake['message']
                )
        
        # Update current rep metrics
        if self.rep_state != RepState.IDLE and self.workout_state == WorkoutState.ACTIVE:
            self.form_analyzer.update_current_rep_metrics(angle, form_score, mistakes)
        
        # Update rep counting
        if self.workout_state == WorkoutState.ACTIVE and not self.paused:
            new_rep = self.update_rep_counting(angle, timestamp)
            
            # Check for fatigue
            is_fatigued, fatigue_pct = self.form_analyzer.detect_fatigue()
            if is_fatigued and self.rep_count > 10:
                self.audio.announce_fatigue_detected()
        
        # Update dashboard data
        self.dashboard.update_realtime_data(angle, form_score, timestamp)
        
        # Draw visualizations
        if self.show_skeleton:
            frame = self.pose_detector.draw_skeleton(
                frame, landmarks, COLORS['skeleton_good'],
                COLORS['skeleton_bad'], angle
            )
        
        # Draw angle arc
        if UI_CONFIG['show_angles']:
            frame = self.pose_detector.draw_angle_arc(
                frame, landmarks, self.frame_width,
                self.frame_height, tracking_side
            )
        
        # Draw main metrics
        frame = self.dashboard.draw_main_metrics(
            frame, angle, form_score, mistakes
        )
        
        # Draw quality distribution chart (if we have reps)
        if self.rep_count > 0:
            frame = self.dashboard.draw_quality_distribution(
                frame, 20, self.frame_height - 200, 300, 150
            )
        
        # Draw live graphs if enabled
        if self.show_graphs:
            frame = self.dashboard.draw_live_graphs(frame)
        
        # Draw FPS
        if UI_CONFIG['fps_display']:
            frame = self.dashboard.draw_fps(frame, self.fps)
        
        # Draw control help
        frame = self.dashboard.draw_controls_help(frame)
        
        # Draw state indicators
        if self.paused:
                    cv2.putText(frame, "PAUSED", (self.frame_width // 2 - 100, 100),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS['warning'], 4)
        
        return frame, True
    
    def complete_workout(self):
        """Complete the workout and save data"""
        if self.workout_state != WorkoutState.ACTIVE:
            return
        
        self.workout_state = WorkoutState.COMPLETED
        
        # Calculate workout duration
        duration = time.time() - self.workout_start_time
        
        # Get workout summary
        summary = self.form_analyzer.get_workout_summary()
        summary['duration'] = duration
        summary['date'] = datetime.now()
        summary['workout_mode'] = self.workout_mode
        
        # Add dashboard metrics
        summary['total_reps'] = self.rep_count
        summary['total_calories'] = self.dashboard.total_calories
        summary['max_streak'] = self.dashboard.max_streak
        
        # Save to database
        try:
            workout_id = self.database.save_workout_with_reps(summary, self.workout_data)
            print(f"Workout saved! ID: {workout_id}")
            
            # Check for achievements
            self._check_achievements(summary)
            
        except Exception as e:
            print(f"Error saving workout: {e}")
        
        # Audio announcement
        self.audio.announce_workout_complete(
            self.rep_count,
            summary.get('average_form_score', 0)
        )
        
        return summary
    
    def _check_achievements(self, summary: Dict):
        """Check and award achievements"""
        profile = self.database.get_user_profile()
        
        # Check milestone achievements
        lifetime_total = profile['total_pushups_lifetime']
        
        for achievement_id, achievement in ACHIEVEMENTS.items():
            if 'requirement' in achievement:
                if lifetime_total >= achievement['requirement']:
                    earned = self.database.add_achievement(
                        achievement_id, achievement['name']
                    )
                    if earned:
                        self.audio.announce_achievement(achievement['name'])
        
        # Check special achievements
        if self.dashboard.max_streak >= 10:
            self.database.add_achievement('perfect_10', 'Perfect 10')
        
        if self.rep_count >= 100:
            self.database.add_achievement('century', 'Century Club')
    
    def handle_key_press(self, key: int) -> bool:
        """
        Handle keyboard input
        
        Args:
            key: Key code
            
        Returns:
            True to continue, False to exit
        """
        # Q - Quit
        if key == ord('q') or key == ord('Q'):
            if self.workout_state == WorkoutState.ACTIVE:
                self.complete_workout()
            return False
        
        # P - Pause/Resume
        elif key == ord('p') or key == ord('P'):
            if self.workout_state == WorkoutState.ACTIVE:
                self.paused = not self.paused
                print("Paused" if self.paused else "Resumed")
        
        # R - Reset/Restart
        elif key == ord('r') or key == ord('R'):
            if self.workout_state == WorkoutState.COMPLETED:
                self.reset_workout()
            elif self.workout_state == WorkoutState.ACTIVE:
                # Confirm reset
                print("Press 'R' again to confirm reset")
        
        # S - Start (if idle)
        elif key == ord('s') or key == ord('S'):
            if self.workout_state == WorkoutState.IDLE:
                self.start_workout()
        
        # V - Toggle voice
        elif key == ord('v') or key == ord('V'):
            self.audio.toggle()
            print(f"Voice feedback: {'ON' if self.audio.enabled else 'OFF'}")
        
        # G - Toggle graphs
        elif key == ord('g') or key == ord('G'):
            self.show_graphs = not self.show_graphs
            print(f"Graphs: {'ON' if self.show_graphs else 'OFF'}")
        
        # K - Toggle skeleton
        elif key == ord('k') or key == ord('K'):
            self.show_skeleton = not self.show_skeleton
            print(f"Skeleton: {'ON' if self.show_skeleton else 'OFF'}")
        
        return True
    
    def reset_workout(self):
        """Reset for a new workout"""
        self.workout_state = WorkoutState.IDLE
        self.rep_state = RepState.IDLE
        self.rep_count = 0
        self.workout_data = []
        self.dashboard.reset()
        self.form_analyzer.reset()
        self.paused = False
        print("Workout reset!")
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        
        elapsed = time.time() - self.fps_start_time
        if elapsed >= 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()
    
    def run(self):
        """Main application loop"""
        # Initialize camera
        if not self.initialize_camera():
            print("Failed to initialize camera. Exiting.")
            return
        
        # Create window
        cv2.namedWindow(UI_CONFIG['window_name'], cv2.WINDOW_NORMAL)
        cv2.resizeWindow(UI_CONFIG['window_name'],
                        UI_CONFIG['window_width'],
                        UI_CONFIG['window_height'])
        
        print("\n" + "="*60)
        print("AI-POWERED PUSH-UP COUNTER")
        print("="*60)
        print("\nControls:")
        print("  S - Start workout")
        print("  P - Pause/Resume")
        print("  R - Reset")
        print("  Q - Quit")
        print("  V - Toggle voice feedback")
        print("  G - Toggle graphs")
        print("  K - Toggle skeleton")
        print("\nPosition yourself so your full body is visible from the side.")
        print("Press 'S' to start when ready!")
        print("="*60 + "\n")
        
        # Main loop
        try:
            while True:
                # Read frame
                ret, frame = self.cap.read()
                
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                # Mirror frame for better UX
                frame = cv2.flip(frame, 1)
                
                # Process frame
                if self.workout_state == WorkoutState.IDLE:
                    # Idle state - just show camera feed
                    cv2.putText(frame, "Press 'S' to START",
                               (self.frame_width // 2 - 200, self.frame_height // 2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS['good'], 3)
                    
                    # Still detect pose for preview
                    landmarks, _ = self.pose_detector.detect_pose(frame)
                    if landmarks:
                        # Get angle for preview
                        angle, _ = self.pose_detector.get_elbow_angle(
                            landmarks, self.frame_width, self.frame_height
                        )
                        alignment_score, _ = self.pose_detector.get_body_alignment(landmarks)
                        form_score, _ = self.form_analyzer.calculate_form_score(
                            landmarks, angle, alignment_score
                        )
                        
                        # Update dashboard for preview
                        self.dashboard.update_realtime_data(angle, form_score, time.time())
                        
                        if self.show_skeleton:
                            frame = self.pose_detector.draw_skeleton(
                                frame, landmarks, COLORS['skeleton_good'],
                                COLORS['skeleton_bad']
                            )
                    
                    # Show graphs even in idle state
                    if self.show_graphs:
                        frame = self.dashboard.draw_live_graphs(frame)
                    
                    # Show basic metrics
                    frame = self.dashboard.draw_main_metrics(
                        frame, self.current_angle if hasattr(self, 'current_angle') else 160,
                        self.current_form_score if hasattr(self, 'current_form_score') else 0,
                        []
                    )
                
                elif self.workout_state == WorkoutState.COMPLETED:
                    # Show summary
                    summary = self.form_analyzer.get_workout_summary()
                    summary['duration'] = time.time() - self.workout_start_time
                    frame = self.dashboard.draw_session_summary(frame, summary)
                
                else:
                    # Active workout - process frame
                    if not self.paused:
                        frame, pose_detected = self.process_frame(frame)
                        
                        # Start rep tracking after first movement
                        if pose_detected and self.rep_state == RepState.IDLE:
                            self.rep_state = RepState.UP
                            self.form_analyzer.start_rep(time.time())
                
                # Update FPS
                self.update_fps()
                
                # Display frame
                cv2.imshow(UI_CONFIG['window_name'], frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key != 255:  # Key was pressed
                    if not self.handle_key_press(key):
                        break
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        except Exception as e:
            print(f"\nError in main loop: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        
        # Complete workout if active
        if self.workout_state == WorkoutState.ACTIVE:
            print("Saving workout...")
            self.complete_workout()
        
        # Release camera
        if self.cap:
            self.cap.release()
        
        # Close windows
        cv2.destroyAllWindows()
        
        # Shutdown audio
        self.audio.shutdown()
        
        # Close database
        self.database.close()
        
        print("Goodbye!")


def main():
    """Main entry point"""
    print("\n" + "="*60)
    print("  AI-POWERED PUSH-UP COUNTER WITH POSE DETECTION")
    print("  Advanced Form Analysis & Real-Time Feedback")
    print("="*60 + "\n")
    
    # Parse command line arguments (simple version)
    workout_mode = 'FREE'
    goal = 0
    
    if len(sys.argv) > 1:
        if sys.argv[1].upper() == 'GOAL' and len(sys.argv) > 2:
            workout_mode = 'GOAL'
            goal = int(sys.argv[2])
            print(f"Mode: GOAL ({goal} reps)")
        else:
            workout_mode = sys.argv[1].upper()
            print(f"Mode: {workout_mode}")
    else:
        print("Mode: FREE (unlimited)")
    
    print()
    
    # Create and run counter
    counter = PushUpCounter(workout_mode=workout_mode, goal=goal)
    counter.run()


if __name__ == "__main__":
    main()



