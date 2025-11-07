"""
Audio Feedback Module
Handles text-to-speech and sound feedback for the push-up counter
"""

import pyttsx3
import threading
import time
from queue import Queue, Empty
from typing import Optional, List
import random

from config import (
    AUDIO_CONFIG,
    MOTIVATIONAL_PHRASES,
    FeedbackLevel
)


class AudioFeedback:
    """Manages voice and sound feedback during workouts"""
    
    def __init__(self, enabled: bool = True, verbosity: int = FeedbackLevel.MODERATE):
        """
        Initialize audio feedback system
        
        Args:
            enabled: Whether audio is enabled
            verbosity: Feedback verbosity level
        """
        self.enabled = enabled and AUDIO_CONFIG['voice_feedback_enabled']
        self.verbosity = verbosity
        
        # Text-to-speech engine
        self.engine = None
        self.engine_ready = False
        
        # Speech queue for non-blocking announcements
        self.speech_queue = Queue()
        self.speech_thread = None
        self.running = False
        
        # Cooldown tracking
        self.last_announcement_time = {}
        self.cooldown = AUDIO_CONFIG['alert_cooldown']
        
        # Count tracking
        self.count_interval = AUDIO_CONFIG['count_announcement_interval']
        self.milestones = AUDIO_CONFIG['milestone_announcements']
        self.last_announced_count = 0
        
        # Initialize if enabled
        if self.enabled:
            self._initialize_engine()
            self._start_speech_thread()
    
    def _initialize_engine(self):
        """Initialize the TTS engine"""
        try:
            self.engine = pyttsx3.init()
            
            # Configure voice properties
            self.engine.setProperty('rate', AUDIO_CONFIG['tts_rate'])
            self.engine.setProperty('volume', AUDIO_CONFIG['tts_volume'])
            
            # Try to set a voice (optional)
            voices = self.engine.getProperty('voices')
            if voices:
                # Use first available voice (can be customized)
                self.engine.setProperty('voice', voices[0].id)
            
            self.engine_ready = True
        except Exception as e:
            print(f"Warning: Could not initialize text-to-speech: {e}")
            self.engine_ready = False
            self.enabled = False
    
    def _start_speech_thread(self):
        """Start background thread for speech synthesis"""
        if not self.engine_ready:
            return
        
        self.running = True
        self.speech_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.speech_thread.start()
    
    def _speech_worker(self):
        """Background worker for processing speech queue"""
        while self.running:
            try:
                # Get message from queue with timeout
                message = self.speech_queue.get(timeout=0.5)
                
                if message and self.engine_ready:
                    try:
                        self.engine.say(message)
                        self.engine.runAndWait()
                    except Exception as e:
                        print(f"Speech error: {e}")
                
                self.speech_queue.task_done()
                
            except Empty:
                continue
            except Exception as e:
                print(f"Speech worker error: {e}")
    
    def _can_announce(self, message_type: str) -> bool:
        """
        Check if enough time has passed since last announcement of this type
        
        Args:
            message_type: Type of message
            
        Returns:
            True if can announce
        """
        if message_type not in self.last_announcement_time:
            return True
        
        elapsed = time.time() - self.last_announcement_time[message_type]
        return elapsed >= self.cooldown
    
    def _record_announcement(self, message_type: str):
        """Record that an announcement was made"""
        self.last_announcement_time[message_type] = time.time()
    
    def announce(self, message: str, message_type: str = 'general', 
                priority: bool = False):
        """
        Announce a message via text-to-speech
        
        Args:
            message: Message to speak
            message_type: Type of message (for cooldown tracking)
            priority: If True, bypass cooldown
        """
        if not self.enabled or not self.engine_ready:
            return
        
        # Check cooldown unless priority
        if not priority and not self._can_announce(message_type):
            return
        
        # Add to queue
        self.speech_queue.put(message)
        self._record_announcement(message_type)
    
    def announce_count(self, count: int):
        """
        Announce rep count at appropriate intervals
        
        Args:
            count: Current rep count
        """
        if self.verbosity < FeedbackLevel.MINIMAL:
            return
        
        # Check if this is a milestone
        if count in self.milestones:
            message = f"{count} push-ups! {random.choice(MOTIVATIONAL_PHRASES['milestone'])}"
            self.announce(message, 'milestone', priority=True)
            self.last_announced_count = count
            return
        
        # Check if should announce based on interval
        if count > 0 and count % self.count_interval == 0:
            if count != self.last_announced_count:
                message = f"{count}"
                self.announce(message, 'count')
                self.last_announced_count = count
    
    def announce_form_correction(self, mistake_type: str, message: str):
        """
        Announce form correction alert
        
        Args:
            mistake_type: Type of mistake
            message: Correction message
        """
        if self.verbosity < FeedbackLevel.MODERATE:
            return
        
        # Only announce if verbose mode or high severity
        if self.verbosity >= FeedbackLevel.VERBOSE:
            self.announce(message, f'form_{mistake_type}')
    
    def announce_perfect_form(self):
        """Announce when user achieves perfect form"""
        if self.verbosity < FeedbackLevel.MODERATE:
            return
        
        message = random.choice(MOTIVATIONAL_PHRASES['perfect_form'])
        self.announce(message, 'perfect_form')
    
    def announce_workout_start(self):
        """Announce workout start"""
        if self.verbosity < FeedbackLevel.MINIMAL:
            return
        
        messages = [
            "Get ready to start!",
            "Let's begin!",
            "Starting workout!",
            "Time to push!"
        ]
        message = random.choice(messages)
        self.announce(message, 'workout_start', priority=True)
    
    def announce_countdown(self, seconds: int):
        """
        Announce countdown
        
        Args:
            seconds: Seconds remaining
        """
        if self.verbosity < FeedbackLevel.MINIMAL:
            return
        
        if seconds in [3, 2, 1]:
            self.announce(str(seconds), 'countdown', priority=True)
        elif seconds == 0:
            self.announce("Go!", 'countdown', priority=True)
    
    def announce_workout_complete(self, total_reps: int, avg_form_score: float):
        """
        Announce workout completion with summary
        
        Args:
            total_reps: Total reps completed
            avg_form_score: Average form score
        """
        if self.verbosity < FeedbackLevel.MINIMAL:
            return
        
        # Base completion message
        message = f"Workout complete! {total_reps} push-ups. "
        
        # Add form score feedback
        if avg_form_score >= 90:
            message += "Excellent form!"
        elif avg_form_score >= 75:
            message += "Good form!"
        elif avg_form_score >= 60:
            message += "Keep practicing!"
        
        self.announce(message, 'workout_complete', priority=True)
    
    def announce_new_record(self, record_type: str):
        """
        Announce new personal record
        
        Args:
            record_type: Type of record (e.g., 'most_reps', 'best_form')
        """
        if self.verbosity < FeedbackLevel.MINIMAL:
            return
        
        message = random.choice(MOTIVATIONAL_PHRASES['new_record'])
        self.announce(message, 'new_record', priority=True)
    
    def announce_streak(self, streak_count: int):
        """
        Announce perfect form streak
        
        Args:
            streak_count: Number of consecutive perfect reps
        """
        if self.verbosity < FeedbackLevel.MODERATE:
            return
        
        if streak_count in [5, 10, 15, 20]:
            message = f"{streak_count} perfect reps in a row!"
            self.announce(message, 'streak')
    
    def announce_rest_suggestion(self):
        """Suggest taking a rest"""
        if self.verbosity < FeedbackLevel.MODERATE:
            return
        
        messages = [
            "Consider taking a rest",
            "Your form is getting tired. Rest when needed.",
            "Take a break if you need one"
        ]
        message = random.choice(messages)
        self.announce(message, 'rest_suggestion')
    
    def announce_fatigue_detected(self):
        """Announce fatigue detection"""
        if self.verbosity < FeedbackLevel.MODERATE:
            return
        
        messages = [
            "Fatigue detected. Take a break!",
            "Your form is deteriorating. Rest up!",
            "Time for a rest!"
        ]
        message = random.choice(messages)
        self.announce(message, 'fatigue', priority=True)
    
    def announce_goal_reached(self, goal: int):
        """
        Announce goal completion
        
        Args:
            goal: Goal that was reached
        """
        if self.verbosity < FeedbackLevel.MINIMAL:
            return
        
        message = f"Goal reached! {goal} push-ups completed!"
        self.announce(message, 'goal_reached', priority=True)
    
    def announce_halfway_to_goal(self, goal: int):
        """
        Announce halfway point to goal
        
        Args:
            goal: Total goal
        """
        if self.verbosity < FeedbackLevel.MODERATE:
            return
        
        message = f"Halfway there! {goal // 2} down, {goal // 2} to go!"
        self.announce(message, 'halfway')
    
    def announce_achievement(self, achievement_name: str):
        """
        Announce achievement unlock
        
        Args:
            achievement_name: Name of the achievement
        """
        if self.verbosity < FeedbackLevel.MINIMAL:
            return
        
        message = f"Achievement unlocked! {achievement_name}!"
        self.announce(message, 'achievement', priority=True)
    
    def announce_calibration_start(self):
        """Announce calibration start"""
        message = "Starting calibration. Please do a few push-ups with full range of motion."
        self.announce(message, 'calibration', priority=True)
    
    def announce_calibration_complete(self):
        """Announce calibration completion"""
        message = "Calibration complete! Ready to start."
        self.announce(message, 'calibration', priority=True)
    
    def play_sound_effect(self, sound_type: str):
        """
        Play a sound effect (beep/tone)
        
        Args:
            sound_type: Type of sound ('rep', 'milestone', 'error', 'start', 'end')
        """
        # Placeholder for sound effects
        # In a full implementation, this would play actual sound files
        # using libraries like pygame or playsound
        
        # For now, we'll use system beep as a simple alternative
        if sound_type in ['rep', 'milestone']:
            # Could play success sound
            pass
        elif sound_type == 'error':
            # Could play error sound
            pass
    
    def set_verbosity(self, level: int):
        """
        Set feedback verbosity level
        
        Args:
            level: FeedbackLevel constant
        """
        self.verbosity = level
    
    def enable(self):
        """Enable audio feedback"""
        if not self.engine_ready:
            self._initialize_engine()
        
        if self.engine_ready and not self.running:
            self._start_speech_thread()
        
        self.enabled = True
    
    def disable(self):
        """Disable audio feedback"""
        self.enabled = False
    
    def toggle(self):
        """Toggle audio feedback on/off"""
        if self.enabled:
            self.disable()
        else:
            self.enable()
    
    def clear_queue(self):
        """Clear all pending announcements"""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except Empty:
                break
    
    def wait_for_queue(self, timeout: float = 5.0):
        """
        Wait for all queued announcements to complete
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        try:
            self.speech_queue.join()
        except:
            pass
    
    def shutdown(self):
        """Shutdown audio system"""
        self.running = False
        
        # Wait for speech thread to finish
        if self.speech_thread and self.speech_thread.is_alive():
            self.speech_thread.join(timeout=2.0)
        
        # Stop engine
        if self.engine_ready and self.engine:
            try:
                self.engine.stop()
            except:
                pass
    
    def __del__(self):
        """Cleanup on deletion"""
        self.shutdown()


class SoundPlayer:
    """Helper class for playing sound effects"""
    
    def __init__(self):
        """Initialize sound player"""
        self.sounds_enabled = True
        self.sounds = {}
        
        # In a full implementation, load sound files here
        # self.sounds['rep'] = 'assets/sounds/rep.wav'
        # self.sounds['milestone'] = 'assets/sounds/milestone.wav'
        # etc.
    
    def play(self, sound_name: str):
        """
        Play a sound effect
        
        Args:
            sound_name: Name of the sound to play
        """
        if not self.sounds_enabled:
            return
        
        # Placeholder - in full implementation:
        # if sound_name in self.sounds:
        #     playsound(self.sounds[sound_name], block=False)
        pass
    
    def enable(self):
        """Enable sound effects"""
        self.sounds_enabled = True
    
    def disable(self):
        """Disable sound effects"""
        self.sounds_enabled = False
    
    def toggle(self):
        """Toggle sound effects"""
        self.sounds_enabled = not self.sounds_enabled


# Create global instances for easy access
_audio_feedback = None
_sound_player = None


def get_audio_feedback(enabled: bool = True, 
                      verbosity: int = FeedbackLevel.MODERATE) -> AudioFeedback:
    """
    Get or create the global AudioFeedback instance
    
    Args:
        enabled: Whether audio is enabled
        verbosity: Feedback verbosity level
        
    Returns:
        AudioFeedback instance
    """
    global _audio_feedback
    
    if _audio_feedback is None:
        _audio_feedback = AudioFeedback(enabled, verbosity)
    
    return _audio_feedback


def get_sound_player() -> SoundPlayer:
    """
    Get or create the global SoundPlayer instance
    
    Returns:
        SoundPlayer instance
    """
    global _sound_player
    
    if _sound_player is None:
        _sound_player = SoundPlayer()
    
    return _sound_player





