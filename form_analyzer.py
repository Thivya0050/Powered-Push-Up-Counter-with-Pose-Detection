"""
Form Analyzer Module
Advanced form analysis and quality scoring for push-ups
"""

import numpy as np
import time
from typing import Dict, List, Tuple, Optional
from collections import deque

from config import (
    FORM_WEIGHTS,
    FORM_QUALITY_THRESHOLDS,
    MAX_BODY_ALIGNMENT_DEVIATION,
    MAX_ELBOW_ANGLE_FROM_BODY,
    MIN_REP_DURATION,
    MAX_REP_DURATION,
    DOWN_ANGLE_THRESHOLD,
    UP_ANGLE_THRESHOLD,
    LandmarkIndices
)


class FormAnalyzer:
    """Analyzes push-up form quality and provides detailed feedback"""
    
    def __init__(self):
        """Initialize form analyzer"""
        # Rep tracking
        self.current_rep_metrics = {}
        self.rep_history = []
        self.max_reps_history = 100  # Keep last 100 reps
        
        # Form tracking
        self.form_score_history = deque(maxlen=30)  # Last 30 frames
        self.mistake_counts = {}
        
        # Tempo tracking
        self.phase_times = {
            'descend': [],
            'hold': [],
            'ascend': []
        }
        
        # Fatigue detection
        self.baseline_form_score = None
        self.fatigue_threshold = 0.75  # 75% of baseline
        
        # Current state
        self.active_mistakes = []
        self.rep_start_time = None
        self.rep_phase = 'up'  # 'up', 'descending', 'down', 'ascending'
        
    def calculate_form_score(self, landmarks, angle: float, 
                            alignment_score: float) -> Tuple[float, Dict]:
        """
        Calculate comprehensive form score
        
        Args:
            landmarks: MediaPipe landmarks object
            angle: Current elbow angle
            alignment_score: Body alignment score from pose detector
            
        Returns:
            Tuple of (overall_score 0-100, component_scores dict)
        """
        scores = {}
        
        # 1. Body alignment (30%)
        scores['alignment'] = alignment_score * FORM_WEIGHTS['body_alignment']
        
        # 2. Elbow consistency (25%)
        elbow_score = self._check_elbow_position(landmarks)
        scores['elbow'] = elbow_score * FORM_WEIGHTS['elbow_consistency']
        
        # 3. Depth consistency (25%)
        depth_score = self._check_depth(angle)
        scores['depth'] = depth_score * FORM_WEIGHTS['depth_consistency']
        
        # 4. Speed control (20%)
        tempo_score = self._check_tempo()
        scores['tempo'] = tempo_score * FORM_WEIGHTS['speed_control']
        
        # Calculate total score
        total_score = sum(scores.values())
        
        # Store in history
        self.form_score_history.append(total_score)
        
        # Update baseline if not set
        if self.baseline_form_score is None and len(self.form_score_history) > 10:
            self.baseline_form_score = np.mean(list(self.form_score_history)[:10])
        
        return total_score, scores
    
    def _check_elbow_position(self, landmarks) -> float:
        """
        Check if elbows are properly positioned (close to body, not flared)
        
        Args:
            landmarks: MediaPipe landmarks object
            
        Returns:
            Score 0-100
        """
        # Get shoulder, elbow, and hip positions
        left_shoulder = landmarks.landmark[LandmarkIndices.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[LandmarkIndices.RIGHT_SHOULDER]
        left_elbow = landmarks.landmark[LandmarkIndices.LEFT_ELBOW]
        right_elbow = landmarks.landmark[LandmarkIndices.RIGHT_ELBOW]
        left_hip = landmarks.landmark[LandmarkIndices.LEFT_HIP]
        right_hip = landmarks.landmark[LandmarkIndices.RIGHT_HIP]
        
        # Calculate shoulder width
        shoulder_width = np.sqrt(
            (right_shoulder.x - left_shoulder.x)**2 +
            (right_shoulder.y - left_shoulder.y)**2
        )
        
        # Calculate elbow distance from body centerline
        # Body centerline is midpoint between shoulders and hips
        body_center_x = (left_shoulder.x + right_shoulder.x + 
                        left_hip.x + right_hip.x) / 4
        
        left_elbow_dist = abs(left_elbow.x - body_center_x)
        right_elbow_dist = abs(right_elbow.x - body_center_x)
        
        # Ideal elbow position is about shoulder_width/2 from center
        ideal_dist = shoulder_width / 2
        
        # Calculate deviation
        left_deviation = abs(left_elbow_dist - ideal_dist) / shoulder_width
        right_deviation = abs(right_elbow_dist - ideal_dist) / shoulder_width
        avg_deviation = (left_deviation + right_deviation) / 2
        
        # Convert to score (less deviation = higher score)
        score = max(0, 100 - (avg_deviation * 200))
        
        return score
    
    def _check_depth(self, angle: float) -> float:
        """
        Check if user is achieving proper depth
        
        Args:
            angle: Current elbow angle
            
        Returns:
            Score 0-100
        """
        if self.rep_phase in ['down', 'descending']:
            # During down phase, reward reaching target angle
            if angle <= DOWN_ANGLE_THRESHOLD:
                score = 100
            else:
                # Penalize proportionally to how far above threshold
                deficit = angle - DOWN_ANGLE_THRESHOLD
                score = max(0, 100 - (deficit * 2))
        else:
            # During up phase, check if maintaining extended position
            if angle >= UP_ANGLE_THRESHOLD - 10:
                score = 100
            else:
                score = 80  # Slight penalty if not fully extended
        
        return score
    
    def _check_tempo(self) -> float:
        """
        Check if movement tempo is controlled (not too fast/slow)
        
        Returns:
            Score 0-100
        """
        if not self.phase_times['descend']:
            return 100  # No data yet
        
        # Get recent descend times
        recent_descends = self.phase_times['descend'][-5:]
        if not recent_descends:
            return 100
        
        avg_descend = np.mean(recent_descends)
        
        # Ideal tempo: 1-2 seconds down, 1-2 seconds up
        # Too fast: < 0.8 seconds
        # Too slow: > 3 seconds
        
        if 0.8 <= avg_descend <= 3.0:
            score = 100
        elif avg_descend < 0.8:
            # Too fast - penalize more
            score = max(0, 50 - ((0.8 - avg_descend) * 50))
        else:
            # Too slow - minor penalty
            score = max(60, 100 - ((avg_descend - 3.0) * 10))
        
        return score
    
    def detect_mistakes(self, landmarks, angle: float, 
                       alignment_details: Dict) -> List[Dict]:
        """
        Detect specific form mistakes
        
        Args:
            landmarks: MediaPipe landmarks object
            angle: Current elbow angle
            alignment_details: Alignment information from pose detector
            
        Returns:
            List of mistake dictionaries with type, severity, and message
        """
        mistakes = []
        
        # 1. Check for sagging/raised hips
        if alignment_details.get('issue'):
            issue = alignment_details['issue']
            if issue == 'hips_sagging':
                mistakes.append({
                    'type': 'hips_sagging',
                    'severity': 'high',
                    'message': 'Keep your hips up! Engage your core.'
                })
            elif issue == 'hips_too_high':
                mistakes.append({
                    'type': 'hips_too_high',
                    'severity': 'high',
                    'message': 'Lower your hips! Keep body straight.'
                })
        
        # 2. Check for incomplete range of motion
        if self.rep_phase == 'down' and angle > DOWN_ANGLE_THRESHOLD + 15:
            mistakes.append({
                'type': 'incomplete_depth',
                'severity': 'medium',
                'message': 'Go deeper! Lower your chest more.'
            })
        
        # 3. Check for flared elbows
        elbow_score = self._check_elbow_position(landmarks)
        if elbow_score < 60:
            mistakes.append({
                'type': 'flared_elbows',
                'severity': 'medium',
                'message': 'Keep elbows closer to your body!'
            })
        
        # 4. Check for too fast tempo
        if self.phase_times['descend'] and len(self.phase_times['descend']) > 0:
            last_descend = self.phase_times['descend'][-1]
            if last_descend < 0.6:
                mistakes.append({
                    'type': 'too_fast',
                    'severity': 'low',
                    'message': 'Slow down! Control the movement.'
                })
        
        # 5. Check head position (neck alignment)
        head_score = self._check_head_position(landmarks)
        if head_score < 70:
            mistakes.append({
                'type': 'head_dropping',
                'severity': 'low',
                'message': 'Keep your head neutral! Look slightly ahead.'
            })
        
        # 6. Check for uneven arms
        if self._check_arm_symmetry(landmarks) < 70:
            mistakes.append({
                'type': 'uneven_arms',
                'severity': 'medium',
                'message': 'Keep both arms even!'
            })
        
        # Update mistake counts
        for mistake in mistakes:
            mistake_type = mistake['type']
            self.mistake_counts[mistake_type] = \
                self.mistake_counts.get(mistake_type, 0) + 1
        
        self.active_mistakes = mistakes
        return mistakes
    
    def _check_head_position(self, landmarks) -> float:
        """Check if head is in neutral position"""
        nose = landmarks.landmark[LandmarkIndices.NOSE]
        left_shoulder = landmarks.landmark[LandmarkIndices.LEFT_SHOULDER]
        right_shoulder = landmarks.landmark[LandmarkIndices.RIGHT_SHOULDER]
        
        # Head should be roughly in line with shoulders
        shoulder_mid_y = (left_shoulder.y + right_shoulder.y) / 2
        head_deviation = abs(nose.y - shoulder_mid_y)
        
        # Convert to score
        score = max(0, 100 - (head_deviation * 300))
        return score
    
    def _check_arm_symmetry(self, landmarks) -> float:
        """Check if both arms are moving symmetrically"""
        # Get both elbow angles
        left_shoulder = landmarks.landmark[LandmarkIndices.LEFT_SHOULDER]
        left_elbow = landmarks.landmark[LandmarkIndices.LEFT_ELBOW]
        left_wrist = landmarks.landmark[LandmarkIndices.LEFT_WRIST]
        
        right_shoulder = landmarks.landmark[LandmarkIndices.RIGHT_SHOULDER]
        right_elbow = landmarks.landmark[LandmarkIndices.RIGHT_ELBOW]
        right_wrist = landmarks.landmark[LandmarkIndices.RIGHT_WRIST]
        
        # Calculate angles (simple version, no smoothing)
        left_angle = self._calculate_simple_angle(
            (left_shoulder.x, left_shoulder.y),
            (left_elbow.x, left_elbow.y),
            (left_wrist.x, left_wrist.y)
        )
        
        right_angle = self._calculate_simple_angle(
            (right_shoulder.x, right_shoulder.y),
            (right_elbow.x, right_elbow.y),
            (right_wrist.x, right_wrist.y)
        )
        
        # Calculate difference
        angle_diff = abs(left_angle - right_angle)
        
        # Good symmetry: < 10 degrees difference
        score = max(0, 100 - (angle_diff * 5))
        return score
    
    def _calculate_simple_angle(self, a: tuple, b: tuple, c: tuple) -> float:
        """Simple angle calculation without smoothing"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        
        angle = np.arccos(cosine_angle)
        return np.degrees(angle)
    
    def assess_rep_quality(self, rep_data: Dict) -> Tuple[str, float]:
        """
        Assess quality of a completed rep
        
        Args:
            rep_data: Dictionary containing rep metrics
            
        Returns:
            Tuple of (quality_grade, quality_score)
        """
        form_score = rep_data.get('form_score', 0)
        duration = rep_data.get('duration', 0)
        min_angle = rep_data.get('min_angle', 180)
        mistakes = rep_data.get('mistakes', [])
        
        # Start with form score
        quality_score = form_score
        
        # Penalize for duration issues
        if duration < MIN_REP_DURATION:
            quality_score *= 0.9  # 10% penalty for too fast
        elif duration > MAX_REP_DURATION:
            quality_score *= 0.85  # 15% penalty for too slow
        
        # Penalize for insufficient depth
        if min_angle > DOWN_ANGLE_THRESHOLD + 20:
            quality_score *= 0.8  # 20% penalty
        
        # Penalize for high-severity mistakes
        high_severity_count = sum(1 for m in mistakes if m.get('severity') == 'high')
        quality_score -= (high_severity_count * 10)
        
        # Ensure score is in valid range
        quality_score = max(0, min(100, quality_score))
        
        # Assign grade
        if quality_score >= FORM_QUALITY_THRESHOLDS['perfect']:
            grade = 'Perfect'
        elif quality_score >= FORM_QUALITY_THRESHOLDS['good']:
            grade = 'Good'
        elif quality_score >= FORM_QUALITY_THRESHOLDS['fair']:
            grade = 'Fair'
        else:
            grade = 'Poor'
        
        return grade, quality_score
    
    def start_rep(self, timestamp: float):
        """Mark the start of a new rep"""
        self.rep_start_time = timestamp
        self.current_rep_metrics = {
            'start_time': timestamp,
            'angles': [],
            'form_scores': [],
            'mistakes': [],
        }
    
    def end_rep(self, timestamp: float, count: int) -> Dict:
        """
        Complete the current rep and calculate metrics
        
        Args:
            timestamp: End timestamp
            count: Rep number
            
        Returns:
            Complete rep data dictionary
        """
        if self.rep_start_time is None:
            return {}
        
        duration = timestamp - self.rep_start_time
        
        # Compile rep data
        rep_data = {
            'rep_number': count,
            'timestamp': timestamp,
            'duration': duration,
            'max_angle': max(self.current_rep_metrics.get('angles', [160])),
            'min_angle': min(self.current_rep_metrics.get('angles', [90])),
            'form_score': np.mean(self.current_rep_metrics.get('form_scores', [70])),
            'mistakes': self.current_rep_metrics.get('mistakes', []),
            'calories_burned': self._estimate_calories(duration)
        }
        
        # Assess quality
        grade, quality_score = self.assess_rep_quality(rep_data)
        rep_data['quality_grade'] = grade
        rep_data['quality_score'] = quality_score
        
        # Add to history
        self.rep_history.append(rep_data)
        if len(self.rep_history) > self.max_reps_history:
            self.rep_history.pop(0)
        
        # Reset for next rep
        self.rep_start_time = None
        self.current_rep_metrics = {}
        
        return rep_data
    
    def update_rep_phase(self, phase: str, timestamp: float):
        """
        Update current rep phase and track timing
        
        Args:
            phase: 'up', 'descending', 'down', 'ascending'
            timestamp: Current timestamp
        """
        # Track phase transition times
        if phase != self.rep_phase:
            if hasattr(self, 'last_phase_change'):
                phase_duration = timestamp - self.last_phase_change
                
                if self.rep_phase == 'descending':
                    self.phase_times['descend'].append(phase_duration)
                elif self.rep_phase == 'ascending':
                    self.phase_times['ascend'].append(phase_duration)
                
                # Keep only recent history
                for key in self.phase_times:
                    if len(self.phase_times[key]) > 20:
                        self.phase_times[key] = self.phase_times[key][-20:]
            
            self.last_phase_change = timestamp
            self.rep_phase = phase
    
    def update_current_rep_metrics(self, angle: float, form_score: float, 
                                   mistakes: List[Dict]):
        """Update metrics for the current rep in progress"""
        if self.current_rep_metrics:
            self.current_rep_metrics.setdefault('angles', []).append(angle)
            self.current_rep_metrics.setdefault('form_scores', []).append(form_score)
            
            # Only add unique mistakes
            for mistake in mistakes:
                if mistake not in self.current_rep_metrics.get('mistakes', []):
                    self.current_rep_metrics.setdefault('mistakes', []).append(mistake)
    
    def _estimate_calories(self, rep_duration: float) -> float:
        """
        Estimate calories burned for one rep
        
        Args:
            rep_duration: Duration of the rep in seconds
            
        Returns:
            Estimated calories
        """
        # Base calorie burn per push-up
        base_calories = 0.32
        
        # Adjust based on duration (slower = more calories)
        if rep_duration > 2.5:
            modifier = 1.2
        elif rep_duration < 1.5:
            modifier = 0.8
        else:
            modifier = 1.0
        
        return base_calories * modifier
    
    def detect_fatigue(self) -> Tuple[bool, float]:
        """
        Detect if user is showing signs of fatigue
        
        Returns:
            Tuple of (is_fatigued bool, fatigue_percentage 0-1)
        """
        if not self.baseline_form_score or len(self.form_score_history) < 10:
            return False, 0.0
        
        # Calculate recent average form score
        recent_scores = list(self.form_score_history)[-10:]
        recent_avg = np.mean(recent_scores)
        
        # Calculate fatigue as percentage drop from baseline
        fatigue_percentage = 1 - (recent_avg / self.baseline_form_score)
        fatigue_percentage = max(0, min(1, fatigue_percentage))
        
        # Check if fatigued
        is_fatigued = recent_avg < (self.baseline_form_score * self.fatigue_threshold)
        
        return is_fatigued, fatigue_percentage
    
    def get_most_common_mistakes(self, top_n: int = 3) -> List[Tuple[str, int]]:
        """
        Get the most common mistakes
        
        Args:
            top_n: Number of top mistakes to return
            
        Returns:
            List of (mistake_type, count) tuples
        """
        if not self.mistake_counts:
            return []
        
        sorted_mistakes = sorted(
            self.mistake_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return sorted_mistakes[:top_n]
    
    def get_improvement_suggestions(self) -> List[str]:
        """
        Generate personalized improvement suggestions
        
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        # Get most common mistakes
        common_mistakes = self.get_most_common_mistakes(3)
        
        mistake_suggestions = {
            'hips_sagging': 'Focus on engaging your core throughout the movement.',
            'hips_too_high': 'Lower your hips to maintain a straight body line.',
            'incomplete_depth': 'Work on achieving full range of motion - lower until elbows reach 90°.',
            'flared_elbows': 'Keep your elbows at 45° angle from your body.',
            'too_fast': 'Slow down and control the movement for better results.',
            'head_dropping': 'Keep your neck neutral - look slightly ahead, not down.',
            'uneven_arms': 'Focus on symmetry - both arms should move together.',
        }
        
        for mistake_type, count in common_mistakes:
            if mistake_type in mistake_suggestions:
                suggestions.append(mistake_suggestions[mistake_type])
        
        # Check for fatigue
        is_fatigued, _ = self.detect_fatigue()
        if is_fatigued:
            suggestions.append('Your form is deteriorating. Consider taking a rest.')
        
        # Check rep consistency
        if len(self.rep_history) > 5:
            recent_grades = [rep['quality_grade'] for rep in self.rep_history[-5:]]
            if recent_grades.count('Poor') >= 3:
                suggestions.append('Quality over quantity! Focus on perfect form over more reps.')
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def get_workout_summary(self) -> Dict:
        """
        Generate summary statistics for the workout
        
        Returns:
            Dictionary with summary stats
        """
        if not self.rep_history:
            return {}
        
        total_reps = len(self.rep_history)
        
        # Calculate grade distribution
        grades = [rep['quality_grade'] for rep in self.rep_history]
        grade_counts = {
            'Perfect': grades.count('Perfect'),
            'Good': grades.count('Good'),
            'Fair': grades.count('Fair'),
            'Poor': grades.count('Poor')
        }
        
        # Calculate averages
        avg_form_score = np.mean([rep['form_score'] for rep in self.rep_history])
        avg_duration = np.mean([rep['duration'] for rep in self.rep_history])
        total_calories = sum([rep['calories_burned'] for rep in self.rep_history])
        
        # Find best and worst reps
        best_rep = max(self.rep_history, key=lambda x: x['quality_score'])
        worst_rep = min(self.rep_history, key=lambda x: x['quality_score'])
        
        summary = {
            'total_reps': total_reps,
            'grade_distribution': grade_counts,
            'average_form_score': avg_form_score,
            'average_rep_duration': avg_duration,
            'total_calories': total_calories,
            'best_rep': best_rep['rep_number'],
            'worst_rep': worst_rep['rep_number'],
            'most_common_mistakes': self.get_most_common_mistakes(),
            'improvement_suggestions': self.get_improvement_suggestions()
        }
        
        return summary
    
    def reset(self):
        """Reset analyzer for new workout"""
        self.rep_history = []
        self.form_score_history.clear()
        self.mistake_counts = {}
        self.phase_times = {'descend': [], 'hold': [], 'ascend': []}
        self.baseline_form_score = None
        self.active_mistakes = []
        self.current_rep_metrics = {}





