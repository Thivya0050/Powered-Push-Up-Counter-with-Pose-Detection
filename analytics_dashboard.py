"""
Analytics Dashboard Module
Real-time visualization and analytics for push-up counter
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
from collections import deque
from typing import List, Dict, Tuple, Optional
import time

from config import COLORS, FONTS, UI_CONFIG


class AnalyticsDashboard:
    """Real-time analytics and visualization for workouts"""
    
    def __init__(self, max_history: int = 100):
        """
        Initialize dashboard
        
        Args:
            max_history: Maximum number of data points to store
        """
        self.max_history = max_history
        
        # Real-time data
        self.angle_history = deque(maxlen=max_history)
        self.form_score_history = deque(maxlen=max_history)
        self.time_history = deque(maxlen=max_history)
        
        # Rep data
        self.rep_durations = []
        self.rep_quality_grades = {'Perfect': 0, 'Good': 0, 'Fair': 0, 'Poor': 0}
        
        # Workout metrics
        self.start_time = None
        self.current_rep_count = 0
        self.total_calories = 0
        self.perfect_streak = 0
        self.max_streak = 0
        
        # Graph update throttling
        self.last_graph_update = 0
        self.graph_update_interval = 0.05  # Update every 0.05 seconds (faster)
        
    def update_realtime_data(self, angle: float, form_score: float, timestamp: float):
        """
        Update real-time metrics
        
        Args:
            angle: Current elbow angle
            form_score: Current form score
            timestamp: Current timestamp
        """
        self.angle_history.append(angle)
        self.form_score_history.append(form_score)
        self.time_history.append(timestamp)
    
    def record_rep(self, rep_data: Dict):
        """
        Record a completed rep
        
        Args:
            rep_data: Rep data dictionary
        """
        self.current_rep_count = rep_data.get('rep_number', 0)
        self.total_calories += rep_data.get('calories_burned', 0)
        
        # Update grade distribution
        grade = rep_data.get('quality_grade', 'Poor')
        self.rep_quality_grades[grade] = self.rep_quality_grades.get(grade, 0) + 1
        
        # Update duration history
        self.rep_durations.append(rep_data.get('duration', 0))
        
        # Update streak
        if grade == 'Perfect':
            self.perfect_streak += 1
            self.max_streak = max(self.max_streak, self.perfect_streak)
        else:
            self.perfect_streak = 0
    
    def start_workout(self):
        """Mark workout start time"""
        self.start_time = time.time()
    
    def get_elapsed_time(self) -> float:
        """Get elapsed workout time in seconds"""
        if self.start_time is None:
            return 0
        return time.time() - self.start_time
    
    def format_time(self, seconds: float) -> str:
        """
        Format seconds as MM:SS
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted time string
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"
    
    def draw_main_metrics(self, frame: np.ndarray, current_angle: float,
                         current_form_score: float, active_mistakes: List[Dict]) -> np.ndarray:
        """
        Draw main metrics overlay on frame
        
        Args:
            frame: Video frame
            current_angle: Current elbow angle
            current_form_score: Current form score
            active_mistakes: List of active form mistakes
            
        Returns:
            Frame with metrics overlay
        """
        h, w = frame.shape[:2]
        
        # Draw semi-transparent panel for metrics
        panel_height = 200
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = COLORS['panel_bg'][:3]
        
        # Blend panel with frame
        alpha = 0.7
        frame[0:panel_height] = cv2.addWeighted(
            frame[0:panel_height], 1 - alpha, panel, alpha, 0
        )
        
        # Draw rep count (large)
        count_text = str(self.current_rep_count)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 3.0
        thickness = 5
        
        text_size = cv2.getTextSize(count_text, font, font_scale, thickness)[0]
        count_x = 30
        count_y = 100
        
        # Draw shadow
        cv2.putText(frame, count_text, (count_x + 3, count_y + 3),
                   font, font_scale, (0, 0, 0), thickness)
        # Draw text
        cv2.putText(frame, count_text, (count_x, count_y),
                   font, font_scale, COLORS['ui_primary'], thickness)
        
        # Label
        cv2.putText(frame, "REPS", (count_x, count_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
        
        # Draw form score with color coding
        form_x = 250
        form_y = 80
        
        # Determine color based on score
        if current_form_score >= 85:
            form_color = COLORS['good']
        elif current_form_score >= 70:
            form_color = COLORS['warning']
        else:
            form_color = COLORS['error']
        
        # Draw circular progress indicator
        center = (form_x + 50, form_y)
        radius = 40
        
        # Background circle
        cv2.circle(frame, center, radius, (50, 50, 50), -1)
        cv2.circle(frame, center, radius, COLORS['ui_secondary'], 2)
        
        # Progress arc
        angle_start = -90
        angle_end = -90 + (current_form_score / 100 * 360)
        cv2.ellipse(frame, center, (radius, radius), 0,
                   angle_start, angle_end, form_color, 5)
        
        # Score text
        score_text = f"{int(current_form_score)}"
        text_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = center[0] - text_size[0] // 2
        text_y = center[1] + text_size[1] // 2
        cv2.putText(frame, score_text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['text'], 2)
        
        # Label
        cv2.putText(frame, "FORM", (form_x + 10, form_y + 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 1)
        
        # Draw current angle
        angle_x = 400
        angle_y = 60
        cv2.putText(frame, f"Angle: {int(current_angle)}°",
                   (angle_x, angle_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   COLORS['text'], 2)
        
        # Draw elapsed time
        elapsed = self.get_elapsed_time()
        time_text = f"Time: {self.format_time(elapsed)}"
        cv2.putText(frame, time_text, (angle_x, angle_y + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
        
        # Draw calories
        cal_text = f"Calories: {self.total_calories:.1f}"
        cv2.putText(frame, cal_text, (angle_x, angle_y + 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['text'], 2)
        
        # Draw perfect streak
        if self.perfect_streak > 0:
            streak_x = w - 200
            streak_y = 60
            streak_text = f"🔥 Streak: {self.perfect_streak}"
            cv2.putText(frame, streak_text, (streak_x, streak_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['good'], 2)
        
        # Draw active mistakes
        if active_mistakes:
            mistake_y = panel_height + 30
            cv2.putText(frame, "Form Issues:", (20, mistake_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['warning'], 2)
            
            for i, mistake in enumerate(active_mistakes[:3]):  # Show top 3
                msg = mistake.get('message', 'Form issue')
                severity = mistake.get('severity', 'low')
                
                # Color based on severity
                if severity == 'high':
                    color = COLORS['error']
                elif severity == 'medium':
                    color = COLORS['warning']
                else:
                    color = COLORS['ui_secondary']
                
                cv2.putText(frame, f"• {msg}", (30, mistake_y + 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return frame
    
    def draw_quality_distribution(self, frame: np.ndarray, x: int, y: int,
                                  width: int, height: int) -> np.ndarray:
        """
        Draw rep quality distribution bar chart
        
        Args:
            frame: Video frame
            x, y: Position to draw
            width, height: Size of chart
            
        Returns:
            Frame with chart
        """
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height),
                     COLORS['panel_bg'][:3], -1)
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Title
        cv2.putText(frame, "Rep Quality", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 2)
        
        # Calculate bar positions
        total_reps = sum(self.rep_quality_grades.values())
        if total_reps == 0:
            return frame
        
        bar_height = 20
        bar_spacing = 35
        bar_start_y = y + 50
        
        grade_colors = {
            'Perfect': COLORS['good'],
            'Good': (100, 200, 100),
            'Fair': COLORS['warning'],
            'Poor': COLORS['error']
        }
        
        for i, (grade, count) in enumerate(self.rep_quality_grades.items()):
            by = bar_start_y + i * bar_spacing
            
            # Label
            label = f"{grade}: {count}"
            cv2.putText(frame, label, (x + 10, by + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['text'], 1)
            
            # Bar
            bar_width = int((count / total_reps) * (width - 120))
            if bar_width > 0:
                cv2.rectangle(frame, (x + 100, by), (x + 100 + bar_width, by + bar_height),
                             grade_colors[grade], -1)
                
                # Percentage
                pct = f"{count / total_reps * 100:.0f}%"
                cv2.putText(frame, pct, (x + 105 + bar_width, by + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['text'], 1)
        
        return frame
    
    def create_angle_graph(self, width: int = 400, height: int = 200) -> Optional[np.ndarray]:
        """
        Create angle history line graph
        
        Args:
            width: Graph width
            height: Graph height
            
        Returns:
            Graph as numpy array or None
        """
        # Check throttling (less strict)
        current_time = time.time()
        if current_time - self.last_graph_update < self.graph_update_interval:
            return None
        
        self.last_graph_update = current_time
        
        # Show placeholder if not enough data
        if len(self.angle_history) < 2:
            # Create empty graph with placeholder
            try:
                dpi = 100
                fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, 'Waiting for data...', 
                       ha='center', va='center', color='white', fontsize=12)
                ax.set_facecolor('#1a1a1a')
                ax.set_title('Elbow Angle', color='white', fontsize=10)
                ax.set_ylim(50, 180)
                ax.tick_params(colors='white', labelsize=7)
                fig.patch.set_facecolor('#1a1a1a')
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                graph_image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                graph_image = graph_image.reshape(height, width, 4)
                graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2BGR)
                plt.close(fig)
                return graph_image
            except:
                return None
        
        try:
            # Create figure
            dpi = 100
            fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
            ax = fig.add_subplot(111)
            
            # Plot data
            angles = list(self.angle_history)
            x = range(len(angles))
            
            ax.plot(x, angles, color='cyan', linewidth=2)
            ax.fill_between(x, angles, alpha=0.3, color='cyan')
            
            # Add threshold lines
            ax.axhline(y=160, color='green', linestyle='--', alpha=0.5, label='Up')
            ax.axhline(y=90, color='red', linestyle='--', alpha=0.5, label='Down')
            
            # Styling
            ax.set_title('Elbow Angle', color='white', fontsize=10)
            ax.set_ylabel('Angle (°)', color='white', fontsize=8)
            ax.set_ylim(50, 180)
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white', labelsize=7)
            ax.legend(fontsize=7, loc='upper right')
            
            fig.patch.set_facecolor('#1a1a1a')
            
            # Convert to numpy array
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            graph_image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            graph_image = graph_image.reshape(height, width, 4)
            
            # Convert RGBA to BGR
            graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2BGR)
            
            plt.close(fig)
            
            return graph_image
            
        except Exception as e:
            print(f"Error creating angle graph: {e}")
            return None
    
    def create_form_score_graph(self, width: int = 400, height: int = 200) -> Optional[np.ndarray]:
        """
        Create form score history line graph
        
        Args:
            width: Graph width
            height: Graph height
            
        Returns:
            Graph as numpy array or None
        """
        # Show placeholder if not enough data
        if len(self.form_score_history) < 2:
            try:
                dpi = 100
                fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
                ax = fig.add_subplot(111)
                ax.text(0.5, 0.5, 'Waiting for data...', 
                       ha='center', va='center', color='white', fontsize=12)
                ax.set_facecolor('#1a1a1a')
                ax.set_title('Form Score', color='white', fontsize=10)
                ax.set_ylim(0, 100)
                ax.tick_params(colors='white', labelsize=7)
                fig.patch.set_facecolor('#1a1a1a')
                canvas = FigureCanvasAgg(fig)
                canvas.draw()
                graph_image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
                graph_image = graph_image.reshape(height, width, 4)
                graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2BGR)
                plt.close(fig)
                return graph_image
            except:
                return None
        
        try:
            # Create figure
            dpi = 100
            fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
            ax = fig.add_subplot(111)
            
            # Plot data
            scores = list(self.form_score_history)
            x = range(len(scores))
            
            # Color gradient based on score
            ax.plot(x, scores, color='yellow', linewidth=2)
            ax.fill_between(x, scores, alpha=0.3, color='yellow')
            
            # Add quality threshold lines
            ax.axhline(y=90, color='green', linestyle='--', alpha=0.3, label='Perfect')
            ax.axhline(y=75, color='yellow', linestyle='--', alpha=0.3, label='Good')
            ax.axhline(y=60, color='orange', linestyle='--', alpha=0.3, label='Fair')
            
            # Styling
            ax.set_title('Form Score', color='white', fontsize=10)
            ax.set_ylabel('Score', color='white', fontsize=8)
            ax.set_ylim(0, 100)
            ax.set_facecolor('#1a1a1a')
            ax.tick_params(colors='white', labelsize=7)
            ax.legend(fontsize=6, loc='lower right')
            
            fig.patch.set_facecolor('#1a1a1a')
            
            # Convert to numpy array
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            graph_image = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            graph_image = graph_image.reshape(height, width, 4)
            
            # Convert RGBA to BGR
            graph_image = cv2.cvtColor(graph_image, cv2.COLOR_RGBA2BGR)
            
            plt.close(fig)
            
            return graph_image
            
        except Exception as e:
            print(f"Error creating form score graph: {e}")
            return None
    
    def draw_live_graphs(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw live graphs on frame
        
        Args:
            frame: Video frame
            
        Returns:
            Frame with graphs
        """
        h, w = frame.shape[:2]
        
        # Position graphs on right side (make them more visible)
        graph_width = 400
        graph_height = 180
        graph_x = w - graph_width - 10
        
        # Angle graph
        angle_graph = self.create_angle_graph(graph_width, graph_height)
        if angle_graph is not None:
            graph_y = h - graph_height * 2 - 30
            try:
                # Ensure we don't go out of bounds
                if graph_y + graph_height <= h and graph_x + graph_width <= w:
                    frame[graph_y:graph_y + graph_height,
                          graph_x:graph_x + graph_width] = angle_graph
            except Exception as e:
                pass
        
        # Form score graph
        form_graph = self.create_form_score_graph(graph_width, graph_height)
        if form_graph is not None:
            graph_y = h - graph_height - 10
            try:
                # Ensure we don't go out of bounds
                if graph_y + graph_height <= h and graph_x + graph_width <= w:
                    frame[graph_y:graph_y + graph_height,
                          graph_x:graph_x + graph_width] = form_graph
            except Exception as e:
                pass
        
        return frame
    
    def draw_session_summary(self, frame: np.ndarray, summary: Dict) -> np.ndarray:
        """
        Draw workout session summary overlay
        
        Args:
            frame: Video frame
            summary: Summary statistics dictionary
            
        Returns:
            Frame with summary overlay
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent overlay
        overlay = frame.copy()
        
        # Draw centered panel
        panel_width = 600
        panel_height = 500
        panel_x = (w - panel_width) // 2
        panel_y = (h - panel_height) // 2
        
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     (30, 30, 30), -1)
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + panel_height),
                     COLORS['ui_primary'], 3)
        
        frame = cv2.addWeighted(frame, 0.3, overlay, 0.7, 0)
        
        # Title
        title_y = panel_y + 50
        cv2.putText(frame, "WORKOUT COMPLETE!", (panel_x + 120, title_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, COLORS['good'], 3)
        
        # Stats
        stats_x = panel_x + 50
        stats_y = title_y + 60
        line_height = 40
        
        stats = [
            f"Total Reps: {summary.get('total_reps', 0)}",
            f"Average Form Score: {summary.get('average_form_score', 0):.1f}",
            f"Total Calories: {summary.get('total_calories', 0):.1f}",
            f"Duration: {self.format_time(summary.get('duration', 0))}",
            f"Perfect Reps: {self.rep_quality_grades.get('Perfect', 0)}",
            f"Max Streak: {self.max_streak}",
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(frame, stat, (stats_x, stats_y + i * line_height),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS['text'], 2)
        
        # Grade distribution
        dist_y = stats_y + len(stats) * line_height + 30
        cv2.putText(frame, "Quality Distribution:", (stats_x, dist_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS['ui_primary'], 2)
        
        for i, (grade, count) in enumerate(self.rep_quality_grades.items()):
            text = f"{grade}: {count}"
            cv2.putText(frame, text, (stats_x + 20, dist_y + 30 + i * 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['text'], 1)
        
        # Instructions
        cv2.putText(frame, "Press 'Q' to exit or 'R' to restart",
                   (panel_x + 130, panel_y + panel_height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['ui_secondary'], 1)
        
        return frame
    
    def draw_fps(self, frame: np.ndarray, fps: float) -> np.ndarray:
        """
        Draw FPS counter
        
        Args:
            frame: Video frame
            fps: Current FPS
            
        Returns:
            Frame with FPS overlay
        """
        h, w = frame.shape[:2]
        
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(frame, fps_text, (w - 150, h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS['ui_secondary'], 1)
        
        return frame
    
    def draw_controls_help(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw control instructions
        
        Args:
            frame: Video frame
            
        Returns:
            Frame with instructions
        """
        h, w = frame.shape[:2]
        
        controls = [
            "Controls:",
            "Q - Quit",
            "P - Pause",
            "R - Reset",
            "S - Settings",
            "V - Toggle Voice"
        ]
        
        help_x = 20
        help_y = h - 180
        
        for i, control in enumerate(controls):
            cv2.putText(frame, control, (help_x, help_y + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS['ui_secondary'], 1)
        
        return frame
    
    def reset(self):
        """Reset dashboard for new workout"""
        self.angle_history.clear()
        self.form_score_history.clear()
        self.time_history.clear()
        self.rep_durations = []
        self.rep_quality_grades = {'Perfect': 0, 'Good': 0, 'Fair': 0, 'Poor': 0}
        self.start_time = None
        self.current_rep_count = 0
        self.total_calories = 0
        self.perfect_streak = 0
        self.max_streak = 0



