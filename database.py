"""
Database Module
Handles SQLite database operations for workout history and user data
"""

import sqlite3
import json
import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import pandas as pd

from config import DATABASE_CONFIG


class WorkoutDatabase:
    """Manages workout data persistence and analytics"""
    
    def __init__(self, db_path: str = None):
        """
        Initialize database connection
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path or DATABASE_CONFIG['db_path']
        
        # Ensure data directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize database
        self._initialize_database()
    
    def _initialize_database(self):
        """Create database tables if they don't exist"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Create workouts table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS workouts (
                workout_id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATETIME NOT NULL,
                duration REAL NOT NULL,
                total_reps INTEGER NOT NULL,
                average_form_score REAL,
                total_calories REAL,
                workout_mode TEXT DEFAULT 'FREE',
                personal_records TEXT,
                notes TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create reps table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reps (
                rep_id INTEGER PRIMARY KEY AUTOINCREMENT,
                workout_id INTEGER NOT NULL,
                rep_number INTEGER NOT NULL,
                timestamp REAL NOT NULL,
                duration REAL NOT NULL,
                form_score REAL NOT NULL,
                quality_grade TEXT NOT NULL,
                quality_score REAL NOT NULL,
                max_angle REAL NOT NULL,
                min_angle REAL NOT NULL,
                mistakes TEXT,
                calories_burned REAL,
                FOREIGN KEY (workout_id) REFERENCES workouts(workout_id)
            )
        ''')
        
        # Create user_profile table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_profile (
                user_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                age INTEGER,
                weight REAL,
                fitness_level TEXT,
                total_pushups_lifetime INTEGER DEFAULT 0,
                best_single_session INTEGER DEFAULT 0,
                best_form_score REAL DEFAULT 0,
                current_streak_days INTEGER DEFAULT 0,
                last_workout_date DATE,
                achievements TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create indexes for better query performance
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_workouts_date 
            ON workouts(date)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_reps_workout 
            ON reps(workout_id)
        ''')
        
        conn.commit()
        conn.close()
        
        # Ensure user profile exists
        self._ensure_user_profile()
    
    def _get_connection(self) -> sqlite3.Connection:
        """Get database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn
    
    def _ensure_user_profile(self):
        """Ensure at least one user profile exists"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT COUNT(*) as count FROM user_profile')
        count = cursor.fetchone()['count']
        
        if count == 0:
            # Create default profile
            cursor.execute('''
                INSERT INTO user_profile (name, fitness_level, achievements)
                VALUES (?, ?, ?)
            ''', ('User', 'Beginner', json.dumps([])))
            conn.commit()
        
        conn.close()
    
    def save_workout(self, workout_data: Dict) -> int:
        """
        Save workout session to database
        
        Args:
            workout_data: Dictionary containing workout information
            
        Returns:
            workout_id of saved workout
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO workouts (
                date, duration, total_reps, average_form_score, 
                total_calories, workout_mode, personal_records, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            workout_data.get('date', datetime.now()),
            workout_data.get('duration', 0),
            workout_data.get('total_reps', 0),
            workout_data.get('average_form_score', 0),
            workout_data.get('total_calories', 0),
            workout_data.get('workout_mode', 'FREE'),
            json.dumps(workout_data.get('personal_records', {})),
            workout_data.get('notes', '')
        ))
        
        workout_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # Update user stats
        self.update_user_stats(workout_data)
        
        return workout_id
    
    def save_rep(self, workout_id: int, rep_data: Dict):
        """
        Save individual rep data
        
        Args:
            workout_id: ID of the workout this rep belongs to
            rep_data: Dictionary containing rep information
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO reps (
                workout_id, rep_number, timestamp, duration,
                form_score, quality_grade, quality_score,
                max_angle, min_angle, mistakes, calories_burned
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            workout_id,
            rep_data.get('rep_number', 0),
            rep_data.get('timestamp', 0),
            rep_data.get('duration', 0),
            rep_data.get('form_score', 0),
            rep_data.get('quality_grade', 'Poor'),
            rep_data.get('quality_score', 0),
            rep_data.get('max_angle', 180),
            rep_data.get('min_angle', 90),
            json.dumps(rep_data.get('mistakes', [])),
            rep_data.get('calories_burned', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def save_workout_with_reps(self, workout_data: Dict, reps_data: List[Dict]) -> int:
        """
        Save complete workout with all reps in one transaction
        
        Args:
            workout_data: Workout information
            reps_data: List of rep data dictionaries
            
        Returns:
            workout_id
        """
        workout_id = self.save_workout(workout_data)
        
        for rep_data in reps_data:
            self.save_rep(workout_id, rep_data)
        
        return workout_id
    
    def get_workout_history(self, days: int = 30) -> List[Dict]:
        """
        Get workout history for the last N days
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of workout dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT * FROM workouts
            WHERE date >= ?
            ORDER BY date DESC
        ''', (cutoff_date,))
        
        workouts = []
        for row in cursor.fetchall():
            workout = dict(row)
            # Parse JSON fields
            workout['personal_records'] = json.loads(workout.get('personal_records', '{}'))
            workouts.append(workout)
        
        conn.close()
        return workouts
    
    def get_workout_details(self, workout_id: int) -> Optional[Dict]:
        """
        Get detailed information about a specific workout
        
        Args:
            workout_id: Workout ID
            
        Returns:
            Workout dictionary with all details or None
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM workouts WHERE workout_id = ?', (workout_id,))
        row = cursor.fetchone()
        
        if not row:
            conn.close()
            return None
        
        workout = dict(row)
        workout['personal_records'] = json.loads(workout.get('personal_records', '{}'))
        
        # Get associated reps
        cursor.execute('''
            SELECT * FROM reps
            WHERE workout_id = ?
            ORDER BY rep_number
        ''', (workout_id,))
        
        reps = []
        for rep_row in cursor.fetchall():
            rep = dict(rep_row)
            rep['mistakes'] = json.loads(rep.get('mistakes', '[]'))
            reps.append(rep)
        
        workout['reps'] = reps
        
        conn.close()
        return workout
    
    def get_rep_details(self, workout_id: int) -> List[Dict]:
        """
        Get all rep details for a workout
        
        Args:
            workout_id: Workout ID
            
        Returns:
            List of rep dictionaries
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM reps
            WHERE workout_id = ?
            ORDER BY rep_number
        ''', (workout_id,))
        
        reps = []
        for row in cursor.fetchall():
            rep = dict(row)
            rep['mistakes'] = json.loads(rep.get('mistakes', '[]'))
            reps.append(rep)
        
        conn.close()
        return reps
    
    def update_user_stats(self, workout_data: Dict):
        """
        Update user profile statistics after a workout
        
        Args:
            workout_data: Completed workout data
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        # Get current stats
        cursor.execute('SELECT * FROM user_profile WHERE user_id = 1')
        profile = dict(cursor.fetchone())
        
        # Update stats
        total_reps = workout_data.get('total_reps', 0)
        avg_form_score = workout_data.get('average_form_score', 0)
        workout_date = workout_data.get('date', datetime.now())
        
        new_lifetime_total = profile['total_pushups_lifetime'] + total_reps
        new_best_session = max(profile['best_single_session'], total_reps)
        new_best_form = max(profile['best_form_score'], avg_form_score)
        
        # Calculate streak
        last_workout = profile.get('last_workout_date')
        if last_workout:
            last_date = datetime.strptime(last_workout, '%Y-%m-%d').date()
            current_date = workout_date.date() if isinstance(workout_date, datetime) else workout_date
            
            days_diff = (current_date - last_date).days
            
            if days_diff == 1:
                # Consecutive day
                new_streak = profile['current_streak_days'] + 1
            elif days_diff == 0:
                # Same day
                new_streak = profile['current_streak_days']
            else:
                # Streak broken
                new_streak = 1
        else:
            new_streak = 1
        
        # Update profile
        cursor.execute('''
            UPDATE user_profile
            SET total_pushups_lifetime = ?,
                best_single_session = ?,
                best_form_score = ?,
                current_streak_days = ?,
                last_workout_date = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE user_id = 1
        ''', (
            new_lifetime_total,
            new_best_session,
            new_best_form,
            new_streak,
            workout_date.date() if isinstance(workout_date, datetime) else workout_date
        ))
        
        conn.commit()
        conn.close()
    
    def get_user_profile(self) -> Dict:
        """Get user profile data"""
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM user_profile WHERE user_id = 1')
        row = cursor.fetchone()
        
        profile = dict(row)
        profile['achievements'] = json.loads(profile.get('achievements', '[]'))
        
        conn.close()
        return profile
    
    def add_achievement(self, achievement_id: str, achievement_name: str):
        """
        Add an achievement to user profile
        
        Args:
            achievement_id: Unique achievement identifier
            achievement_name: Display name of achievement
        """
        profile = self.get_user_profile()
        achievements = profile['achievements']
        
        # Check if already earned
        if achievement_id not in [a['id'] for a in achievements]:
            achievements.append({
                'id': achievement_id,
                'name': achievement_name,
                'earned_at': datetime.now().isoformat()
            })
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                UPDATE user_profile
                SET achievements = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE user_id = 1
            ''', (json.dumps(achievements),))
            
            conn.commit()
            conn.close()
            
            return True
        
        return False
    
    def get_statistics(self, days: int = 30) -> Dict:
        """
        Calculate various statistics
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary with statistics
        """
        workouts = self.get_workout_history(days)
        
        if not workouts:
            return {
                'total_workouts': 0,
                'total_reps': 0,
                'average_reps_per_workout': 0,
                'total_calories': 0,
                'average_form_score': 0,
                'best_workout': None,
                'improvement_trend': 0
            }
        
        total_workouts = len(workouts)
        total_reps = sum(w['total_reps'] for w in workouts)
        total_calories = sum(w['total_calories'] for w in workouts)
        avg_reps = total_reps / total_workouts if total_workouts > 0 else 0
        avg_form = sum(w['average_form_score'] for w in workouts) / total_workouts
        
        # Find best workout
        best_workout = max(workouts, key=lambda w: w['total_reps'])
        
        # Calculate improvement trend (compare first half vs second half)
        if len(workouts) >= 4:
            mid = len(workouts) // 2
            first_half_avg = sum(w['total_reps'] for w in workouts[mid:]) / mid
            second_half_avg = sum(w['total_reps'] for w in workouts[:mid]) / mid
            improvement = ((second_half_avg - first_half_avg) / first_half_avg * 100) if first_half_avg > 0 else 0
        else:
            improvement = 0
        
        stats = {
            'total_workouts': total_workouts,
            'total_reps': total_reps,
            'average_reps_per_workout': avg_reps,
            'total_calories': total_calories,
            'average_form_score': avg_form,
            'best_workout': best_workout,
            'improvement_trend': improvement
        }
        
        return stats
    
    def get_most_common_mistakes(self, days: int = 30, top_n: int = 5) -> List[Tuple[str, int]]:
        """
        Get most common form mistakes across all workouts
        
        Args:
            days: Number of days to analyze
            top_n: Number of top mistakes to return
            
        Returns:
            List of (mistake_type, count) tuples
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        cursor.execute('''
            SELECT r.mistakes
            FROM reps r
            JOIN workouts w ON r.workout_id = w.workout_id
            WHERE w.date >= ?
        ''', (cutoff_date,))
        
        mistake_counts = {}
        for row in cursor.fetchall():
            mistakes = json.loads(row['mistakes'])
            for mistake in mistakes:
                mistake_type = mistake.get('type', 'unknown')
                mistake_counts[mistake_type] = mistake_counts.get(mistake_type, 0) + 1
        
        conn.close()
        
        # Sort and return top N
        sorted_mistakes = sorted(mistake_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_mistakes[:top_n]
    
    def export_to_csv(self, output_path: str, days: int = 30):
        """
        Export workout data to CSV file
        
        Args:
            output_path: Path to save CSV file
            days: Number of days to export
        """
        workouts = self.get_workout_history(days)
        
        if not workouts:
            print("No workout data to export.")
            return
        
        # Convert to DataFrame
        df_data = []
        for workout in workouts:
            df_data.append({
                'Date': workout['date'],
                'Duration (min)': workout['duration'] / 60,
                'Total Reps': workout['total_reps'],
                'Avg Form Score': workout['average_form_score'],
                'Calories': workout['total_calories'],
                'Mode': workout['workout_mode']
            })
        
        df = pd.DataFrame(df_data)
        df.to_csv(output_path, index=False)
        print(f"Data exported to {output_path}")
    
    def backup_database(self, backup_path: str = None):
        """
        Create a backup of the database
        
        Args:
            backup_path: Path for backup file
        """
        if backup_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = f"data/backup_sessions_{timestamp}.db"
        
        import shutil
        shutil.copy2(self.db_path, backup_path)
        print(f"Database backed up to {backup_path}")
    
    def clear_old_data(self, days: int = 365):
        """
        Delete data older than specified days
        
        Args:
            days: Keep data newer than this many days
        """
        conn = self._get_connection()
        cursor = conn.cursor()
        
        cutoff_date = datetime.now() - timedelta(days=days)
        
        # Delete old workouts (reps will cascade if foreign key is set up)
        cursor.execute('DELETE FROM workouts WHERE date < ?', (cutoff_date,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        print(f"Deleted {deleted_count} old workout records.")
    
    def close(self):
        """Close database connection"""
        # Connection is opened and closed per operation
        pass





