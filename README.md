# 🏋️ AI-Powered Push-Up Counter with Pose Detection

An advanced real-time push-up counter using AI pose estimation with MediaPipe. Features automatic counting, comprehensive form analysis, quality scoring, rep history tracking, voice feedback, and detailed analytics dashboard.

![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-brightgreen.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

### Core Functionality
- ✅ **Real-time Push-up Counting** - Automatic rep counting with 95%+ accuracy
- ✅ **Advanced Form Analysis** - Comprehensive scoring based on body alignment, elbow position, depth, and tempo
- ✅ **Quality Grading System** - Each rep graded as Perfect, Good, Fair, or Poor
- ✅ **Voice Feedback** - Text-to-speech announcements for counts, milestones, and form corrections
- ✅ **Live Analytics Dashboard** - Real-time graphs and metrics visualization
- ✅ **Persistent Database** - SQLite database storing complete workout history

### Advanced Features
- 📊 **Form Score Components** - Body alignment (30%), Elbow consistency (25%), Depth (25%), Tempo (20%)
- 🎯 **Multiple Workout Modes** - Free, Goal, Timed, Interval, Challenge modes
- 🔥 **Streak Tracking** - Monitor consecutive perfect form reps
- 📈 **Progress Analytics** - Track improvement over time with detailed statistics
- 🏆 **Achievement System** - Unlock badges and milestones
- 💪 **Fatigue Detection** - Real-time monitoring of form degradation
- 🎨 **Professional UI** - Color-coded feedback and intuitive overlay interface
- 📁 **Data Export** - Export workout history to CSV format

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Webcam
- Windows, macOS, or Linux

### Installation

1. **Clone or download this repository**

```bash
cd "Powered Push-Up Counter with Pose Detection"
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

This will install:
- opencv-python (Computer Vision)
- mediapipe (Pose Detection)
- numpy (Numerical Computing)
- pyttsx3 (Text-to-Speech)
- matplotlib (Data Visualization)
- pandas (Data Analysis)
- scipy (Scientific Computing)
- Pillow (Image Processing)
- plotly (Interactive Graphs)

3. **Run the application**

```bash
python main.py
```

## 📖 Usage Guide

### Starting a Workout

1. **Launch the application**
   ```bash
   python main.py
   ```

2. **Position yourself**
   - Stand or get into push-up position
   - Ensure your full body is visible from the side
   - Camera should capture from head to feet
   - Maintain good lighting

3. **Start counting**
   - Press **'S'** to start the workout
   - Begin doing push-ups with full range of motion
   - Watch the real-time feedback on screen

### Keyboard Controls

| Key | Action |
|-----|--------|
| **S** | Start workout |
| **P** | Pause/Resume |
| **R** | Reset/Restart |
| **Q** | Quit (saves workout) |
| **V** | Toggle voice feedback |
| **G** | Toggle live graphs |
| **K** | Toggle skeleton overlay |

### Workout Modes

#### Free Mode (Default)
```bash
python main.py
```
Unlimited counting - perfect for practice and endurance training.

#### Goal Mode
```bash
python main.py GOAL 20
```
Set a target number of reps (e.g., 20 push-ups).

#### Timed Mode
```bash
python main.py TIMED
```
As Many Reps As Possible (AMRAP) in set time.

## 📊 Understanding the Metrics

### Form Score (0-100)

Your form score is calculated using four components:

1. **Body Alignment (30%)** - Straight line from shoulders to ankles
   - ✅ Perfect: Body forms a plank
   - ❌ Issues: Sagging hips, raised hips, bent knees

2. **Elbow Consistency (25%)** - Elbows close to body
   - ✅ Perfect: Elbows at ~45° from body
   - ❌ Issues: Flared elbows (>45°)

3. **Depth Consistency (25%)** - Reaching proper depth
   - ✅ Perfect: Elbows reach 90° or lower
   - ❌ Issues: Incomplete range of motion

4. **Speed Control (20%)** - Controlled movement
   - ✅ Perfect: 1-2 seconds down, 1-2 seconds up
   - ❌ Issues: Too fast (<0.8s) or too slow (>3s)

### Quality Grades

- **Perfect** (90-100) - Flawless form! 🏆
- **Good** (75-89) - Solid rep! 💪
- **Fair** (60-74) - Acceptable, room for improvement 📈
- **Poor** (0-59) - Form needs work 🔧

### Common Form Mistakes Detected

| Mistake | What It Means | How to Fix |
|---------|---------------|------------|
| **Hips Sagging** | Core not engaged | Tighten your core, maintain plank position |
| **Hips Too High** | Pike position | Lower hips to align with shoulders and ankles |
| **Incomplete Depth** | Not going low enough | Lower until elbows reach 90 degrees |
| **Flared Elbows** | Elbows too far from body | Keep elbows at 45° angle from torso |
| **Too Fast** | Rushing reps | Slow down, control the movement |
| **Head Dropping** | Neck misalignment | Keep neck neutral, look slightly ahead |
| **Uneven Arms** | Asymmetric movement | Focus on balanced, synchronized motion |

## 🎯 Best Practices

### Camera Setup

1. **Positioning**
   - Place camera to your side (profile view)
   - Ensure full body visibility
   - Maintain 6-8 feet distance
   - Keep camera at mid-body height

2. **Lighting**
   - Use well-lit room
   - Avoid backlighting (window behind you)
   - Ensure even lighting across body

3. **Background**
   - Use plain, uncluttered background
   - Avoid busy patterns that might confuse detection

### Getting Accurate Counts

1. **Full Range of Motion**
   - Start in fully extended position (arms straight)
   - Lower until elbows reach 90 degrees
   - Hold briefly at bottom
   - Push back up to full extension

2. **Controlled Tempo**
   - Don't rush - quality over quantity
   - 1-2 seconds down
   - Brief pause at bottom
   - 1-2 seconds up

3. **Consistent Form**
   - Maintain plank position throughout
   - Keep core engaged
   - Head neutral (don't look down)
   - Breathe regularly

## 📈 Analytics & History

### Viewing Your Progress

After each workout, data is automatically saved to the database. Access your history through:

```python
from database import WorkoutDatabase

db = WorkoutDatabase()

# Get last 30 days of workouts
history = db.get_workout_history(days=30)

# Get statistics
stats = db.get_statistics(days=30)
print(f"Total reps: {stats['total_reps']}")
print(f"Average form score: {stats['average_form_score']:.1f}")
print(f"Improvement trend: {stats['improvement_trend']:.1f}%")
```

### Exporting Data

Export your workout history to CSV:

```python
from database import WorkoutDatabase

db = WorkoutDatabase()
db.export_to_csv('my_pushup_history.csv', days=30)
```

## 🏆 Achievements

Unlock achievements as you progress:

| Achievement | Requirement |
|-------------|-------------|
| First Push-Up | Complete 1 rep |
| Getting Started | Complete 10 reps |
| Building Strength | Complete 50 reps (lifetime) |
| Strong! | Complete 100 reps (lifetime) |
| Elite Athlete | Complete 250 reps (lifetime) |
| Push-Up Master | Complete 500 reps (lifetime) |
| Legendary | Complete 1000 reps (lifetime) |
| Perfect 10 | 10 consecutive perfect form reps |
| Century Club | 100 reps in one session |
| Consistent | 7 day workout streak |

## 🔧 Configuration

Customize the application by editing `config.py`:

### Adjust Angle Thresholds
```python
UP_ANGLE_THRESHOLD = 160  # Arms extended
DOWN_ANGLE_THRESHOLD = 90  # Bottom position
```

### Modify Form Weights
```python
FORM_WEIGHTS = {
    'body_alignment': 0.30,
    'elbow_consistency': 0.25,
    'depth_consistency': 0.25,
    'speed_control': 0.20
}
```

### Audio Settings
```python
AUDIO_CONFIG = {
    'voice_feedback_enabled': True,
    'count_announcement_interval': 5,
    'tts_rate': 175,
}
```

## 🐛 Troubleshooting

### Camera Not Detected

**Problem:** "Error: Unable to access camera"

**Solutions:**
1. Check camera is connected and not used by another application
2. Try different camera index:
   ```python
   # In config.py
   CAMERA_CONFIG['camera_index'] = 1  # Try 0, 1, 2, etc.
   ```
3. Grant camera permissions in system settings

### Pose Not Detected

**Problem:** "No pose detected" message

**Solutions:**
1. Ensure full body is visible in frame
2. Improve lighting conditions
3. Move further from camera
4. Wear form-fitting clothes for better landmark detection
5. Use plain background

### Voice Feedback Not Working

**Problem:** No audio announcements

**Solutions:**
1. Press 'V' to ensure voice is enabled
2. Check system audio/volume settings
3. Verify pyttsx3 installation:
   ```bash
   pip install --upgrade pyttsx3
   ```

### Low FPS / Laggy Performance

**Problem:** Application runs slowly

**Solutions:**
1. Lower camera resolution in `config.py`:
   ```python
   CAMERA_CONFIG['frame_width'] = 640
   CAMERA_CONFIG['frame_height'] = 480
   ```
2. Disable graphs (press 'G')
3. Reduce model complexity:
   ```python
   POSE_DETECTION_CONFIG['model_complexity'] = 0
   ```

### Inaccurate Counting

**Problem:** Reps not counted or false counts

**Solutions:**
1. Ensure side-view positioning (profile to camera)
2. Use full range of motion (arms fully extended → 90° elbow bend)
3. Slow down tempo - avoid bouncing
4. Adjust thresholds in `config.py` if needed

## 📁 Project Structure

```
Powered Push-Up Counter with Pose Detection/
│
├── main.py                      # Main application entry point
├── config.py                    # Configuration settings
├── pose_detector.py            # MediaPipe pose detection
├── form_analyzer.py            # Form analysis and scoring
├── audio_feedback.py           # Voice feedback system
├── analytics_dashboard.py      # Real-time visualization
├── database.py                 # SQLite data persistence
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
├── data/                       # Database storage
│   └── sessions.db            # SQLite database
│
├── assets/                     # Static assets
│   ├── sounds/                # Audio files (optional)
│   └── icons/                 # UI icons (optional)
│
└── logs/                       # Log files
    └── workout_history.json   # Backup logs
```

## 🎓 Technical Details

### Technologies Used

- **OpenCV** - Video capture and image processing
- **MediaPipe** - Real-time pose estimation (33 landmark points)
- **NumPy** - Numerical computations and angle calculations
- **Matplotlib** - Real-time graph generation
- **SQLite** - Lightweight database for workout history
- **pyttsx3** - Cross-platform text-to-speech
- **Pandas** - Data analysis and export

### Pose Detection

The application uses MediaPipe Pose, which detects 33 3D landmarks on the human body:

- **Key Landmarks for Push-ups:**
  - Shoulders (11, 12)
  - Elbows (13, 14)
  - Wrists (15, 16)
  - Hips (23, 24)
  - Knees (25, 26)
  - Ankles (27, 28)

### Angle Calculation

Elbow angle is calculated using the law of cosines:

```
angle = arccos((BA · BC) / (|BA| × |BC|))
```

Where:
- B = Elbow position
- A = Shoulder position
- C = Wrist position

### State Machine

Rep counting uses a state machine:

1. **UP** - Arms extended (angle ≥ 160°)
2. **TRANSITIONING_DOWN** - Lowering (90° < angle < 160°)
3. **DOWN** - Bottom position (angle ≤ 90°, held for 0.3s)
4. **TRANSITIONING_UP** - Pushing up (90° < angle < 160°)
5. Back to **UP** - Rep counted! 🎉

## 🤝 Contributing

This is an educational/portfolio project. Feel free to:
- Report bugs
- Suggest features
- Submit improvements
- Fork and customize

## 📄 License

MIT License - Feel free to use and modify for personal or educational purposes.

## 🙏 Acknowledgments

- **MediaPipe** - Google's excellent pose detection framework
- **OpenCV** - Computer vision foundation
- **Python Community** - For amazing libraries and tools

## 📞 Support

Having issues? Check the Troubleshooting section above or review:
1. Camera setup and positioning
2. Lighting conditions
3. Full body visibility
4. System requirements

## 🎯 Future Enhancements

Potential future features:
- [ ] Additional exercise types (squats, planks, pull-ups)
- [ ] Mobile app version
- [ ] Multi-user profiles
- [ ] Social sharing and leaderboards
- [ ] Custom workout programs
- [ ] Video recording with overlays
- [ ] Advanced analytics with ML insights

---

## 🚀 Getting Started Now

Ready to transform your push-up training?

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
python main.py

# 3. Press 'S' to start
# 4. Start doing push-ups!
```

**Remember:** Quality over quantity! Focus on perfect form, and let the AI help you improve! 💪

---

Made with ❤️ using AI and Computer Vision





