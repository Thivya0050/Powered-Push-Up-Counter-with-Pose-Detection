# 🚀 Quick Start Guide

## ✅ Installation Complete!

All dependencies are installed and the application is ready to run!

## 🎯 How to Run

### Option 1: Windows (Easiest)
Double-click: **`RUN_ME.bat`**

### Option 2: Command Line
```bash
py main.py
```

### Option 3: With Goal Mode
```bash
py main.py GOAL 20
```
(Set a goal of 20 push-ups)

## 📹 Setup Your Camera

1. **Position yourself SIDE VIEW to the camera**
   - Stand or get in push-up position
   - Camera should see you from the side (profile view)
   - Ensure full body visible (head to feet)

2. **Lighting**
   - Use a well-lit room
   - Avoid backlighting (don't have window behind you)

3. **Distance**
   - Stay 6-8 feet from camera
   - Entire body should fit in frame

## ⌨️ Controls

Once the application starts:

| Key | Action |
|-----|--------|
| **S** | Start workout |
| **P** | Pause/Resume |
| **Q** | Quit (saves workout) |
| **V** | Toggle voice feedback |
| **G** | Toggle graphs |
| **K** | Toggle skeleton overlay |

## 🏋️ How to Get Counted

1. Press **'S'** to start
2. Do push-ups with **FULL range of motion**:
   - Start: Arms fully extended (straight)
   - Down: Lower until elbows reach 90 degrees
   - Hold briefly at bottom
   - Up: Push back to fully extended
3. Watch the real-time feedback on screen!

## 📊 What You'll See

- **Rep Count** - Large number showing current reps
- **Form Score** - Circular gauge (0-100)
- **Live Angle** - Current elbow angle
- **Skeleton Overlay** - Your body tracking
- **Form Issues** - Real-time corrections
- **Live Graphs** - Angle and form score over time

## 🎤 Voice Feedback

The app will announce:
- Rep counts at milestones (5, 10, 25, 50, 100)
- Form corrections
- Perfect form achievements
- Workout completion

Toggle voice on/off with **'V'** key

## 💾 Your Data

All workouts are automatically saved to `data/sessions.db`

View your progress anytime!

## ⚠️ Troubleshooting

### Camera not working?
- Check camera permissions in Windows settings
- Close other apps using the camera (Zoom, Teams, etc.)
- Try changing camera index in `config.py`

### Pose not detected?
- Ensure full body is visible
- Improve lighting
- Wear form-fitting clothes
- Use plain background

### Slow/Laggy?
- Press **'G'** to turn off graphs
- Close other applications
- Lower resolution in `config.py`

## 🏆 Tips for Best Results

1. **Side view is critical** - Camera must see you from the side
2. **Full range of motion** - Go all the way down to 90° elbows
3. **Control your tempo** - Don't rush! 1-2 seconds down, 1-2 seconds up
4. **Keep form** - Straight body line (plank position)
5. **Quality over quantity** - Focus on perfect form reps

---

## Ready? Let's Go! 💪

```bash
py main.py
```

Press **'S'** to start and begin your workout!



