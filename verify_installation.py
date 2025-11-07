"""
Verification script to ensure everything is set up correctly
"""

print("\n" + "="*60)
print("  VERIFYING AI PUSH-UP COUNTER INSTALLATION")
print("="*60 + "\n")

all_good = True

# Test 1: Python dependencies
print("1. Testing Python dependencies...")
dependencies = {
    'opencv-python': 'cv2',
    'mediapipe': 'mediapipe',
    'numpy': 'numpy',
    'pyttsx3': 'pyttsx3',
    'matplotlib': 'matplotlib',
    'pandas': 'pandas',
    'scipy': 'scipy',
    'plotly': 'plotly',
    'Pillow': 'PIL'
}

for package, import_name in dependencies.items():
    try:
        __import__(import_name)
        print(f"   ✓ {package}")
    except ImportError:
        print(f"   ✗ {package} - MISSING!")
        all_good = False

# Test 2: Custom modules
print("\n2. Testing custom modules...")
modules = [
    'config',
    'pose_detector',
    'form_analyzer',
    'audio_feedback',
    'analytics_dashboard',
    'database',
    'main'
]

for module in modules:
    try:
        __import__(module)
        print(f"   ✓ {module}.py")
    except Exception as e:
        print(f"   ✗ {module}.py - ERROR: {str(e)[:50]}")
        all_good = False

# Test 3: Directory structure
print("\n3. Testing directory structure...")
import os

directories = ['data', 'assets', 'logs']
for directory in directories:
    if os.path.exists(directory):
        print(f"   ✓ {directory}/ exists")
    else:
        print(f"   ✗ {directory}/ missing")
        all_good = False

# Test 4: Database initialization
print("\n4. Testing database...")
try:
    from database import WorkoutDatabase
    db = WorkoutDatabase()
    profile = db.get_user_profile()
    print(f"   ✓ Database initialized")
    print(f"   ✓ User profile created")
except Exception as e:
    print(f"   ✗ Database error: {str(e)[:50]}")
    all_good = False

# Test 5: MediaPipe Pose
print("\n5. Testing MediaPipe Pose detection...")
try:
    import mediapipe as mp
    pose = mp.solutions.pose.Pose()
    print(f"   ✓ MediaPipe Pose initialized")
    pose.close()
except Exception as e:
    print(f"   ✗ MediaPipe error: {str(e)[:50]}")
    all_good = False

# Summary
print("\n" + "="*60)
if all_good:
    print("  ✓ ALL CHECKS PASSED!")
    print("="*60)
    print("\n  Your AI Push-Up Counter is ready to use!")
    print("\n  TO START:")
    print("    - Windows: Double-click 'RUN_ME.bat'")
    print("    - Or run: py main.py")
    print("\n  REQUIREMENTS:")
    print("    - Webcam connected and working")
    print("    - Full body visible from side view")
    print("    - Good lighting conditions")
    print("\n" + "="*60 + "\n")
else:
    print("  ✗ SOME CHECKS FAILED")
    print("="*60)
    print("\n  Please install missing dependencies:")
    print("    py -m pip install -r requirements.txt")
    print("\n" + "="*60 + "\n")



