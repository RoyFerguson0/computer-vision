# Computer Vision Projects (Python + MediaPipe)

This repository is a collection of real-time computer vision mini-projects built with Python, OpenCV, and MediaPipe.

The goal is simple: detect hands, faces, and body pose from a webcam, then use that data for practical demos like finger counting, volume control, and sign-language action recognition.

## What This Project Includes

- Hand tracking and hand landmark detection
- Face detection and face mesh tracking
- Pose estimation
- Finger counting demo
- Hand gesture volume control demo
- Sign-language/action recognition training and inference files

## Tech Stack

- Python 3.11
- OpenCV
- MediaPipe
- NumPy
- TensorFlow
- scikit-learn

## Project Layout

- `HandTrackingModule.py`, `FaceDetectionModule.py`, `FaceMeshModule.py`, `PoseModule.py`: reusable detection modules
- `*Project.py` and `*Basics.py` files: runnable demo scripts
- `AiTrainerProject.py` and `Sign-Language.py`: model training and sign-language workflow
- `MP_DATA/`: saved training sequence data
- `images/` and `videos/`: assets used by demos

## Quick Start

1. Clone the repo
2. Install dependencies
3. Run any demo script

```bash
pip install -r requirements.txt
python HandTrackingProject.py
```

Other examples:

```bash
python FaceDetectionBasics.py
python FaceMeshBasics.py
python PoseProject.py
python FingerCountingProject.py
python VolumeHandControl.py
```

Or use UV

```bash
uv sync
uv run HandTrackingProject.py
```

Same for Other examples

## Notes

- Most scripts use your webcam.
- If one script does not run on your machine immediately, check webcam permissions and Python package versions first.

## Status

Active learning/project repository with multiple independent experiments.
