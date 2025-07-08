# Player Tracking and Re-identification System

A computer vision system for tracking and re-identifying players in sports videos using YOLO (You Only Look Once) object detection and ByteTrack tracking algorithm.

## Overview

This project implements a player tracking system that can:
- Detect players in video footage
- Track players across frames using ByteTrack
- Maintain consistent player identities throughout the video
- Output annotated video with tracking information

## Requirements

### System Requirements
- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- Minimum 8GB RAM
- Storage space for video files and model weights

### Dependencies

Create a `requirements.txt` file with the following dependencies:

```
ultralytics>=8.0.0
opencv-python>=4.5.0
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.21.0
Pillow>=8.3.0
PyYAML>=6.0
```

## Installation

### Option 1: Using pip

1. Clone or download the project files
2. Navigate to the project directory
3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Option 2: Using Conda

1. Create a new conda environment:

```bash
conda create -n player_tracking python=3.9
conda activate player_tracking
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Setup

### 1. Model Setup

You need a trained YOLO model file (`best.pt`). This should be:
- A YOLOv8 model trained on player detection
- Placed in the same directory as `playertrack.py`
- If you don't have a custom model, you can use a pre-trained YOLO model:

```python
from ultralytics import YOLO
model = YOLO('yolov11n.pt')  # Downloads pre-trained model
```

### 2. Video Input

- Place your input video file in the project directory
- Supported formats: MP4, AVI, MOV, MKV

### 3. Configuration

Edit the parameters in `playertrack.py` as needed:

```python
input_video = '15sec_input_720p.mp4'  # Your input video file
yolo_model = 'best.pt'          # Your trained model file
```

## Running the Code

### Basic Usage

```bash
python playertrack.py
```

### Advanced Usage

You can modify the tracking parameters in the code:

```python
results = model.track(
    source=video_path,
    tracker="bytetrack.yaml",  # Tracking algorithm
    conf=0.25,                 # Confidence threshold
    iou=0.7,                   # IoU threshold
    persist=True,              # Persist tracks between frames
    save=True,                 # Save output video
    show=True                  # Display real-time tracking
)
```

## Output

The system will:
1. Display real-time tracking in a window
2. Save the annotated video to `runs/detect/track/` directory
3. Print tracking completion status

## Troubleshooting

### Common Issues

1. **"Video file not found"**
   - Check that the video file exists in the specified path
   - Verify the filename and extension

2. **"Model file not found"**
   - Ensure the model file (`best.pt`) is in the correct location
   - Check file permissions

3. **CUDA out of memory**
   - Reduce video resolution
   - Process shorter video segments
   - Use CPU mode by setting `device='cpu'`

4. **Slow processing**
   - Ensure GPU is available and CUDA is properly installed
   - Check `torch.cuda.is_available()` returns `True`

### Performance Tips

- Use GPU acceleration when available
- Reduce video resolution for faster processing
- Adjust confidence threshold based on your use case
- Use appropriate model size (nano, small, medium, large, extra-large)

## File Structure

```
project_directory/
├── playertrack.py          # Main tracking script
├── requirements.txt        # Dependencies
├── README.md              # This file
├── best.pt                # Trained YOLO model
├── 15sec_input_720p.mp4       # Input video file
└── runs/
    └── detect/
        └── track/         # Output directory
```

## Configuration Options

### Tracking Parameters

- `conf`: Confidence threshold (0.0-1.0)
- `iou`: Intersection over Union threshold
- `tracker`: Tracking algorithm ("bytetrack.yaml" or "botsort.yaml")
- `persist`: Maintain track IDs across frames
- `save`: Save output video
- `show`: Display real-time tracking

### Model Selection

Choose appropriate YOLO model based on your needs:
- `yolov8n.pt`: Fastest, least accurate
- `yolov8s.pt`: Balanced speed/accuracy
- `yolov8m.pt`: Medium accuracy
- `yolov8l.pt`: High accuracy
- `yolov8x.pt`: Highest accuracy, slowest

## Support

For issues related to:
- **Ultralytics YOLO**: Check [official documentation](https://docs.ultralytics.com/)
- **ByteTrack**: Refer to the [ByteTrack repository](https://github.com/ifzhang/ByteTrack)
- **OpenCV**: Visit [OpenCV documentation](https://opencv.org/)

## License

This project uses open-source libraries. Check individual library licenses for commercial use.
