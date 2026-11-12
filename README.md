# PoseClassifierFull

The pose classification library from my final university project (Universidad Nacional del Sur, 2021).
It detects a person's pose with MoveNet and classifies it in real time. It is light enough for a Raspberry Pi.

To use it, run `classifier_optimized.py` with at least the first 4 arguments in the table below.
Each model folder in `Models/` shows the classes that the model knows.

## Requirements

- Python 3
- `opencv-python` and `numpy`
- On a Raspberry Pi or other Linux device: `tflite-runtime`
- On Windows or a desktop: `tensorflow` (the code uses `tf.lite` when `tflite-runtime` is not available)

## Example

```bash
python classifier_optimized.py Models/13class_thunder_v2 False True False
```

This command uses the 13-class model, reads frames from the webcam, and shows the output in a window.

## Arguments

| Argument | Type | Description |
| ----------- | ----------- | ----------- |
| model_path | string | relative path to the classifier model |
| process_video | Bool | if True, gets frames from 'test_video.mp4', if False it gets frames from the webcam stream |
| show_output | Bool | shows the output in a window, for debugging (it decreases performance) |
| generate_video | Bool | if True, creates 'output.mp4' with the pose overlay and the predictions |
| fps | int | rate at which the detection algorithm reads data, default is 30 fps |
| pose_detection_threshold | float | threshold for a successful pose detection, default is 0.5 |
| pose_prediction_threshold | float | threshold for a successful pose classification, default is 0.8 |
