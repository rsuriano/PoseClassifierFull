# PoseClassifierFull

This is a simple pose classifier intended to be used in real time in devices such as the Raspberry Pi. To use it, run `classifier_full.py` with at least the first 4 parameters described in the table below. 
Classes are illustrated inside each model folder.

Requirements: TBA

| Argument | Type | Description |
| ----------- | ----------- | ----------- |
| model_path | string | relative path to the classifier model |
| process_video | Bool | if True, gets frames from 'test_video.mp4', if False it get frames from webcam stream |
| show_output | Bool | shows output in a window, for debugging as decreases performance |
| generate_video | Bool | if True, creates 'output.mp4' with pose overlay and predictions |
| fps | int | rate at which the detection algorithm reads data,   default is 30 fps |
| pose_detection_threshold | int | threshold for successful pose detection, default is 0.5 |
| pose_prediction_threshold | int | threshold for successful pose classification, default is 0.8 |
