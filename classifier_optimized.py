# Library imports
import os
import sys
import platform
import time 
import numpy as np
import cv2

FILEDIR = os.path.dirname(os.path.abspath(__file__))
OS = platform.system()
if OS=='Linux':
    import tflite_runtime.interpreter as tflite
if OS=='Windows':
    import tensorflow as tf
else:
    print('Not recognized Operative System.')
    sys.exit()

from Movenet.new_utils_movenet import detect, draw_pose_on_image
from videoStream import CamVideoStream

# Utility functions
class Classifier():

    def __init__(self, model_path, labels_path):
        #Load the TFLite model and allocate tensors
        if OS=='Linux':
            self.interpreter = tflite.Interpreter(model_path=model_path)  # if os is linux
        elif OS=='Windows':
            self.interpreter = tf.lite.Interpreter(model_path=model_path) # if os is windows
        else:
            print('Not recognized Operative System.')
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.labels_path = labels_path

    def predictPose(self, keypoint_with_scores):
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        input_data = np.expand_dims(keypoint_with_scores.flatten(), axis=0).astype('float32')
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)

        # Run inference.
        self.interpreter.invoke()

        # Post-processing: remove batch dimension and find the class with highest
        # probability.
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_label = np.argmax(output_data[0])

        with open(self.labels_path) as f:
            lines = [str.strip(line) for line in f.readlines()]
            
        return lines[predicted_label], output_data[0]


# Main prediction script
pose_detection_threshold = 0.3
pose_prediction_threshold = 0.7
model_name = 'pose_classifier'
labels_name = 'pose_labels'
skip_frames = 1

# Command line data:
#   model_path                  string  relative path to the classifier model
#   process_video               Bool    if True, gets frames from 'test_video.mp4', if False it get frames from webcam stream
#   show_output                 Bool    shows output in a window, for debugging as decreases performance       
#   generate_video              Bool    if True, creates 'output.mp4' with pose overlay and predictions
#   fps                         int     rate at which the detection algorithm reads data,   default is 30 fps
#   pose_detection_threshold    int     threshold for successful pose detection,            default is 0.5
#   pose_prediction_threshold   int     threshold for successful pose classification,       default is 0.8
if len(sys.argv) > 1:
    model_path      =   sys.argv[1]
    process_video   =   sys.argv[2].lower() == 'true'
    show_output     =   sys.argv[3].lower() == 'true'
    generate_video  =   sys.argv[4].lower() == 'true'
    if generate_video:
        fps = 30
    if len(sys.argv) > 5:
        fps = int(sys.argv[5])
        pose_detection_threshold = int(sys.argv[6])
        pose_prediction_threshold = int(sys.argv[7])
else:
    model_path = 'Models/8class_thunder_v1'
    process_video = False
    show_output = True
    generate_video = False

print(f'\n\nVISION SETTINGS:\n\tmodel path: {FILEDIR}/{model_path}\n\tprocess_video: {process_video}\n\tshow output: {show_output}\n\tgenerate video: {generate_video}\n')

if process_video:
    cap = CamVideoStream('test_video.mp4', live_mode=False)
    if generate_video:
        size = 864 , 1536
        fps = 30
        skip_frames = 30 // fps
        out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (size[1], size[0]), True)
else:
    cap = CamVideoStream(0)


# Classifier initialization
classifier = Classifier(
                    f'{FILEDIR}/{model_path}/{model_name}.tflite', 
                    f'{FILEDIR}/{model_path}/{labels_name}.txt')

i = 0
cap.start()
while(True):
    frame = cap.read()
    if frame is not None:

        start_time = time.time()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if not i%skip_frames:
            keypoint_with_scores = detect(frame)

            min_landmark_score = np.mean(keypoint_with_scores[0:13, 2])
            if min_landmark_score < pose_detection_threshold:
                outputLabel = ('Deteccion de pose imprecisa.')
                outputVector = 0
                confidence = "-" 
                color = (255, 0, 0)
            else:
                output = classifier.predictPose(keypoint_with_scores)
                outputLabel = output[0]
                outputVector = output[1]
                confidence = np.round(np.amax(outputVector)*100, 0)
                color = (0, 255, 0) if confidence>60 else (0, 0, 255)
            
            FPS = 1.0 / (time.time() - start_time)
            if show_output:
                outputImage, _ = draw_pose_on_image(frame, keypoint_with_scores, pose_threshold=pose_detection_threshold)
                cv2.putText(outputImage, f'PREDICCION: {outputLabel}', (10,40), cv2.FONT_HERSHEY_PLAIN, 2.5, color, 2)
                cv2.putText(outputImage, f'Confianza: {confidence}%', (10,70), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
                FPS = 1.0 / (time.time() - start_time)
                cv2.putText(outputImage, f'FPS: {FPS}%', (10,100), cv2.FONT_HERSHEY_PLAIN, 1.5, color, 2)
                cv2.imshow('Frame', outputImage)
            else:
                print(f'{np.round(FPS,2)}FPS\t|\tPREDICCION: {outputLabel}\tConfianza: {confidence}%')
            
            if generate_video: 
                out.write(outputImage)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        i+=1
    else:
        break

if generate_video: out.release()
cap.stop()
cv2.destroyAllWindows()