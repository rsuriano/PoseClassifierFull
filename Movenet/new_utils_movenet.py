import os, sys
import numpy as np
import cv2

# MoveNet Imports
from . import utils_movenet as utils
from .movenet import Movenet

FILEDIR = os.path.dirname(os.path.abspath(__file__))
movenet = Movenet(f'{FILEDIR}/movenet_thunder.tflite')

# Define function to run pose estimation using MoveNet Thunder.
# You'll apply MoveNet's cropping algorithm and run inference multiple times on
# the input image to improve pose estimation accuracy.
def detect(input_tensor, inference_count=3):

  # Detect pose using the full input image
  movenet.detect(input_tensor, reset_crop_region=True)

  # Repeatedly using previous detection result to identify the region of
  # interest and only croping that region to improve detection accuracy
  for _ in range(inference_count - 1):
    keypoint_with_scores = movenet.detect(input_tensor, reset_crop_region=False)

  return keypoint_with_scores

def draw_pose_on_image(image, keypoints, pose_threshold=0):
    size = image.shape[:2]

    (keypoint_locs, 
    keypoint_edges, 
    edge_colors) = utils._keypoints_and_edges_for_display(np.expand_dims(keypoints, (0,1)), size[0], size[1])

    keypoint_locs = keypoint_locs.astype(int)
    keypoint_edges = keypoint_edges.astype(int)

    overlay = cv2.cvtColor(image.copy(), cv2.COLOR_RGB2BGR)
    if np.average(keypoints[:11, 2]) > pose_threshold:
        # draw edges
        for i, edge in enumerate(keypoint_edges):
            
            if i < 13:
                try:
                    edge_color = edge_colors[i]
                except:
                    pass
                if edge_color=='m':     edge_color = [118, 52, 207]
                if edge_color=='c':     edge_color = [255, 255, 0]
                if edge_color=='y':     edge_color = [0, 255, 255]
                cv2.line(overlay, edge[0], edge[1], edge_color, thickness=int(size[0]/80))
        
        # draw points
        for keypoint in keypoint_locs:
            cv2.circle(overlay, keypoint, int(size[0]/90), color=[0, 255, 0], thickness=-1)
    
    return overlay, keypoint_locs