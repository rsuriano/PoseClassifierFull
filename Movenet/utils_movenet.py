import numpy as np
import cv2
import tensorflow as tf
import keras
#from matplotlib import pyplot as plt
#from matplotlib.collections import LineCollection
#
#import matplotlib.patches as patches


# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:

      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


# def draw_prediction_on_image(
#     image, keypoints_with_scores, crop_region=None, close_figure=False,
#     output_image_height=None):
#   """Draws the keypoint predictions on image.

#   Args:
#     image: A numpy array with shape [height, width, channel] representing the
#       pixel values of the input image.
#     keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
#       the keypoint coordinates and scores returned from the MoveNet model.
#     crop_region: A dictionary that defines the coordinates of the bounding box
#       of the crop region in normalized coordinates (see the init_crop_region
#       function below for more detail). If provided, this function will also
#       draw the bounding box on the image.
#     output_image_height: An integer indicating the height of the output image.
#       Note that the image aspect ratio will be the same as the input image.

#   Returns:
#     A numpy array with shape [out_height, out_width, channel] representing the
#     image overlaid with keypoint predictions.
#   """
#   height, width, channel = image.shape
#   aspect_ratio = float(width) / height
#   fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
#   # To remove the huge white borders
#   fig.tight_layout(pad=0)
#   ax.margins(0)
#   ax.set_yticklabels([])
#   ax.set_xticklabels([])
#   plt.axis('off')

#   im = ax.imshow(image)
#   line_segments = LineCollection([], linewidths=(4), linestyle='solid')
#   ax.add_collection(line_segments)
#   # Turn off tick labels
#   scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

#   (keypoint_locs, keypoint_edges,
#    edge_colors) = _keypoints_and_edges_for_display(
#        keypoints_with_scores, height, width)

#   line_segments.set_segments(keypoint_edges)
#   line_segments.set_color(edge_colors)
#   if keypoint_edges.shape[0]:
#     line_segments.set_segments(keypoint_edges)
#     line_segments.set_color(edge_colors)
#   if keypoint_locs.shape[0]:
#     scat.set_offsets(keypoint_locs)

#   if crop_region is not None:
#     xmin = max(crop_region['x_min'] * width, 0.0)
#     ymin = max(crop_region['y_min'] * height, 0.0)
#     rec_width = min(crop_region['x_max'], 0.99) * width - xmin
#     rec_height = min(crop_region['y_max'], 0.99) * height - ymin
#     rect = patches.Rectangle(
#         (xmin,ymin),rec_width,rec_height,
#         linewidth=1,edgecolor='b',facecolor='none')
#     ax.add_patch(rect)

#   fig.canvas.draw()
#   image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
#   image_from_plot = image_from_plot.reshape(
#       fig.canvas.get_width_height()[::-1] + (3,))
#   plt.close(fig)
#   if output_image_height is not None:
#     output_image_width = int(output_image_height / height * width)
#     image_from_plot = cv2.resize(
#         image_from_plot, dsize=(output_image_width, output_image_height),
#          interpolation=cv2.INTER_CUBIC)

#   return image_from_plot


# Define functions to convert the pose landmarks to a pose embedding (a.k.a. feature vector) for pose classification

def get_center_point(landmarks, left_name, right_name):
  """Calculates the center point of the two given landmarks."""
  left = tf.gather(landmarks, KEYPOINT_DICT[left_name], axis=1)
  right = tf.gather(landmarks, KEYPOINT_DICT[right_name], axis=1)
  center = left * 0.5 + right * 0.5

  return center


def get_pose_size(landmarks, torso_size_multiplier=2.5):
  """Calculates pose size.

  It is the maximum of two values:

    * Torso size multiplied by `torso_size_multiplier`
    * Maximum distance from pose center to any pose landmark
  """
  # Hips center
  hips_center = get_center_point(landmarks, "left_hip", "right_hip")

  # Shoulders center
  shoulders_center = get_center_point(landmarks, 
                                      "left_shoulder", "right_shoulder")
  # Torso size as the minimum body size
  torso_size = tf.linalg.norm(shoulders_center - hips_center, axis=1) # <== aca empezo el problema con la paralelizacion

  # Pose center
  pose_center_new = get_center_point(landmarks, "left_hip", "right_hip")
  pose_center_new = tf.expand_dims(pose_center_new, axis=1)

  # Broadcast the pose center to the same size as the landmark vector to
  # perform substraction
  pose_center_new = tf.broadcast_to(pose_center_new,
                                    [tf.size(landmarks) // (17*2), 17, 2])
  
  # Dist to pose center
  #d = tf.gather(landmarks - pose_center_new, 1, axis=1,
  #              name="dist_to_pose_center")
  d = landmarks - pose_center_new
  
  # Max dist to pose center
  max_dist = tf.reduce_max(tf.linalg.norm(d, axis=-1), axis=1)

  # Normalize scale
  pose_size = tf.maximum(torso_size * torso_size_multiplier, max_dist)
  return pose_size


def normalize_pose_landmarks(landmarks):
  """Normalizes the landmarks translation by moving the pose center to (0,0) and
  scaling it to a constant pose size.
  """

  # Move landmarks so that the pose center becomes (0,0)
  pose_center = get_center_point(landmarks, "left_hip", "right_hip")
  pose_center = tf.expand_dims(pose_center, axis=1)

  # Broadcast the pose center to the same size as the landmark vector to perform
  # substraction
  pose_center = tf.broadcast_to(pose_center, 
                                [tf.size(landmarks) // (17*2), 17, 2])
  landmarks = landmarks - pose_center

  # Scale the landmarks to a constant pose size
  pose_size = get_pose_size(landmarks)
  pose_size = tf.expand_dims( tf.expand_dims(pose_size, axis=1), axis=1)
  landmarks /= pose_size
  return landmarks


def landmarks_to_embedding(landmarks_unprocessed):
  """Converts the input landmarks into a pose embedding."""
  
  # Reshape the flat input into a matrix with shape=(17, 3)
  reshaped_inputs = np.expand_dims( np.reshape(landmarks_unprocessed, (17, 3)), axis=0 )

  # Normalize landmarks 2D
  landmarks = normalize_pose_landmarks(reshaped_inputs[:, :, :2])
  landmarks = landmarks[:, :13, :]

  # Flatten the normalized landmark coordinates into a vector
  embedding = keras.layers.Flatten()(landmarks)

  return embedding