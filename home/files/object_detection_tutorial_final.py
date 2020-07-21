
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# In[ ]:


# import os
# import re


# In[ ]:


# (os.getcwd())


# In[ ]:


# os.chdir(".\protos")


# In[ ]:


# files = os.listdir()


# In[ ]:


# finalList = []
# for file in files:
#     if re.search(".proto",file):
#         finalList.append(file)


# In[ ]:


# finalList[2]


# In[1]:


# for file in finalList:
#     print(file)
#     try:
#         !protoc --python_out=. {file}
#     except:
#         print("error",file)


# # Imports

# In[1]:


import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile


# In[2]:


from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image


# In[3]:


# !pip install protobuf --upgrade


# In[4]:




from collections import defaultdict


# if tf.__version__ < '1.4.0':
#   raise ImportError('Please upgrade your tensorflow installation to v1.4.* or later!')


# In[5]:


tf.__version__ 


# In[6]:


# tf.test.gpu_device_name()


# In[7]:


# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# In[8]:


# This is needed to display the images.
# get_ipython().magic('matplotlib inline')


# ## Env setup

# In[9]:



# # This is needed since the notebook is stored in the object_detection folder.
# sys.path.append("..")


# In[10]:


# !set PYTHONPATH=C:\tensorflow1\models;C:\tensorflow1\models\research;C:\tensorflow1\models\research\slim


# ## Object detection imports
# Here are the imports from the object detection module.

# In[11]:


from . import label_map_util

from . import visualization_utils as vis_util


# # Model preparation 

# ## Variables
# 
# Any model exported using the `export_inference_graph.py` tool can be loaded here simply by changing `PATH_TO_CKPT` to point to a new .pb file.  
# 
# By default we use an "SSD with Mobilenet" model here. See the [detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.

# In[12]:


# What model to download.
MODEL_NAME = 'home/files/ssd_mobilenet_v1_coco_2017_11_17'
MODEL_FILE = MODEL_NAME + '.tar.gz'
DOWNLOAD_BASE = 'http://download.tensorflow.org/models/object_detection/'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# pb_path = "inference_graph"
# PATH_TO_CKPT = pb_path + '/frozen_inference_graph.pb'


# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

PATH_TO_LABELS = os.path.join(MODEL_NAME, 'mscoco_label_map.pbtxt')


NUM_CLASSES = 90


# ## Download Model

# In[13]:


# opener = urllib.request.URLopener()
# opener.retrieve(DOWNLOAD_BASE + MODEL_FILE, MODEL_FILE)
# tar_file = tarfile.open(MODEL_FILE)
# for file in tar_file.getmembers():
#   file_name = os.path.basename(file.name)
#   if 'frozen_inference_graph.pb' in file_name:
#     tar_file.extract(file, os.getcwd())


# ## Load a (frozen) Tensorflow model into memory.

# In[14]:


detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

# In[15]:


label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# In[16]:


category_index


# ## Helper code

# In[17]:


def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


# # Detection

# In[18]:


import glob


# In[19]:

# For the sake of simplicity we will use only 2 images:
# image1.jpg
# image2.jpg
# If you want to test the code with your images, just add path to the images to the TEST_IMAGE_PATHS.
# PATH_TO_TEST_IMAGES_DIR = 'home/'
# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image{}.jpg'.format(i)) for i in range(1, 3) ]

# Size, in inches, of the output images.
IMAGE_SIZE = (8,8)


# In[20]:


# TEST_IMAGE_PATHS = glob.glob(PATH_TO_TEST_IMAGES_DIR+'*.jpg')


# In[21]:


# print(TEST_IMAGE_PATHS)


# In[20]:


# TEST_IMAGE_PATHS = TEST_IMAGE_PATHS[:20]


# In[21]:


# # --working code--
# with detection_graph.as_default():
#   with tf.Session(graph=detection_graph) as sess:
#     # Definite input and output Tensors for detection_graph
#     image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#     # Each box represents a part of the image where a particular object was detected.
#     detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#     # Each score represent how level of confidence for each of the objects.
#     # Score is shown on the result image, together with the class label.
#     detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#     detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#     num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#     for image_path in TEST_IMAGE_PATHS:
#       image = Image.open(image_path)
#       # the array based representation of the image will be used later in order to prepare the
#       # result image with boxes and labels on it.
#       image_np = load_image_into_numpy_array(image)
#       # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#       image_np_expanded = np.expand_dims(image_np, axis=0)
#       # Actual detection.
#       (boxes, scores, classes, num) = sess.run(
#           [detection_boxes, detection_scores, detection_classes, num_detections],
#           feed_dict={image_tensor: image_np_expanded})
#       # Visualization of the results of a detection.
#       vis_util.visualize_boxes_and_labels_on_image_array(
#           image_np,
#           np.squeeze(boxes),
#           np.squeeze(classes).astype(np.int32),
#           np.squeeze(scores),
#           category_index,
#           use_normalized_coordinates=True,
#           line_thickness=8)
#       plt.figure(figsize=IMAGE_SIZE)
#       plt.imshow(image_np)
      


# In[22]:


# predictionList = []
# with detection_graph.as_default():
#   with tf.Session(graph=detection_graph) as sess:
#     # Definite input and output Tensors for detection_graph
#     image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#     # Each box represents a part of the image where a particular object was detected.
#     detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#     # Each score represent how level of confidence for each of the objects.
#     # Score is shown on the result image, together with the class label.
#     detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
#     detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
#     num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    
#     fig = plt.figure()
    
#     for image_path in TEST_IMAGE_PATHS:
#       image = Image.open(image_path)
#       # the array based representation of the image will be used later in order to prepare the
#       # result image with boxes and labels on it.
#       image_np = load_image_into_numpy_array(image)
#       # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#       image_np_expanded = np.expand_dims(image_np, axis=0)
#       # Actual detection.
#       (boxes, scores, classes, num) = sess.run(
#           [detection_boxes, detection_scores, detection_classes, num_detections],
#           feed_dict={image_tensor: image_np_expanded})
#       values = image_path,image_np,boxes,classes,scores,category_index
#       predictionList.append(values)


# In[23]:


# # Visualization of the results of a detection.
# vis_util.visualize_boxes_and_labels_on_image_array(
#   image_np,
#   np.squeeze(boxes),
#   np.squeeze(classes).astype(np.int32),
#   np.squeeze(scores),
#   category_index,
#   use_normalized_coordinates=True,
#   line_thickness=8)
# plt.figure(figsize=IMAGE_SIZE)
# plt.imshow(image_np)


# In[28]:


# detected_image = []
# for i in range(len(predictionList)):
#     image_path,image_np,boxes,classes,scores,category_index = predictionList[i]
#     # Visualization of the results of a detection.
#     vis_util.visualize_boxes_and_labels_on_image_array(
#       image_np,
#       np.squeeze(boxes),
#       np.squeeze(classes).astype(np.int32),
#       np.squeeze(scores),
#       category_index,
#       use_normalized_coordinates=True,
#       line_thickness=12)
    
#     plt.figure(figsize=(4,4))
#     plt.imshow(image_np)
#     detected_image.append(image_np)


# In[28]:


def detectObjectFromPathList(TEST_IMAGE_PATHS, detection_graph=detection_graph, category_index=category_index):
    predictionList = []
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
        # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            fig = plt.figure()

            for image_path in TEST_IMAGE_PATHS:
                image = Image.open(image_path)
          # the array based representation of the image will be used later in order to prepare the
          # result image with boxes and labels on it.
                image_np = load_image_into_numpy_array(image)
          # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
          # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                   [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
                values = image_path,image_np,boxes,classes,scores,category_index
                predictionList.append(values)
                
    detected_image = []
    for i in range(len(predictionList)):
        image_path,image_np,boxes,classes,scores,category_index = predictionList[i]
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=12)

        #plt.figure(figsize=(4,4))
        #plt.imshow(image_np)
        detected_image.append(image_np)

    return detected_image



# In[29]:


# numpy format
# image_array = detectObjectFromPathList(TEST_IMAGE_PATHS, detection_graph, category_index)


# In[30]:


# print(image_array[0])

