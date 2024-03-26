import streamlit as st
import cv2
import numpy as np
from torchvision.models import detection
import torch
from keras.models import load_model
from keras.preprocessing import image as im

st.title("Person Finder + Classifier App")

st.header("About")
st.write("This app detects if there is a human/s in a picture. If a human is present, it classifies each human on an object level as a Zomato delivery agent or any other normal human. If there are no human/s detected it simply writes No Human detected on console")

st.header("Approach")
st.image(cv2.imread("./app_images/1.png"),channels='BGR', caption='Approach')

st.header("Limitations")
st.write("Due to limited training data, the classifier is biased towards a single class i.e the model always identifies the other people as zomato delivery agent. This can be improved over time with enough available data for both the cases. Also, we need to explore logo detection to identify the people in a more fine grained way")
st.write("")
st.sidebar.header("Demo")

model_keras = load_model('../model.h5')

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
CLASSES = ['person',
'bicycle',
'car',
'motorcycle',
'airplane',
'bus',
'train',
'truck',
'boat',
'traffic light',
'fire hydrant',
'street sign',
'stop sign',
'parking meter',
'bench',
'bird',
'cat',
'dog',
'horse',
'sheep',
'cow',
'elephant',
'bear',
'zebra',
'giraffe',
'hat',
'backpack',
'umbrella',
'shoe',
'eye glasses',
'handbag',
'tie',
'suitcase',
'frisbee',
'skis',
'snowboard',
'sports ball',
'kite',
'baseball bat',
'baseball glove',
'skateboard',
'surfboard',
'tennis racket',
'bottle',
'plate',
'wine glass',
'cup',
'fork',
'knife',
'spoon',
'bowl',
'banana',
'apple',
'sandwich',
'orange',
'broccoli',
'carrot',
'hot dog',
'pizza',
'donut',
'cake',
'chair',
'couch',
'potted plant',
'bed',
'mirror',
'dining table',
'window',
'desk',
'toilet',
'door',
'tv',
'laptop',
'mouse',
'remote',
'keyboard',
'cell phone',
'microwave',
'oven',
'toaster',
'sink',
'refrigerator',
'blender',
'book',
'clock',
'vase',
'scissors',
'teddy bear',
'hair drier',
'toothbrush',
'hair brush']

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# initialize a dictionary containing model name and its corresponding 
# torchvision function call
MODELS = {
"frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
"retinanet": detection.retinanet_resnet50_fpn
}

model = MODELS["retinanet"](pretrained=True, progress=True,
num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()

uploaded_file = st.sidebar.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    # Convert the uploaded file to OpenCV format
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Display the uploaded image using OpenCV
    st.image(opencv_image, channels='BGR', caption='Uploaded Image')

    orig = opencv_image.copy()
    # convert the image from BGR to RGB channel ordering and change the
    # image from channels last to channels first ordering
    image = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))
    # add the batch dimension, scale the raw pixel intensities to the
    # range [0, 1], and convert the image to a floating point tensor
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)
    # send the input to the device and pass the it through the network to
    # get the detections and predictions
    image = image.to(DEVICE)
    detections = model(image)[0]

    for i in range(0, len(detections["boxes"])):
    # extract the confidence (i.e., probability) associated with the
    # prediction
        confidence = detections["scores"][i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            idx = int(detections["labels"][i])
            if CLASSES[idx] == 'bicycle': ## this is interpreted as person as the arrangement of labels in list is not aligned
                st.write("Human Detected")

                box = detections["boxes"][i].detach().cpu().numpy()
                (startX, startY, endX, endY) = box.astype("int")
                roi = orig[startY:endY, startX:endX]
                
                ## run the classification model
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                roi = cv2.resize(roi, ((224,224)))

                img_array = im.img_to_array(roi)
                img_array = np.expand_dims(img_array, axis=0)
                
                predictions = model_keras.predict(img_array) 

                if predictions[0] > 0.5:
                    label = 'zomato'
                else:
                    label = 'other'

                cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[idx], 2)
            else:
                st.write("No Human detected")
    st.image(orig, channels='BGR', caption='Output Image')

    