# import the necessary packages
from torchvision.models import detection
import numpy as np
import torch
import cv2

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

# load the image from disk
image = cv2.imread("./data/val/zomato_delivery_boy_Image_6.jpg")
orig = image.copy()
# convert the image from BGR to RGB channel ordering and change the
# image from channels last to channels first ordering
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
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
            print("Human Detected")

            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, "Person", (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        else:
            print("No Human detected")
