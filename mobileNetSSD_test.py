# USAGE                                      
# python mobileNet.py --prototxt i../mobilenet_ssd/MobileNetSSD_deploy.prototxt \
#       --model ../mobilenet_ssd/MobileNetSSD_deploy.caffemodel --img people.png

import cv2
import argparse
import imutils
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
                help="path to Caff 'delpoy' prototxt file")
ap.add_argument("-m", "--model", required=True,
                help="path to Caff pre-trained model")
ap.add_argument("-i", "--img", required=True,
                help="input a image name")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")

args = vars(ap.parse_args())

net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# initialize the list of class labels MobileNet SSD was trained to
# detect
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                        "sofa", "train", "tvmonitor"]



img = cv2.imread(args["img"])
img = imutils.resize(img, width = 600)

(h,w) = img.shape[:2]
print("h:%d" %h)
print("w:%d" %h)
blob = cv2.dnn.blobFromImage(img, 0.007843, (w, h), 127.5)

net.setInput(blob)
detections = net.forward()
print(detections.shape[2])
print("args[confidence]:%f" % args["confidence"])
for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    print("confidence:%f" % confidence)
    if confidence > args["confidence"]:
        idx = int(detections[0, 0, i, 1])
        label = CLASSES[idx]
        print("label:%s" % label)
        if CLASSES[idx]!="person":
            continue
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        bb = (startX, startY, endX, endY)
        #print(bb)
        cv2.rectangle(img, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(img, label, (startX, startY-15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

cv2.imshow("test", img)

cv2.waitKey(0)
cv2.destoryAllWindows()
