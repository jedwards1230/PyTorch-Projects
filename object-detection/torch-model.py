from torchvision import models
from torchvision import transforms as T
import torch
import numpy as np
import cv2
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_stream = "http://192.168.1.143:56000/mjpeg"

coco_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

class MobileNetDetection:
    def __init__(self):
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, pretrained_backbone=True)
        #self.model = models.detection.retinanet_resnet50_fpn(pretrained=True, pretrained_backbone=True)
        self.model.eval()

    def get_prediction(self, img, threshold=0.5):
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        
        pred = self.model([img])
        scores = list(pred[0]['scores'].detach().numpy())
        boxes = list(pred[0]['boxes'].detach().numpy())
        labels = list(pred[0]['labels'].numpy())
        
        if len(scores) > 0 and max(scores) > threshold:
            pred_tensor = [scores.index(x) for x in scores if x > threshold][-1]
            
            predicted_boxes = [[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in boxes]
            predicted_boxes = predicted_boxes[:pred_tensor + 1]
            
            predicted_classes = [coco_classes[i] for i in labels]
            predicted_classes = predicted_classes[:pred_tensor + 1]
            
            return predicted_boxes, predicted_classes
        else:
            return None, None

    def object_detection(self, img, threshold=0.5, rect_th=3, text_size=3, text_th=3):
        boxes, classes = self.get_prediction(img, threshold=threshold)
        
        if boxes is not None:
            for i in range(len(boxes)):
                r, g, b = (0, 255, 0)
                cv2.rectangle(img, boxes[i][0], boxes[i][1], color=(r, g, b), thickness=rect_th)
                text = classes[i].upper()
                cv2.putText(img, text, boxes[i][0], cv2.FONT_HERSHEY_SIMPLEX, text_size, (r, g, b), thickness=text_th)
            
        cv2.imshow('Output', img)
        cv2.waitKey(1)

def main():
    mobile_net_detection = MobileNetDetection()
        
    vcap = cv2.VideoCapture(video_stream)
    if vcap.isOpened():
        W = vcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        H = vcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        vcap.set(cv2.CAP_PROP_FRAME_WIDTH, W/3)
        vcap.set(cv2.CAP_PROP_FRAME_HEIGHT, H/3)
        
        print(vcap.get(cv2.CAP_PROP_FRAME_WIDTH), vcap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        while True:
            _, img = vcap.read()
            if img is not None:
                mobile_net_detection.object_detection(img)
        
if __name__ == "__main__":
    main()