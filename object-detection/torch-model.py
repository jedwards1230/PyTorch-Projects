from torchvision import models
from torchvision import transforms as T
from imutils.video import FPS
import torch
import numpy as np
import cv2
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

video_stream = "http://192.168.1.143:56000/mjpeg"

box_color = [0, 255, 0]
fps = None

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
    def __init__(self, threshold=0.5):
        self.threshold = threshold
        self.model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(pretrained=True, pretrained_backbone=True)
        self.model.eval()

    def get_prediction(self, img):
        # Process frame through model
        transform = T.Compose([T.ToTensor()])
        img = transform(img)
        pred = self.model([img])
        
        # Pull relevant data
        scores = list(pred[0]['scores'].detach().numpy())
        boxes = list(pred[0]['boxes'].detach().numpy())
        labels = list(pred[0]['labels'].numpy())
        
        # Filter for objects that meet confidence threshold 
        if len(scores) > 0 and max(scores) > self.threshold:
            pred_tensor = [scores.index(x) for x in scores if x > self.threshold][-1]
            
            predicted_boxes = [[int(i[0]), int(i[1]), int(i[2]), int(i[3])] for i in boxes]
            predicted_boxes = predicted_boxes[:pred_tensor + 1]
            
            predicted_classes = [coco_classes[i] for i in labels]
            predicted_classes = predicted_classes[:pred_tensor + 1]
            
            predicted_scores = scores[:pred_tensor + 1]
            
            return predicted_boxes, predicted_classes, predicted_scores
        else:
            return None, None

    def object_detection(self, img):
        fps = FPS().start()
        boxes, classes, scores = self.get_prediction(img)
        
        if boxes is not None:
            fps.update()
            fps.stop()
            # draw rectangles and labels
            for i in range(len(boxes)):
                cv2.rectangle(img, (boxes[i][0], boxes[i][1]), (boxes[i][2], boxes[i][3]), color=box_color, thickness=3)
                text = classes[i].upper() + ' Acc: ' + str(round(scores[i]*100)) + '%'
                cv2.putText(img, text, (boxes[i][0], boxes[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 2, box_color, thickness=3)
        
        # display fps counter
        fps_text = 'FPS: ' + str(round(fps.fps(), 1))
        cv2.putText(img, fps_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, thickness=3)
        cv2.imshow('Output', img)

def main():
    mobile_net_detection = MobileNetDetection()
    vcap = cv2.VideoCapture(video_stream)
    
    if not vcap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        _, img = vcap.read()
        if img is not None:
            mobile_net_detection.object_detection(img)
        key = cv2.waitKey(1)
        if key is ord('q'):
            break
        
if __name__ == "__main__":
    main()