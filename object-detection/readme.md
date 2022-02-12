Playing with object detection

bb-selector.py uses openCV trackers to track a custom selected bounding box.
 - press 'k' during the stream to select a box, then 'enter' to apply and track

torch-model.py uses the fasterrcnn_mobilenet_v3_large_320_fpn detector to classify and provide bounding boxes. This outputs any object predictions with >50% confidence scores.