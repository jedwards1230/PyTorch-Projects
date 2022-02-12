from imutils.video import FPS
import cv2

OPENCV_OBJECT_TRACKERS = {
    "csrt": cv2.legacy.TrackerCSRT_create,
    "kcf": cv2.TrackerKCF_create,
    "mil": cv2.TrackerMIL_create
}

video_stream = "http://192.168.1.143:56000/mjpeg"

vcap = cv2.VideoCapture(video_stream)
if not vcap.isOpened():
    print("Cannot open camera")
    exit()
vcap.set(3, 640)
vcap.set(4, 480)
tracker_choice = 'kcf'

tracker = OPENCV_OBJECT_TRACKERS[tracker_choice]()
initBB = None
fps = None

while True:
    ret, img = vcap.read()
    if not ret:
        continue
    (H, W) = img.shape[:2]
    if initBB is not None:
        # grab the new bounding box coordinates of the object
        (success, box) = tracker.update(img)
        # check to see if the tracking was a success
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(img, (x, y), (x + w, y + h),
                (0, 255, 0), 2)
        # update the FPS counter
        fps.update()
        fps.stop()
        # initialize the set of information we'll be displaying on
        # the frame
        info = [
            ("Tracker", tracker_choice),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]
        # loop over the info tuples and draw them on our frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(img, text, (10, H - ((i * 20) + 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF
    # if the 's' key is selected, we are going to "select" a bounding
    # box to track
    if key == ord("k"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        initBB = cv2.selectROI("Frame", img, fromCenter=False,
            showCrosshair=True)
        # start OpenCV object tracker using the supplied bounding box
        # coordinates, then start the FPS throughput estimator as well
        tracker.init(img, initBB)
        fps = FPS().start()
    elif key == ord("q"):
        break