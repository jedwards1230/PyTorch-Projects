from imutils.video import FPS
import cv2

video_stream = "http://192.168.1.143:56000/mjpeg"

vcap = cv2.VideoCapture(video_stream)

if not vcap.isOpened():
    print("Cannot open camera")
    exit()
    
vcap.set(3, 640)
vcap.set(4, 480)

tracker = cv2.TrackerKCF_create()
initBB = None
fps = None

while True:
    ret, img = vcap.read()
    if img is not None:
        (H, W) = img.shape[:2]
        if initBB is not None:
            (success, box) = tracker.update(img)
            if success:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(img, (x, y), (x + w, y + h),
                    (0, 255, 0), 2)
                
            fps.update()
            fps.stop()
            
            info = [
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = f'{k}: {v}'
                cv2.putText(img, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
        cv2.imshow("Frame", img)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord("k"):
            initBB = cv2.selectROI("Frame", img, fromCenter=False, showCrosshair=True)
            tracker.init(img, initBB)
            fps = FPS().start()
        elif key == ord("q"):
            break