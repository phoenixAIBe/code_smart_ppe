import cv2

cap = cv2.VideoCapture(os.getenv('RTSP_IN'))
if(cap.isOpened()):
    ret, frame = cap.read()
    if frame is not None :
        cv2.imwrite("public/snapshot.png",frame)
    else :
        print("image non")
else :
    print("error")
cap.release()
