
import needle_detection.parameters as p
import cv2 as cv

#def read_cam(channel=0):
#    cap = cv2.VideoCapture(channel)
#    cap.open(channel)
#    cap.set(cv2.CAP_PROP_FRAME_WIDTH, p.image_size_x)
#    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, p.image_size_y)
#    print("Capture device opened!")#

                # Get image and crop to ROI
#            ret, frame = cap.read()
#            if ret == False:
#                print("Video grabber disconnected!")
#                exit
#    return

def read_frame(path):
    #try:
    src = cv.imread(path, cv.IMREAD_GRAYSCALE)
        # Check if image is loaded fine    return frame
    #except:
    #    print('Looser')
    return src

  