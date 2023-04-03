import numpy as np
import torch
import cv2
import sys
import os

from blazebase import resize_pad, denormalize_detections
from blazeface import BlazeFace
from blazeface_landmark import BlazeFaceLandmark
from blazehand_landmark import BlazeHandLandmark

from visualization import draw_detections, draw_landmarks, draw_roi, HAND_CONNECTIONS, FACE_CONNECTIONS

# gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
gpu = torch.device("cpu")
torch.set_grad_enabled(False)

back_detector = False

face_detector = BlazeFace(back_model=back_detector).to(gpu)
root_dir = os.path.dirname(__file__)

face_detector.load_weights(root_dir + "/blazeface.pth")
face_detector.load_anchors(root_dir + "/anchors_face.npy")

face_regressor = BlazeFaceLandmark().to(gpu)
face_regressor.load_weights(root_dir + "/blazeface_landmark.pth")


WINDOW = 'test'
cv2.namedWindow(WINDOW)

if len(sys.argv) > 1:
    capture = cv2.VideoCapture(sys.argv[1])
    mirror_img = False
else:
    capture = cv2.VideoCapture(1)
    mirror_img = True

if capture.isOpened():
    hasFrame, frame = capture.read()
    frame_ct = 0
else:
    hasFrame = False

while hasFrame:
    frame_ct += 1

    if mirror_img:
        frame = np.ascontiguousarray(frame[:, ::-1, ::-1])
    else:
        frame = np.ascontiguousarray(frame[:, :, ::-1])

    _, img2, scale, pad = resize_pad(frame)

    normalized_face_detections = face_detector.predict_on_image(img2)
    face_detections = denormalize_detections(normalized_face_detections, scale, pad)

    xc, yc, scale, theta = face_detector.detection2roi(face_detections.cpu())
    img, affine, box = face_regressor.extract_roi(frame, xc, yc, theta, scale)
    flags, normalized_landmarks = face_regressor(img.to(gpu))
    landmarks = face_regressor.denormalize_landmarks(normalized_landmarks.cpu(), affine)

    for i in range(len(flags)):
        landmark, flag = landmarks[i], flags[i]
        if flag > .5:
            draw_landmarks(frame, landmark[:, :2], FACE_CONNECTIONS, size=1)

    cv2.imshow(WINDOW, frame[:, :, ::-1])

    hasFrame, frame = capture.read()
    key = cv2.waitKey(1)
    if key == 27:
        break

capture.release()
cv2.destroyAllWindows()
