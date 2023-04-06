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


def save_model(model_, des_weight_path):
    w = {k: v.to(dtype=torch.float32) for k, v in model_.state_dict().items()}
    torch.save(w, des_weight_path)


def print_weight_list(model_, des_txt_filepath, with_weight=False):
    with open(des_txt_filepath, "w+") as f_handle:

        txt = ""
        for name, param in model_.named_parameters():
            txt += "[{}] {}\n".format(name, param.shape)
            if with_weight:
                txt += str(param.numpy())
                txt += "\n" + "---------------" + "\n"

        f_handle.write(txt)


if __name__ == "__main__":
    # save_model(face_detector, "face_detector.pt")
    # save_model(face_regressor, "face_regressor.pt")
    torch.save({"anchor": face_detector.anchors}, "anchors.pt")
