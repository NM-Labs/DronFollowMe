import sys
import cv2
import os
from sys import platform
import glob
import time
import numpy as np
# from keras import backend as K
# from keras.models import model_from_json
import argparse
from collections import namedtuple
from matplotlib import pyplot as plt
import time
from math import pi, atan2, degrees, sqrt


class Params:
    def __init__(self,dir_path):
            self.dir_path=dir_path
    def set_params(self, face_detection=False, hand_detection=False):

            params = dict()
            params["logging_level"] = 3
            params["output_resolution"] = "-1x-1"
            params["net_resolution"] = "-1x368"
            params["model_pose"] = "BODY_25"
            params["alpha_pose"] = 0.6
            params["scale_gap"] = 0.3
            params["scale_number"] = 1
            params["render_threshold"] = 0.05
            # If GPU version is built, and multiple GPUs are available, set the ID here
            params["num_gpu_start"] = 0
            params["disable_blending"] = False
            # Ensure you point to the correct path where models are located
            params["model_folder"] = self.dir_path + "/../../../models/"

            # if not self.openpose_rendering:
            #     params["render_pose"] = 0
            if face_detection:
                params["face"] = True
                params["face_render_threshold"] = 0.4
            if hand_detection:
                params["hand"] = True
            return params


class OP:
    @staticmethod
    def ccw(A,B,C):
        """
            Returns True if the 3 points A,B and C are listed in a counterclockwise order
            ie if the slope of the line AB is less than the slope of AC
            https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
        """
        return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])
    @staticmethod
    def intersect(A,B,C,D):
        """
            Return true if line segments AB and CD intersect
            https://bryceboe.com/2006/10/23/line-segment-intersection-algorithm/
        """
        if A is None or B is None or C is None or D is None:
            return False
        else:
            return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)
    @staticmethod
    def angle (A, B, C):
        """
            Calculate the angle between segment(A,B) and segment (B,C)
        """
        if A is None or B is None or C is None:
            return None
        return degrees(atan2(C[1]-B[1],C[0]-B[0]) - atan2(A[1]-B[1],A[0]-B[0]))%360
    @staticmethod
    def vertical_angle (A, B):
        """
            Calculate the angle between segment(A,B) and vertical axe
        """
        if A is None or B is None:
            return None
        return degrees(atan2(B[1]-A[1],B[0]-A[0]) - pi/2)
    @staticmethod
    def sq_distance (A, B):
        """
            Calculate the square of the distance between points A and B
        """
        return (B[0]-A[0])**2 + (B[0]-A[0])**2
    @staticmethod
    def distance_check (A, B):
        """
            Calculate the square of the distance between points A and B
        """
        return sqrt((B[0]-A[0])**2 + (B[1]-A[1])**2)

    @staticmethod
    def distance_kps(kp1,kp2):
        # kp1 and kp2: numpy array of shape (3,): [x,y,conf]
        x1,y1,c1 = kp1
        x2,y2,c2 = kp2
        if kp1[2] > 0 and kp2[2] > 0:
            return np.linalg.norm(kp1[:2]-kp2[:2])
        else:
            return 0
    @staticmethod
    def distance (p1, p2):
            """
                Distance between p1(x1,y1) and p2(x2,y2)
            """
            return np.linalg.norm(np.array(p1)-np.array(p2))

    def __init__(self,datumin,number_people_max=-1, min_size=-1, openpose_rendering=False, face_detection=False, frt=0.4, hand_detection=False, debug=False):
        """
        openpose_rendering : if True, rendering is made by original Openpose library. Otherwise rendering is to the
        responsability of the user (~0.2 fps faster)
        """
        self.openpose_rendering = openpose_rendering
        self.min_size = min_size
        self.debug = debug
        self.face_detection = face_detection
        self.hand_detection = hand_detection
        self.frt = frt
        self.datumin = datumin

        self.body_kp_id_to_name = {
            0: "Nose",
            1: "Neck",
            2: "RShoulder",
            3: "RElbow",
            4: "RWrist",
            5: "LShoulder",
            6: "LElbow",
            7: "LWrist",
            8: "MidHip",
            9: "RHip",
            10: "RKnee",
            11: "RAnkle",
            12: "LHip",
            13: "LKnee",
            14: "LAnkle",
            15: "REye",
            16: "LEye",
            17: "REar",
            18: "LEar",
            19: "LBigToe",
            20: "LSmallToe",
            21: "LHeel",
            22: "RBigToe",
            23: "RSmallToe",
            24: "RHeel",
            25: "Background"}


        self.body_kp_name_to_id = {v: k for k, v in self.body_kp_id_to_name.items()}

        self.face_kp_id_to_name = {}
        for i in range(17):
            self.face_kp_id_to_name[i] = f"Jaw{i+1}"
        for i in range(5):
            self.face_kp_id_to_name[i+17 ] = f"REyebrow{5-i}"
            self.face_kp_id_to_name[i+22] = f"LEyebrow{i+1}"
        for i in range(6):
            self.face_kp_id_to_name[(39-i) if i<4 else (45-i)] = f"REye{i+1}"
            self.face_kp_id_to_name[i+42] = f"LEye{i+1}"
        self.face_kp_id_to_name[68] = "REyeCenter"
        self.face_kp_id_to_name[69] = "LEyeCenter"
        for i in range(9):
            self.face_kp_id_to_name[27+i] = f"Nose{i+1}"
        for i in range(12):
            self.face_kp_id_to_name[i+48] = f"OuterLips{i+1}"
        for i in range(8):
            self.face_kp_id_to_name[i+60] = f"InnerLips{i+1}"

        self.face_kp_name_to_id = {v: k for k, v in self.face_kp_id_to_name.items()}

    def get_body_kp(self, kp_name="Neck", person_idx=0):
        """
            Return the coordinates of a keypoint named 'kp_name' of the person of index 'person_idx' (from 0), or None if keypoint not detected
        """
        try:
            # kps = self.datum.poseKeypoints[person_idx]
            kps = self.datumin.poseKeypoints[person_idx]
        except:
            print(f"get_body_kp: invalid person_idx '{person_idx}'")
            return None
        try:
            x,y,conf = kps[self.body_kp_name_to_id[kp_name]]
        except:
            print(f"get_body_kp: invalid kp_name '{kp_name}'")
            return None
        if x or y:
            return (int(x),int(y))
        else:
            return None

    def get_face_kp(self, kp_name="Nose_Tip", person_idx=0):
        """
            Return the coordinates of a keypoint named 'kp_name' of the face of the person of index 'person_idx' (from 0), or None if keypoint not detected
        """
        try:
            kps = self.datum.faceKeypoints[person_idx]
            filter = kps[:,:,2]<self.frt
            kps[filter] = 0
        except:
            print(f"get_face_kp: invalid person_idx '{person_idx}'")
            return None
        try:
            x,y,conf = kps[face_kp_name_to_id[kp_name]]
        except:
            print(f"get_face_kp: invalid kp_name '{kp_name}'")
            return None
        if x or y:
            return (int(x),int(y))
        else:
            return None

    def check_eyes(self, person_idx=0):
        """
            Check if the person whose index is 'person_idx' has his eyes closed
            Return :
            0 if both eyes are open,
            1 if only right eye is closed
            2 if only left eye is closed
            3 if both eyes are closed
        """


        eye_aspect_ratio_threshold = 0.2 # If ear < threshold, eye is closed

        reye_closed = False
        reye1 = self.get_face_kp("REye1", person_idx=person_idx)
        reye2 = self.get_face_kp("REye2", person_idx=person_idx)
        reye3 = self.get_face_kp("REye3", person_idx=person_idx)
        reye4 = self.get_face_kp("REye4", person_idx=person_idx)
        reye5 = self.get_face_kp("REye5", person_idx=person_idx)
        reye6 = self.get_face_kp("REye6", person_idx=person_idx)
        if reye1 and reye2 and reye3 and reye4 and reye5 and reye6:
            right_eye_aspect_ratio = (self.distance(reye2, reye6)+self.distance(reye3, reye5))/(2*self.distance(reye1, reye4))
            if right_eye_aspect_ratio < eye_aspect_ratio_threshold:
                reye_closed = True
                print("RIGHT EYE CLOSED")

        leye_closed = False
        leye1 = self.get_face_kp("LEye1", person_idx=person_idx)
        leye2 = self.get_face_kp("LEye2", person_idx=person_idx)
        leye3 = self.get_face_kp("LEye3", person_idx=person_idx)
        leye4 = self.get_face_kp("LEye4", person_idx=person_idx)
        leye5 = self.get_face_kp("LEye5", person_idx=person_idx)
        leye6 = self.get_face_kp("LEye6", person_idx=person_idx)
        if leye1 and leye2 and leye3 and leye4 and leye5 and leye6:
            left_eye_aspect_ratio = (self.distance(leye2, leye6)+self.distance(leye3, leye5))/(2*self.distance(leye1, leye4))
            if left_eye_aspect_ratio < eye_aspect_ratio_threshold:
                leye_closed = True
                print("LEFT EYE CLOSED")
        if reye_closed:
            if leye_closed:
                return 3
            else:
                return 1
        elif leye_closed:
            return 2
        else:
            return 0
    def check_pose(self):
        """
            kps: keypoints of one person. Shape = (25,3)
        """
        neck = self.get_body_kp("Neck")

        r_wrist = self.get_body_kp("RWrist")
        l_wrist = self.get_body_kp("LWrist")
        r_elbow = self.get_body_kp("RElbow")
        l_elbow = self.get_body_kp("LElbow")
        r_shoulder = self.get_body_kp("RShoulder")
        l_shoulder = self.get_body_kp("LShoulder")
        r_ear = self.get_body_kp("REar")
        l_ear = self.get_body_kp("LEar")

        shoulders_width = self.distance_check(r_shoulder,l_shoulder) if r_shoulder and l_shoulder else None

        vert_angle_right_arm = self.vertical_angle(r_wrist, r_elbow)
        vert_angle_left_arm = self.vertical_angle(l_wrist, l_elbow)

        left_hand_up = neck and l_wrist and l_wrist[1] < neck[1]
        right_hand_up = neck and r_wrist and r_wrist[1] < neck[1]

        if right_hand_up:
            if not left_hand_up:
                # Only right arm up
                if r_ear and (r_ear[0]-neck[0])*(r_wrist[0]-neck[0])>0:
                    # Right ear and right hand on the same side
                    if vert_angle_right_arm:
                        if vert_angle_right_arm < -15:
                            return "RIGHT_ARM_UP_OPEN"
                        if 15 < vert_angle_right_arm < 90:
                            return "RIGHT_ARM_UP_CLOSED"
                elif l_ear and r_wrist[1]>l_ear[1] and shoulders_width : #and distance(r_wrist,l_ear) < shoulders_width/4:
                    # Right hand close to left ear
                    return "RIGHT_HAND_ON_LEFT_EAR"

            else:
                # Check if boths hands are closed to each other and above ears
                # (check right hand is above right ear is enough since hands are closed to each other)
                if shoulders_width and r_ear:
                    near_dist = shoulders_width
                    if r_ear[1] > r_wrist[1] and self.distance_check(r_wrist, l_wrist) < near_dist :
                        return "CLOSE_HANDS_UP"




        else:
            if left_hand_up:
                # Only left arm up
                if l_ear and (l_ear[0]-neck[0])*(l_wrist[0]-neck[0])>0:
                    # Left ear and left hand on the same side
                    if vert_angle_left_arm:
                        if vert_angle_left_arm < -15:
                            return "LEFT_ARM_UP_CLOSED"
                        if 15 < vert_angle_left_arm < 90:
                            return "LEFT_ARM_UP_OPEN"
                elif r_ear and l_wrist[1]>r_ear[1] and shoulders_width: # and distance(l_wrist,r_ear) < shoulders_width/4:
                    # Left hand close to right ear
                    return "LEFT_HAND_ON_RIGHT_EAR"

            else:
                # Both wrists under the neck
                if neck and shoulders_width and r_wrist and l_wrist:
                    near_dist = shoulders_width/3
                    if self.distance_check(r_wrist, neck) < near_dist and self.distance_check(l_wrist, neck) < near_dist :
                        return "HANDS_ON_NECK"



        return None
