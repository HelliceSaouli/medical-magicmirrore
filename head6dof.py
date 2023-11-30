import cv2 as cv
import mediapipe as ml
import numpy as np
from util import util


# This class is trying to estimate the 6DOF pose
# using Mediapipe and opencv

class headpose:
    def __init__(self):
        self.face_3d_model_space = util.get_canonical_face_vertices()
        self.face_2d_screen_space = None
        # this camera matrix should be given as parameter
        # for know i'm using simple approximation from blender
        self.camera_matrix = np.array([[888.8889, 0.000000, 320.000000],
                                       [0.000000, 888.8889, 240.000000],
                                       [0.000000, 0.000000, 1.000000]])
        self.disto_para = np.zeros((4, 1), dtype=np.float64)

        self.face_solution = ml.solutions.face_mesh
        self.face_mesh = self.face_solution.FaceMesh(static_image_mode=False, max_num_faces=5,
                                                     min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.result = None
        self.pose_matrix = None
        #for drawing axis
        self.axis = np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]]).reshape(-1, 3)

    @staticmethod
    def drawAxis3D(frame, corner, imgpts):
        imgpts = np.int32(imgpts)
        t_corner = (np.int32(corner[0]), np.int32(corner[1]))
        frame = cv.line(frame, t_corner, tuple(imgpts[0].ravel()), (0, 0, 255), 5)
        frame = cv.line(frame, t_corner, tuple(imgpts[1].ravel()), (0, 255, 0), 5)
        frame = cv.line(frame, t_corner, tuple(imgpts[2].ravel()), (255, 0, 0), 5)
        cv.imshow("Demo", frame)

    def find_pose(self, frame):
        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        self.result = self.face_mesh.process(frame_rgb)

        if not self.result.multi_face_landmarks:
            print("No face found !!")
            return

        # list of pixels for face landmark
        face_2d_camera_space = []
        # we have one face
        for face in self.result.multi_face_landmarks:
            for _, vertices_2d in enumerate(face.landmark):
                face_2d_camera_space.append([int(vertices_2d.x * frame_rgb.shape[1]),
                                             int(vertices_2d.y * frame_rgb.shape[0])])

        self.face_2d_screen_space = np.array(face_2d_camera_space, dtype=np.float64)

        # find the pose using solve PNP
        success, rot_vec, tran_vec = cv.solvePnP(self.face_3d_model_space, self.face_2d_screen_space,
                                                 self.camera_matrix, self.disto_para)
        if success:
            rmat, _ = cv.Rodrigues(rot_vec)
            # pitch, yaw, roll = util.rotationMatrixToEulerAngles(rmat)
            self.pose_matrix = np.array([rmat[0, 0], rmat[0, 1], rmat[0, 2], tran_vec[0, 0],
                                         -rmat[1, 0], -rmat[1, 1], -rmat[1, 2], -tran_vec[1, 0],
                                         -rmat[2, 0], -rmat[2, 1], -rmat[2, 2], -tran_vec[2, 0],
                                         0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            print(self.pose_matrix)
        else:
            print("can't find pose")

        # for debuging just draw stuff
        #nose_2d_model = self.face_2d_screen_space[0]
        #imgpts, _ = cv.projectPoints(self.axis, rot_vec, tran_vec, self.camera_matrix, self.disto_para)
        #self.drawAxis3D(frame, nose_2d_model, imgpts)

