import numpy as np
import pydicom
import glob
import math


class util:
    def __init(self):
        pass

    # this function read file which store the vertices of face model
    # and return numpy array of point in model space

    @staticmethod
    def get_canonical_face_vertices():
        return np.loadtxt("models/canonical_face_model_vertices_opencv_coord.txt", dtype=np.float64)

    @staticmethod
    def getVolumeFromFile(path):
        volume = bytes()
        w = 0
        h = 0
        cnt = 0
        print("Loading data Start")
        print('[', end="")
        for file in glob.glob(path + r'\*.*', recursive=False):
            if cnt % 10 == 0:
                print('.', end="")

            ds = pydicom.dcmread(file)
            w = ds.Rows
            h = ds.Columns
            volume = volume + ds.PixelData
            cnt += 1
        print(']')
        print("Loading data is done")
        pixel_array = np.frombuffer(volume, dtype=np.int16)
        # normalize the pixels [0.0 1.0]
        pixel_array_normalized = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array))
        del pixel_array
        del volume
        # I will cach this to file
        #data = np.float32(pixel_array_normalized)
        #data3dshape = np.reshape(data, (w, h, cnt))
        # Compute the gradient in the x, y, and z directions

        #gradient_x = np.gradient(data3dshape, axis=0).flatten()
        #gradient_y = np.gradient(data3dshape, axis=1).flatten()
        #gradient_z = np.gradient(data3dshape, axis=2).flatten()

        #volume_final = np.column_stack((gradient_x, gradient_y, gradient_z, data))
        #del gradient_z
        #del gradient_y
        #del gradient_x
        #del data3dshape
        #del data
        #volume_final = volume_final.flatten()
        return (w, h, cnt), np.float32(pixel_array_normalized)

    @staticmethod
    def rotationMatrixToEulerAngles(R):

        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(R[2, 0], sy)
            z = 0
        return x, y, z