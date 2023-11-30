import cv2 as cv
import numpy as np

from util import util


# https://github.com/niconielsen32/ComputerVision/commit/434249f186bea0e9a5ac70344a47bfb9ea2f6374
class depthmap:
    def __init__(self):
        self.disparity = None
        #model = "DNN/model-f6b98070.onnx"
        model = "DNN/model-small.onnx"
        # Load model
        self.dnnmodel = cv.dnn.readNet(model)

        if self.dnnmodel.empty():
            print("could not load the MiDas Small model")

        # Setup cuda backendand target support
        # self.dnnmodel.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
        # self.dnnmodel.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    def estimatedepth(self, frame, width, hieght):
        # MiDaS v2.1 Small ( Scale : 1 / 255, Size : 256 x 256,
        # Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        blob = cv.dnn.blobFromImage(frame, 1/255., (256, 256), (123.675, 116.28, 103.53), True, False)

        # MiDaS v2.1 Large ( Scale : 1 / 255, Size : 384 x 384,
        # Mean Subtraction : ( 123.675, 116.28, 103.53 ), Channels Order : RGB )
        #blob = cv.dnn.blobFromImage(frame, 1 / 255., (384, 384), (123.675, 116.28, 103.53), True, False)

        self.dnnmodel.setInput(blob)

        self.disparity = self.dnnmodel.forward()

        self.disparity = self.disparity[0, :, :]
        self.disparity = cv.resize(self.disparity, (width, hieght))
        self.disparity = cv.normalize(self.disparity, None, 0, 1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)


if __name__ == "__main__":
    mono_depth = depthmap()

    capture = cv.VideoCapture(0)

    while capture.isOpened():
        ret, frame = capture.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        mono_depth.estimatedepth(frame, frame.shape[1], frame.shape[0])

        cv.imshow("estimated depth map", mono_depth.disparity)

        if cv.waitKey(10) & 0xFF == ord('q'):
            break

    capture.release()
    cv.destroyAllWindows()



