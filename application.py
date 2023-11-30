import numpy as np
import pygame as pg

import cv2 as cv
import matplotlib.pyplot as plt
import ctypes
import math

from head6dof import headpose
from util import util
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram



class renderTarget:
    def __init__(self, width, height):
        self.fbo = glGenFramebuffers(1)
        self.w = width
        self.h = height
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)

        # need texture to draw into
        self.localposition_texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.localposition_texture)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA16F, width, height, 0, GL_RGBA, GL_FLOAT, None)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.localposition_texture, 0)

        depthrenderbuffer = glGenRenderbuffers(1)
        glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer)

        attachement = [GL_COLOR_ATTACHMENT0]
        glDrawBuffers(1, attachement)
        if glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE:
            print("ERROR Creating frame buffer")
            return
        glBindTexture(GL_TEXTURE_2D, 0)
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0)

    def userendertarget(self):
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbo)
        glViewport(0, 0, self.w, self.h)

    def stoperendertarget(self):
        glBindFramebuffer(GL_FRAMEBUFFER, 0)
        glViewport(0, 0, self.w, self.h)

    def use_texture(self, texture_unit):
        glActiveTexture(texture_unit)
        glBindTexture(GL_TEXTURE_2D, self.localposition_texture)


class volume:
    # this class is to handle 3D texture
    def __init__(self, path):
        self.texture_object = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.texture_object)

        # set texture envirenemnt
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_BASE_LEVEL, 0)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAX_LEVEL, 4)

        # load data
        dim, dicom_volume = util.getVolumeFromFile(path)

        # the pointer thing maybe wrong
        # ptr = dicom_volume.ctypes.data
        glTexImage3D(GL_TEXTURE_3D, 0, GL_R32F, dim[0], dim[1], dim[2], 0, GL_RED, GL_FLOAT, dicom_volume)
        glGenerateMipmap(GL_TEXTURE_3D)
        glBindTexture(GL_TEXTURE_3D, 0)
        # del ptr
        del dicom_volume

    def use_texture(self, texture_unit):
        glActiveTexture(texture_unit)
        glBindTexture(GL_TEXTURE_3D, self.texture_object)


class cube_shader:
    # this class will create shader that only draw cubes
    # it will help to make sure that tracking is working
    # properly
    def __init__(self, vertex_shader_path, fragement_shader_path):
        with open(vertex_shader_path, 'r') as f:
            vertex_src = f.readlines()

        with open(fragement_shader_path, 'r') as f:
            fragement_src = f.readlines()

        self.shader_program = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),
                                             compileShader(fragement_src, GL_FRAGMENT_SHADER))
        # initialize empty dictionary for sotring uniforms
        # this is an overkill since i do not need that much of uniforms
        self.uniforms = {}

    def add_uniform(self, uniform_name):
        uniform_location = glGetUniformLocation(self.shader_program, uniform_name)
        if uniform_location == -1:
            print(f'No uniform with name {uniform_name} is found')
            return
        self.uniforms[uniform_name] = uniform_location

    def uniform3f(self, uniform_name, x, y, z):
        glUniform3f(self.uniforms[uniform_name], x, y, z)

    def uniformMatrix4(self, uniform_name, matrix4, transpose):
        glUniformMatrix4fv(self.uniforms[uniform_name], 1, transpose, matrix4)

    def bind_shader(self):
        glUseProgram(self.shader_program)

    def unbind(self):
        glUseProgram(0)


class cube:
    # this class will represent the cube that will contain the  3D volume
    # CT scan data to be rendered.
    # We use  full transform, but we  use only matrices (camera projection and modelview)

    def __init__(self, shader_object=None):
        # i need only positions x, y, z
        self.vertices = (-0.5, 0.5, -0.5,
                         -0.5, -0.5, -0.5,
                         -0.5, 0.5, 0.5,
                         -0.5, -0.5, 0.5,
                         0.5, 0.5, -0.5,
                         0.5, -0.5, -0.5,
                         0.5, 0.5, 0.5,
                         0.5, -0.5, 0.5)

        self.indices = (2, 3, 1, 4, 7, 3, 8, 5, 7, 6, 1, 5, 7, 1, 3, 4, 6, 8, 2, 4, 3, 4, 8, 7, 8, 6, 5, 6, 2, 1, 7, 5,
                        1, 4, 2, 6)

        self.vertices = np.array(self.vertices, dtype=np.float32)
        # mins 1 from indices because am lazy
        self.indices = np.array(self.indices, dtype=np.int32) - 1
        self.draw_count = 36

        # create vao
        self.vertex_array_object = glGenVertexArrays(1)
        glBindVertexArray(self.vertex_array_object)

        self.vertex_buffer_object = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer_object)

        glBufferData(GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, GL_STATIC_DRAW)
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * 4, ctypes.c_void_p(0))
        glEnableVertexAttribArray(0)

        self.element_buffer_object = glGenBuffers(1)
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, self.element_buffer_object)
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, GL_STATIC_DRAW)
        glBindVertexArray(0)

        self.matrial = shader_object
        # lets  make this outside
        # if self.matrial is not None:
        #    self.matrial.add_uniform("projection_matrix")
        #    self.matrial.add_uniform("modelview_matrix")

    def destroy(self):
        glDeleteVertexArrays(1, (self.vertex_array_object,))
        glDeleteBuffers(2, (self.vertex_buffer_object, self.element_buffer_object))

    def update(self, cameramatrix, modelview):
        if self.matrial is not None:
            self.matrial.bind_shader()
            self.matrial.uniformMatrix4("projection_matrix", cameramatrix, GL_FALSE)
            self.matrial.uniformMatrix4("modelview_matrix", modelview, GL_TRUE)

    def draw(self):
        glBindVertexArray(self.vertex_array_object)
        # passing 0 like C++ will not work None must be used
        glDrawElements(GL_TRIANGLES, self.draw_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)

class TransferControlPoint:
    def __init__(self, r, g, b, a, isovalue):
        self.color = [r, g, b, a]
        self.isovalue = isovalue


class cubic:
    # Cubic class that calculates the cubic spline from a set of control points/knots
    # and performs cubic interpolation.
    def __init__(self, a, b, c, d):
        self.a = a
        self.b = b
        self.c = c
        self.d = d

    def GetPointOnSpline(self, s):
        x = (((self.d[0] * s) + self.c[0]) * s + self.b[0]) * s + self.a[0]
        y = (((self.d[1] * s) + self.c[1]) * s + self.b[1]) * s + self.a[1]
        z = (((self.d[2] * s) + self.c[2]) * s + self.b[2]) * s + self.a[2]
        w = (((self.d[3] * s) + self.c[3]) * s + self.b[3]) * s + self.a[3]
        return [x, y, z, w]

    @staticmethod
    def CalculateCubicSpline(n, list_of_transfer_control_point):
        gama = [None] * (n + 1)
        delta = [None] * (n + 1)
        D = [None] * (n + 1)

        gama[0] = [0.5, 0.5, 0.5, 0.5]
        for i in range(1, n):
            gama_prev = gama[i - 1]
            x = 1.0 / (4 - gama_prev[0])
            y = 1.0 / (4 - gama_prev[1])
            z = 1.0 / (4 - gama_prev[2])
            w = 1.0 / (4 - gama_prev[3])
            gama[i] = [x, y, z, w]

        gama_prev = gama[n - 1]
        x = 1.0 / (2 - gama_prev[0])
        y = 1.0 / (2 - gama_prev[1])
        z = 1.0 / (2 - gama_prev[2])
        w = 1.0 / (2 - gama_prev[3])
        gama[n] = [x, y, z, w]

        color_1 = list_of_transfer_control_point[1].color
        color_0 = list_of_transfer_control_point[0].color

        gama_prev = gama[0]
        r = 3 * (color_1[0] - color_0[0]) * gama_prev[0]
        g = 3 * (color_1[1] - color_0[1]) * gama_prev[1]
        b = 3 * (color_1[2] - color_0[2]) * gama_prev[2]
        a = 3 * (color_1[3] - color_0[3]) * gama_prev[3]
        delta[0] = [r, g, b, a]

        for i in range(1, n):
            color_1 = list_of_transfer_control_point[i + 1].color
            color_0 = list_of_transfer_control_point[i - 1].color
            gama_prev = gama[i]
            delta_prev = delta[i - 1]
            r = (3 * (color_1[0] - color_0[0]) - delta_prev[0]) * gama_prev[0]
            g = (3 * (color_1[1] - color_0[1]) - delta_prev[1]) * gama_prev[1]
            b = (3 * (color_1[2] - color_0[2]) - delta_prev[2]) * gama_prev[2]
            a = (3 * (color_1[3] - color_0[3]) - delta_prev[3]) * gama_prev[3]
            delta[i] = [r, g, b, a]

        color_1 = list_of_transfer_control_point[n].color
        color_0 = list_of_transfer_control_point[n - 1].color
        gama_prev = gama[n]
        delta_prev = delta[n - 1]
        r = (3 * (color_1[0] - color_0[0]) - delta_prev[0]) * gama_prev[0]
        g = (3 * (color_1[1] - color_0[1]) - delta_prev[1]) * gama_prev[1]
        b = (3 * (color_1[2] - color_0[2]) - delta_prev[2]) * gama_prev[2]
        a = (3 * (color_1[3] - color_0[3]) - delta_prev[3]) * gama_prev[3]
        delta[n] = [r, g, b, a]

        D[n] = [r, g, b, a]
        for i in range(n - 1, -1, -1):
            curr_delta = delta[i]
            curr_gama = gama[i]
            next_d = D[i + 1]
            x = curr_delta[0] - curr_gama[0] * next_d[0]
            y = curr_delta[1] - curr_gama[1] * next_d[1]
            z = curr_delta[2] - curr_gama[2] * next_d[2]
            w = curr_delta[3] - curr_gama[3] * next_d[3]
            D[i] = [x, y, z, w]

        result = [None] * n
        for i in range (0, n):
            a = list_of_transfer_control_point[i].color
            b = D[i]
            color_a = list_of_transfer_control_point[i + 1].color
            color_b = list_of_transfer_control_point[i].color
            next_d = D[i + 1]

            c = [3 * (color_a[0] - color_b[0]) - 2 * b[0] - next_d[0], 3 * (color_a[1] - color_b[1]) - 2 * b[1] -
                 next_d[1], 3 * (color_a[2] - color_b[2]) - 2 * b[2] - next_d[2], 3 * (color_a[3] - color_b[3]) - 2 *
                 b[3] - next_d[3]]

            d = [2 * (color_b[0] - color_a[0]) + b[0] + next_d[0], 2 * (color_b[1] - color_a[1]) + b[1] +
                 next_d[1], 2 * (color_b[2] - color_a[2]) + b[2] + next_d[2], 2 * (color_b[3] - color_a[3]) +
                 b[3] + next_d[3]]

            result[i] = cubic(a, b, c, d)

        return result


class transferfunction:
    def __init__(self):
        self.transferFunc = [None] * 256
        pass

    def computeTransferFunction(self, color_knots, alpha_knots):
        color_cubic = cubic.CalculateCubicSpline(len(color_knots) - 1, color_knots)
        alpha_cubic = cubic.CalculateCubicSpline(len(alpha_knots) - 1, alpha_knots)

        numTF = 0
        for i in range(0, len(color_knots) - 1):
            step = color_knots[i + 1].isovalue - color_knots[i].isovalue
            for j in range(0, int(step)):
                k = j / (step - 1)
                self.transferFunc[numTF] = color_cubic[i].GetPointOnSpline(k)
                numTF += 1

        numTF = 0
        for i in range(0, len(alpha_knots) - 1):
            step = alpha_knots[i + 1].isovalue - alpha_knots[i].isovalue
            for j in range(0, int(step)):
                k = j / (step - 1)
                alpha = alpha_cubic[i].GetPointOnSpline(k)
                self.transferFunc[numTF][3] = alpha[3]
                numTF += 1

        self.transferFunc = np.array(self.transferFunc, dtype=np.float32)


class colorlut:
    def __init__(self):
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_1D, self.texture)

        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        # Define control points for opacity and color

        color_Knots = [TransferControlPoint(0.91, 0.0, 0.0, 1.0, 0.0),
                       TransferControlPoint(1.0, 0.2, 0.2, 1.0, 63.0),
                       TransferControlPoint(1.0, 0.0, 0.0, 1.0, 80.0),
                       TransferControlPoint(1.0, 1.0, 0.85, 1.0, 82.0),
                       TransferControlPoint(1.0, 1.0, 0.85, 1.0, 256.0)]

        alpha_Knots = [TransferControlPoint(0.0, 0.0, 0.0, 0.0, 0.0),
                       TransferControlPoint(0.0, 0.0, 0.0, 0.0, 40.0),
                       TransferControlPoint(0.0, 0.0, 0.0, 0.01, 50.0),
                       TransferControlPoint(0.0, 0.0, 0.0, 0.0, 60.0),
                       TransferControlPoint(0.0, 0.0, 0.0, 0.08, 63.0),
                       TransferControlPoint(0.0, 0.0, 0.0, 0.15, 70.0),
                       TransferControlPoint(0.0, 0.0, 0.0, 0.0, 80.0),
                       TransferControlPoint(0.0, 0.0, 0.0, 0.6, 82.0),
                       TransferControlPoint(0.0, 0.0, 0.0, 0.0, 256.0)]

        transfer_function = transferfunction()
        transfer_function.computeTransferFunction(color_Knots, alpha_Knots)
        # np.savetxt("tf.txt", transfer_function.transferFunc, delimiter=',')
        glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA32F, 256, 0, GL_RGBA, GL_FLOAT, transfer_function.transferFunc)
        glBindTexture(GL_TEXTURE_1D, 0)

    def use_texture(self, texture_unit):
        glActiveTexture(texture_unit)
        glBindTexture(GL_TEXTURE_1D, self.texture)


class vision:
    # this class is but just asimple wraper
    def __init__(self, is_video=False, video_path=""):
        self.is_vid = is_video
        if self.is_vid:
            self.vid = cv.VideoCapture(video_path)
        else:
            self.vid = cv.VideoCapture(0)

        self.head_pose_estimator = headpose()
        self.modelview = None
        self.projection = None
        #setup gl texture to use to render video
        self.texture = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture)

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        #do something about size later
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 640, 480, 0, GL_BGR, GL_UNSIGNED_BYTE, None)
        glGenerateMipmap(GL_TEXTURE_2D)
        glBindTexture(GL_TEXTURE_2D, 0)

    def use_texture(self, texture_unit):
        glActiveTexture(texture_unit)
        glBindTexture(GL_TEXTURE_2D, self.texture)

    def estimate(self):
        # capture frame-by-frame
        ret, frame = self.vid.read()

        if self.is_vid:
            if self.vid.get(1) > self.vid.get(7) - 2:
                self.vid.set(1, 0)

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return False
        # send frame to opengl
        glBindTexture(GL_TEXTURE_2D, self.texture)
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_BGR, GL_UNSIGNED_BYTE, frame)

        self.head_pose_estimator.find_pose(frame)
        self.modelview = self.head_pose_estimator.pose_matrix
        return True

class App:
    def __init__(self, screen=(640, 480), app_title='Magic Mirror', show=True):

        pg.init()
        self.back_ground = show
        pg.display.set_caption(app_title)
        pg.display.set_mode(screen, pg.OPENGL | pg.DOUBLEBUF)
        self.clock = pg.time.Clock()
        glClearColor(0, 0, 0, 1)

        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        # glEnable(GL_CULL_FACE)
        # glFrontFace(GL_CW)
        # glCullFace(GL_BACK)

        # initiate mesh with shader
        self.volume_render = cube_shader('shaders/simple_vertex.txt',
                                         'shaders/simple_fragement.txt')
        self.volume_render.add_uniform("projection_matrix")
        self.volume_render.add_uniform("modelview_matrix")

        self.back_front_shader = cube_shader('shaders/frontback_vertex.txt', 'shaders/frontback_fragement.txt')
        self.back_front_shader.add_uniform("projection_matrix")
        self.back_front_shader.add_uniform("modelview_matrix")

        self.back_ground_shader = cube_shader('shaders/background_vert.txt', 'shaders/background_frag.txt')
        #self.back_ground_shader.add_uniform("projection_matrix")
        #self.back_ground_shader.add_uniform("modelview_matrix")

        # create two render targets front and back
        self.front_render_target = renderTarget(screen[0], screen[1])
        self.back_render_target = renderTarget(screen[0], screen[1])

        self.mycube = cube()
        self.bg = cube(self.back_ground_shader)

        # the data is from VHMCT1mm-Head the size of voxel is 0.001
        self.myvolume = volume(r'C:\Users\Abdelhak\Documents\DATASETS\Visible Human\Male\head')
        #set up 1D transfer function
        self.volume_color = colorlut()
        # initalize vision class for head detection
        self.head_estimation = vision()
        self.run()

    def run(self):
        running = True
        # calibration data
        fx = 888.8889
        fy = 888.8889
        cx = 320.000000
        cy = 240.550000
        zN, zF = (0.001, 10000.0)
        a = 640.0 / 480.0
        pMatrix = np.array([fx / cx, 0.0, 0.0, 0.0,
                            0.0, fy / cy, 0.0, 0.0,
                            0.0, 0.0, (zF + zN) / (zN - zF), -1.0,

                            0.0, 0.0, 2.0 * zF * zN / (zN - zF), 0.0], dtype=np.float32)
        # modelview matrix bg
        #mvMatrixbg = np.array([1998*cx/fx, 0.0, 0.0, 0.0,
        #                       0.0, 1998*cy/fy, 0.0, 0.0,
        #                       0.0, 0.0, 1.0, -999.0,
        #                       0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        while running:
            # check events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # Update
            success = self.head_estimation.estimate()

            # first pass cull back
            self.mycube.matrial = self.back_front_shader
            if success:
                self.mycube.update(pMatrix, self.head_estimation.modelview)
            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)
            self.front_render_target.userendertarget()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.mycube.draw()
            self.mycube.matrial.unbind()
            self.front_render_target.stoperendertarget()
            glDisable(GL_CULL_FACE)

            if success:
                self.mycube.update(pMatrix, self.head_estimation.modelview)
            glEnable(GL_CULL_FACE)
            glCullFace(GL_FRONT)
            self.back_render_target.userendertarget()
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.mycube.draw()
            self.mycube.matrial.unbind()
            self.back_render_target.stoperendertarget()
            glDisable(GL_CULL_FACE)

            glEnable(GL_CULL_FACE)
            glCullFace(GL_BACK)

            self.mycube.matrial = self.volume_render
            if success:
                self.mycube.update(pMatrix, self.head_estimation.modelview)
            self.myvolume.use_texture(GL_TEXTURE0)
            self.front_render_target.use_texture(GL_TEXTURE1)
            self.back_render_target.use_texture(GL_TEXTURE2)
            self.volume_color.use_texture(GL_TEXTURE3)

            self.mycube.draw()
            self.mycube.matrial.unbind()

            # draw background
            if self.back_ground:
                self.head_estimation.use_texture(GL_TEXTURE0)
                self.bg.matrial.bind_shader()
                #self.bg.update(pMatrix, mvMatrixbg)
                self.bg.draw()

            pg.display.flip()
            self.clock.tick(60)

        self.stop()

    def stop(self):
        self.mycube.destroy()
        pg.quit()


if __name__ == "__main__":
    engine = App((640, 480), "Magic Mirror Head")
