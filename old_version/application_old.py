import numpy as np
import pygame as pg
import cv2 as cv

import ctypes
import math

from head6dof import headpose
from util import util
from OpenGL.GL import *
from OpenGL.GL.shaders import compileShader, compileProgram


class volume:
    # this class is to handle 3D texture
    def __init__(self, path):
        self.texture_object = glGenTextures(1)
        glBindTexture(GL_TEXTURE_3D, self.texture_object)

        # set texture envirenemnt
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, GL_CLAMP)
        glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, GL_CLAMP)
        glTexParameterf(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR)
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


class cube:
    # this class will represent the cube that will contain the  3D volume
    # CT scan data to be rendered.
    # We use  full transform, but we  use only matrices (camera projection and modelview)

    def __init__(self, shader_object):
        # i need only positions x, y, z
        self.vertices = (-0.079000, -0.079000, 0.079000,
                         -0.079000, 0.079000, 0.079000,
                         -0.079000, -0.079000, -0.079000,
                         -0.079000, 0.079000, -0.079000,
                         0.079000, -0.079000, 0.079000,
                         0.079000, 0.079000, 0.079000,
                         0.079000, -0.079000, -0.079000,
                         0.079000, 0.079000, -0.079000)

        self.indices = (2, 3, 1, 4, 7, 3, 8, 5, 7, 6, 1, 5, 7, 1, 3, 4, 6, 8, 8, 6, 5, 4, 8, 7, 4, 2, 6, 2, 4,
                        3, 6, 2, 1, 7, 5, 1)

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
        self.matrial.add_uniform("projection_matrix")
        self.matrial.add_uniform("modelview_matrix")

    def destroy(self):
        glDeleteVertexArrays(1, (self.vertex_array_object,))
        glDeleteBuffers(2, (self.vertex_buffer_object, self.element_buffer_object))

    def update(self, cameramatrix, modelview):
        self.matrial.bind_shader()
        self.matrial.uniformMatrix4("projection_matrix", cameramatrix, GL_FALSE)
        self.matrial.uniformMatrix4("modelview_matrix", modelview, GL_TRUE)

    def draw(self):
        glBindVertexArray(self.vertex_array_object)
        # passing 0 like C++ will not work None must be used
        glDrawElements(GL_TRIANGLES, self.draw_count, GL_UNSIGNED_INT, None)
        glBindVertexArray(0)


class vision:
    # this class is but just asimple wraper
    def __init__(self):
        self.vid = cv.VideoCapture(0)
        self.head_pose_estimator = headpose()
        self.modelview = None
        self.projection = None

    def estimate(self):
        # capture frame-by-frame
        ret, frame = self.vid.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            return False

        self.head_pose_estimator.find_pose(frame)
        self.modelview = self.head_pose_estimator.pose_matrix
        return True


class App:

    def __init__(self, screen=(640, 480), app_title='Magic Mirror'):

        pg.init()
        pg.display.set_caption(app_title)
        pg.display.set_mode(screen, pg.OPENGL | pg.DOUBLEBUF)
        self.clock = pg.time.Clock()
        glClearColor(0, 0, 0, 1)

        glEnable(GL_DEPTH_TEST)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_CULL_FACE)
        glFrontFace(GL_CW)
        glCullFace(GL_BACK)

        # initiate mesh with shader
        self.mycube = cube(cube_shader('../shaders/simple_vertex.txt', 'shaders/simple_fragement.txt'))
        self.myvolume = volume(r'C:\Users\Abdelhak\Documents\DATASETS\Visible Human\Male\head')
        # initalize vision class for head detection
        self.head_estimation = vision()
        self.run()

    def run(self):
        running = True
        # calibration data
        fx = 3310.400000
        fy = 3325.500000
        cx = 316.730000
        cy = 200.550000
        zN, zF = (0.001, 10000.0)
        a = 640.0 / 480.0
        pMatrix = np.array([fx / cx, 0.0, 0.0, 0.0,
                            0.0, fy / cy, 0.0, 0.0,
                            0.0, 0.0, (zF + zN) / (zN - zF), -1.0,

                            0.0, 0.0, 2.0 * zF * zN / (zN - zF), 0.0], dtype=np.float32)
        # modelview matrix
        # mvMatrix = np.array([1.0, 0.0, 0.0, 0.0,
        #                     0.0, 1.0, 0.0, 0.0,
        #                    0.0, 0.0, 1.0, 0.0,
        #                    0.5, 0.0, -5.0, 1.0], dtype=np.float32)
        while running:
            # check events
            for event in pg.event.get():
                if event.type == pg.QUIT:
                    running = False

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            # Update
            success = self.head_estimation.estimate()

            if success:
                self.mycube.update(pMatrix, self.head_estimation.modelview)
            # Draw the mesh
            self.myvolume.use_texture(GL_TEXTURE0)

            glEnable(GL_BLEND)
            self.mycube.draw()
            glDisable(GL_BLEND)

            pg.display.flip()
            self.clock.tick(60)

        self.stop()

    def stop(self):
        self.mycube.destroy()
        pg.quit()


if __name__ == "__main__":
    engine = App((640, 480), "Magic Mirror Head")
