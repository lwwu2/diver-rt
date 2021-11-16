import moderngl
from pathlib import Path
import numpy as np

from diver import DIVeR

# debugging purpose
import time
from array import array

""" viewer code """
class DIVeRScene:
    def __init__(self, viewer):
        self.viewer = viewer
        self.ctx = viewer.ctx
        self.width = viewer.width
        self.height = viewer.height
        vert_shader_path = Path(__file__).parent / 'shaders' / 'vertex.glsl'
        frag_shader_path = Path(__file__).parent / 'shaders' / 'fragment.glsl'

        with open(vert_shader_path, 'r') as f:
            vert_shader = f.read()

        with open(frag_shader_path, 'r') as f:
            frag_shader = f.read()

        """ we render everything onto a textured plane"""
        self.prog = self.ctx.program(vertex_shader=vert_shader,fragment_shader=frag_shader)

        self.texture = self.ctx.texture((self.width, self.height), 3, dtype='f4', data=np.zeros((self.width, self.height, 3), dtype='f4'))

        self.vertices = self.ctx.buffer(
            array(
                'f',
                [
                    # Triangle strip creating a fullscreen quad
                    # x, y, u, v
                    -1,  1, 0, 1,  # upper left
                    -1, -1, 0, 0, # lower left
                     1,  1, 1, 1, # upper right
                     1, -1, 1, 0, # lower right
                ]
            )
        )
        self.quad = self.ctx.vertex_array(
            self.prog,
            [
                (self.vertices, '2f 2f', 'in_position', 'in_uv'),
            ]
        )

    def clear(self, color=(0, 0, 0, 0)):
        self.ctx.clear(*color)

    def render(self, camera, model: DIVeR):
        start = time.time()
        self.clear()
        out_p = model.generate_image(camera)
        self.texture.write(out_p)

        self.texture.use(0)
        self.ctx.screen.use()

        self.quad.render(mode=moderngl.TRIANGLE_STRIP)
        end = time.time()
        print(f'FPS: {1 / (end - start):.02f} / time:{end - start:.02f}s')
