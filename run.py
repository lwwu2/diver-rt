import torch
torch.backends.cudnn.enabled=True
torch.backends.cudnn.benchmark = False

from argparse import ArgumentParser

import moderngl
from scene import DIVeRScene

from orbit_camera import OrbitCamera
import moderngl_window as mglw
from diver import DIVeR


""" camera viewer code"""
class Viewer(mglw.WindowConfig):
    gl_version = (3, 3)
    title = "DIVeR Viewer"
    window_size = (800, 800)
    aspect_ratio = 1 / 1
    resizable = False


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.width = self.window_size[0]
        self.height = self.window_size[1]
        self.aspect_ratio = self.width / self.height
        self.diver_model = DIVeR(self.args)
        self.scene = DIVeRScene(self)

        # self.camera = OrbitCamera(self.width, self.height, focal_length=self.width * 0.7)
        o = (-self.diver_model.xyzmin / self.diver_model.voxel_size)
        z = self.diver_model.voxel_num * 2
        self.camera = OrbitCamera(pivot=[o, o, o], azimuth=0, elevation=-60, zoom=z)

    def mouse_position_event(self, x, y, dx, dy):
        pass
        # self.update(x, y, dx, dy)
        # print("Mouse position:", x, y, dx, dy)

    def mouse_drag_event(self, x, y, dx, dy):
        self.camera.update(x, y, dx, dy)
        # print("Mouse drag:", x, y, dx, dy)

    def mouse_scroll_event(self, x_offset: float, y_offset: float):
        if y_offset < 0:
            self.camera.zoom_in(0, 0)
        else:
            self.camera.zoom_out(0, 0)
        # print("Mouse wheel:", x_offset, y_offset)

    def mouse_press_event(self, x, y, button):
        if button == 1:
            # perform rotation
            self.camera.rotate_start(x, y)
        elif button == 2:
            # perform translation
            self.camera.pan_start(x, y)
        # print("Mouse button {} pressed at {}, {}".format(button, x, y))

    def mouse_release_event(self, x: int, y: int, button: int):
        if button == 1:
            # perform rotation
            self.camera.rotate_end(x, y)
        elif button == 2:
            # perform translation
            self.camera.pan_end(x, y)
        # print("Mouse button {} released at {}, {}".format(button, x, y))



    @classmethod
    def add_arguments(self, parser: ArgumentParser):
        parser.add_argument("--weight_path", type=str, required=True)
        parser.add_argument("--voxel_num", type=int, default=256)
        parser.add_argument("--voxel_dim", type=int, default=32)
        parser.add_argument("--grid_size", type=str, default="2.8")
        parser.add_argument("--device", type=str, default="")
        self.args = parser.parse_known_args()[0]
        return

    def render(self, time, frame_time):
        return self.scene.render(self.camera, self.diver_model)


if __name__ == '__main__':
    Viewer.run()