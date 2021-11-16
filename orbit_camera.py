import numpy as np
import math
import torch

""" orbit camera model """
class OrbitCamera:
    def __init__(self, pivot=[0,0,0], azimuth=0, elevation=60, zoom=400):
        """
        Args:
            pivot: initial obit center (in voxel grid coordinate system)
            azimuth: initial azimuth
            elevation: initial elevation
            zoom: initial zoom in factor (in voxel grid coordinate system)
        """
        self.pivot = torch.tensor(pivot,dtype=torch.float32)
        self.azimuth = np.deg2rad(azimuth)
        self.elevation = np.deg2rad(elevation)
        self.radius = zoom
        self.zoom_speed = 10.
        self.pan_speed = 0.5
        self.rot_speed = 0.017453292519444
        self.rotating = False
        self.panning = False
        
        self.R = torch.zeros(3,3) # rotation matrix
        self.C = torch.zeros(3,) # camera center

        self.updateR()
        self.updateC()
        
    def updateR(self,):
        """ update rotation matrix given current azimuth, elevation"""
        sp = math.sin(self.azimuth)
        cp = math.cos(self.azimuth)
        st = math.sin(self.elevation)
        ct = math.cos(self.elevation)

        self.R[0,0] = cp
        self.R[0,1] = -sp*ct
        self.R[0,2] = sp*st
        self.R[1,0] = sp
        self.R[1,1] = ct*cp
        self.R[1,2] = -st*cp
        self.R[2,1] = st
        self.R[2,2] = ct
        
    def updateC(self,):
        """ update camera center given current orbit center, zooming, and rotation"""
        self.C = self.pivot + self.R[:,-1]*self.radius
        
    def rotate(self,dx,dy):
        """ 
        Args:
            dx: horizontal shift of mouse (azimuth)
            dy: vertical shift of mouse (elevation)
            Assume left click
        """
        self.azimuth -= dx
        self.elevation += dy
        
        sp = math.sin(self.azimuth)
        cp = math.cos(self.azimuth)
        st = math.sin(self.elevation)
        ct = math.cos(self.elevation)

        self.R[0,0] = cp
        self.R[0,1] = -sp*ct
        self.R[0,2] = sp*st
        self.R[1,0] = sp
        self.R[1,1] = ct*cp
        self.R[1,2] = -st*cp
        self.R[2,1] = st
        self.R[2,2] = ct
        
        self.updateC()

    def translate(self,dx,dy):
        """ 
        Args:
            dx: horizontal shift of mouse (move camera plane horitontally)
            dy: vertical shift of mouse (move camera plane vertically)
            Assume right clik
        """
        right,up = self.R[:,0],self.R[:,1]
        self.pivot -=  up*dy-right*dx
        self.updateC()

    def read_zoom(self,fac):
        """ scroll of middle mouse to zoom (in/out) """
        self.radius += fac
        self.updateC()

    def get_position(self):
        '''
        Return 3x1 tensor
        '''
        return -self.extrinsic[:3, :3].T @ self.extrinsic[:3, 3:]

    def set_position(self, pos):
        '''
        Args:
            pos: 3x1 tensor
        '''
        self.extrinsic[:3, 3:] = -self.R @ pos

    position = property(get_position, set_position)

    def rotate_start(self, x, y):
        self.rotating = True

    def rotate_end(self, x, y):
        self.rotating = False

    def pan_start(self, x, y):
        self.panning = True

    def pan_end(self, x, y):
        self.panning = False

    def zoom_in(self, x, y): 
        """ scroll up to zoom in"""
        self.radius += self.zoom_speed
        self.updateC()

    def zoom_out(self, x, y):
        """ scroll down to zoom out"""
        self.radius -= self.zoom_speed
        self.updateC()

    def update(self, x, y, dx, dy):
        if(self.panning):
            # handle camea pan
            self.translate(dx * self.pan_speed, dy * self.pan_speed)

        if(self.rotating):
            # handle camera rot
            self.rotate(dx * self.rot_speed, dy * self.rot_speed)
