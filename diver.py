import torch
import torch.nn as nn
import torch.nn.functional as NF
from ray_march import aabb_intersect,ray_march
from mlp_evaluation import mlp_eval, upload_weight

""" real-time DIVeR code"""
class DIVeR(nn.Module):
    def __init__(self, hparams):
        super(DIVeR, self).__init__()
        self.voxel_num = hparams.voxel_num
        self.voxel_dim = hparams.voxel_dim
        self.grid_size = float(hparams.grid_size)
        self.voxel_size = self.grid_size/self.voxel_num
        self.weights_path = hparams.weight_path
        self.device = torch.device(hparams.device)

        # TODO: parameterize these
        self.render_shape = (800,800)
        self.max_hits = 8
        
        N = self.voxel_num*self.voxel_size
        self.xyzmin = -N*0.5
        self.xyzmax = N*0.5


        # load in weights
        with torch.no_grad():
            self.params, self.voxel_masks, self.voxels, \
            self.octrees = self.load_weights(hparams.weight_path)
        
        # image rgba buffers
        self.buffer = torch.zeros(*self.render_shape, 4, device=self.device).contiguous()
        # mask of whether a pixel has finished evaluation
        self.finish = torch.zeros(*self.render_shape, dtype=bool, device=self.device).contiguous()
        # buffer to store intersection coordinates, max_hits x h x w x [entry point, exit point]
        self.coords = torch.zeros(self.max_hits, *self.render_shape,6, device=self.device).contiguous()
        # buffer to store ray direction
        self.directions = torch.zeros(*self.render_shape, 3, device=self.device).contiguous()      

    def load_weights(self, weight_path):
        """ load model weights """
        device = self.device
        weight = torch.load(weight_path, map_location='cpu')
        params = weight['p'].contiguous() # paded to fit in a warp (32)

        inds = weight['i'].long() # flattened inices of vertex
        m_mask = weight['m'] # the incides of the occupancy mask (which is a subsec of vertex indices)
        ii = inds // ((self.voxel_num+1)*(self.voxel_num+1))
        inds = inds % ((self.voxel_num+1)*(self.voxel_num+1))
        jj = inds // (self.voxel_num+1)
        kk = inds % (self.voxel_num+1)
        
        idxs = torch.stack([ii,jj,kk],0)
        features = weight['f'] # 1D array of feature vectors
        voxel_masks = torch.zeros(self.voxel_num,\
                    self.voxel_num,self.voxel_num,dtype=bool)
        voxel_masks[ii[m_mask],jj[m_mask],kk[m_mask]] = True
        voxel_masks = voxel_masks.to(device).contiguous()

        # build map from vertex to feature vector indices
        voxel_map = torch.zeros([self.voxel_num+1]*3,dtype=torch.int)
        voxel_map[idxs[0],idxs[1],idxs[2]] = torch.arange(len(idxs[0])).int()
        voxels = features.to(device).contiguous()
        voxel_map = voxel_map.contiguous()

        # build an octree of occupancy map (only used for closest hit calculation)
        level = 3
        scales = [4**i for i in range(1,1+level)]
        octrees = [NF.max_pool3d(voxel_masks[None,None].float(),s,stride=s)[0,0].bool().reshape(-1) for s in scales[::-1]]
        octrees = torch.cat(octrees,0).contiguous()
        dev_index = device.index if device.index is not None else 0
        
        upload_weight(dev_index,params,voxel_map)

        return params, voxel_masks, voxels, octrees

    def generate_image(self, camera):
        # closest hit
        aabb_intersect(self.coords, self.directions, camera.C, camera.R, self.finish, self.buffer, self.octrees, self.voxel_num)
        
        # ray-marching and decode until all the rays terminate 
        while not self.finish.all():
            ray_march(self.coords, self.directions, self.voxel_masks, self.finish, self.voxel_num)
            mlp_eval(self.buffer, self.coords, self.voxels, self.directions ,self.finish)

        # assume background to be white
        return (self.buffer[:,:,:3]+self.buffer[:,:,3:]).cpu().numpy()