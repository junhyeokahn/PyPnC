import numpy as np
import math

class HeightMap(object):
    def __init__(self,world_grid_size,local_grid_size,world_size,local_size):
        if (world_grid_size/world_size == local_grid_size/local_size):
            self.world_grid_size = world_grid_size
            self.local_grid_size = local_grid_size
            self.local_size = local_size # in meters
            self.world_size = world_size # in meters
            self.world_height_map = np.zeros((world_grid_size,world_grid_size))
            self.local_height_map = np.zeros((local_grid_size,local_grid_size))
            self.cells_per_m = math.ceil(local_grid_size/local_size)
        else:
            raise Exception('heightmap cell distribution error')

    def point_cloud_to_height_map(self, wf_pc):
        """
        Parameters
        ----------
        wf_pc (np.ndarray): world frame point cloud data

        Returns
        -------
            2.5 dimensional world height map
        """
        h = wf_pc.shape[0]
        w = wf_pc.shape[1]
        num_points = h * w

        for i in range(h):
            for j in range(w):
                # point_x_ind = math.floor(wf_pc[i,j,0] * self.cells_per_m) +\
                            # self.world_size*self.cells_per_m/2 - 1
                # point_y_ind = math.floor(wf_pc[i,j,1] * self.cells_per_m) +\
                            # self.world_size*self.cells_per_m/2 - 1
                point_x_ind = wf_pc[i,j,0] * self.cells_per_m +\
                            self.world_size*self.cells_per_m/2 - 1
                point_y_ind = wf_pc[i,j,1] * self.cells_per_m +\
                            self.world_size*self.cells_per_m/2 - 1
                point_x_ind = int(point_x_ind)
                point_y_ind = int(point_y_ind)

                if((point_x_ind>=0) &\
                    (point_x_ind < self.world_size*self.cells_per_m)&\
                    (point_y_ind>=0) &\
                   (point_y_ind < self.world_size*self.cells_per_m)):
                    self.world_height_map[point_x_ind,point_y_ind] = wf_pc[i,j,2]

        return self.world_height_map

    def extract_local_from_wf_heightmap(self, global_robot_pose, wf_heightmap):
        """
        Parameters
        ----------
        global_robot_pose(np.ndarray): global robot pose
        wf_heightmap (np.ndarray): world frame heightmap

        Returns
        -------
        lf_heightmap (np.ndarray): local frame heightmap
        """
        pose_x_ind = global_robot_pose[0]*self.cells_per_m +\
                    self.world_size*self.cells_per_m/2 - 1
        pose_y_ind = global_robot_pose[1]*self.cells_per_m +\
                    self.world_size*self.cells_per_m/2 - 1 
        pose_x_ind = int(pose_x_ind)
        pose_y_ind = int(pose_y_ind)

        for i in range(self.local_grid_size):
            for j in range(self.local_grid_size):
                world_map_x_ind = pose_x_ind + i - self.local_grid_size/2
                world_map_y_ind = pose_y_ind + j - self.local_grid_size/2
                world_map_x_ind = int(world_map_x_ind)
                world_map_y_ind = int(world_map_y_ind)

                if((world_map_x_ind>=0) &\
                    (world_map_x_ind < self.world_size*self.cells_per_m)&\
                    (world_map_y_ind>=0) &\
                   (world_map_y_ind < self.world_size*self.cells_per_m)):
                    self.local_height_map[i,j] =\
                        self.world_height_map[world_map_x_ind,world_map_y_ind] 

        return self.local_height_map


