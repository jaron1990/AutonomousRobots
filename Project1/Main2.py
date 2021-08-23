import matplotlib.pyplot as plt 
import matplotlib.image as mgimg
from matplotlib import animation
import os
import pykitti
import numpy as np
import pymap3d as pm
import math
import time
import pandas as pd

start_frame=0
num_of_frames = 100

xmin = -30
ymin = -83
xmax = 61
ymax = 30

alpha = 0.2
max_dist_hit = 30

hit_prob = 0.9
miss_prob = 0.4
occ_thr = 0.8
free_thr = 0.2
occ_sat = 0.95
free_sat = 0.05

hit_prob_logit  = math.log(hit_prob/(1-hit_prob))
miss_prob_logit = math.log(miss_prob/(1-miss_prob))
occ_thr_logit   = math.log(occ_thr/(1-occ_thr))
free_thr_logit  = math.log(free_thr/(1-free_thr))
occ_sat_logit   = math.log(occ_sat/(1-occ_sat))
free_sat_logit  = math.log(free_sat/(1-free_sat))

def remove_long_hits(input_lidar, max_range):
    valid_indices = (input_lidar[:,0]**2 + input_lidar[:,1]**2 < max_range**2)
    short_hits = input_lidar[valid_indices]
    return short_hits

def get_obstacles_from_lidar(raw_lidar, max_range, min_obst_height, min_hits_for_obst):
    lidar_short = remove_long_hits(raw_lidar, max_range)
    lidar_to_indices = np.array(lidar_short[:,:2]*5, dtype=np.int32) + max_range*5

    lidar_to_indices = np.hstack((lidar_to_indices, lidar_short[:,2].reshape(-1,1))) #TODO - check the hstack

    lidar_to_grid = pd.DataFrame(data=lidar_to_indices, columns=['x','y','z'])    
    lidar_height_diff = lidar_to_grid.groupby(['x','y'])
    lidar_height_diff = lidar_height_diff.agg({'z': ['max', 'min']}).reset_index().to_numpy()

    lidar_height_grid = lidar_height_diff[:,-2] - lidar_height_diff[:,-1]
    lidar_filtered = lidar_height_diff[np.where(lidar_height_grid > min_obst_height)]

    lidar_filtered[:,:2] = (lidar_filtered[:,:2] - max_range*5)/5

    lidar_filtered = lidar_filtered[:,:3]
    return lidar_filtered

def show_anim(dataset):
    frame_idx = 1
    fig = plt.figure(figsize=(9,7), dpi=160)
    grid = plt.GridSpec(5,5,hspace=0.4, wspace=0.4)
    vid_plot = fig.add_subplot(grid[0:2,0:3])
    scatter_plot = fig.add_subplot(grid[2:5,0:3])
    occupancy_plot = fig.add_subplot(grid[:,3:5])
    vid_plot.set_title('Scene Video')
    scatter_plot.set_title('Current LIDAR Sample')
    occupancy_plot.set_title('Occupancy Grid')

    grid_e = np.arange(xmin, xmax, 0.2)
    grid_n = np.arange(ymin, ymax,0.2)
    nn, ee = np.meshgrid(grid_e, grid_n)
    occupancy_grid = np.zeros(ee.shape)

    cam_images = []
    for image in dataset.cam2:
        cam_images.append(vid_plot.imshow(image))

    my_anim = animation.ArtistAnimation(fig, cam_images, interval=100, blit=True, repeat=1)

    imu_to_velo = dataset.calib.T_velo_imu
    velo_to_imu = np.linalg.inv(imu_to_velo)

    imgs_scatter = []
    relative_location_scatter = []
    trans_occupance_scatter = []
    self_loc = None
    self_loc_to_int = None
    for lidar_raw, oxts_data in zip(dataset.velo, dataset.oxts):
        start = time.time()
        #scatter raw_lidar
        lidar_raw = lidar_raw[:,:-1] #take only XYZ
        ones = np.ones((len(lidar_raw),1))
        lidar_homo = np.hstack((lidar_raw,ones))
        lidar_in_imu = np.matmul(velo_to_imu, lidar_homo.T).T

        scatter = scatter_plot.scatter(lidar_in_imu[:,0], lidar_in_imu[:,1], c='b', s=0.2, animated=True)
        imgs_scatter.append(scatter)

        occupied = get_obstacles_from_lidar(lidar_raw, max_dist_hit, 0.3, 2)

        lidar_range = range(0, len(occupied))
        ones = np.ones((len(occupied),1))
        occupied_homo = np.hstack((occupied,ones))
        occuppied_in_imu = np.matmul(velo_to_imu, occupied_homo.T).T
        
        relative_location = pm.geodetic2ned(oxts_data[0].lat, oxts_data[0].lon, oxts_data[0].alt , dataset.oxts[0][0].lat, dataset.oxts[0][0].lon, dataset.oxts[0][0].alt)# [xNorth,yEast,zDown]
        relative_location_to_int = [math.floor((relative_location[0]-ymin)*5), math.floor((relative_location[1]-xmin)*5)]
        if self_loc_to_int is None:
            # self_loc = np.array([relative_location])
            self_loc_to_int = np.array([relative_location_to_int])
            
        else:
            self_loc_to_int = np.vstack((self_loc_to_int, np.array([relative_location_to_int])))

        yaw = oxts_data[0].yaw
        trans_mat = [[1, 0, 0, relative_location[1]],
                    [0, 1, 0, relative_location[0]],
                    [0, 0, 1, relative_location[2]],
                    [0, 0, 0, 1]
                    ]
        rotation_mat = [[math.cos(yaw), -math.sin(yaw), 0, 0],
                    [math.sin(yaw), math.cos(yaw), 0, 0],
                    [0, 0, 1, 0],
                    [0, 0, 0, 1]
                    ]
        
        lidar_trans_rot = np.matmul(trans_mat, np.matmul(rotation_mat,occuppied_in_imu.T)).T
        
        lidar_hits_to_grid = np.zeros([ee.shape[0], ee.shape[1], 3], dtype=np.int32)
        lidar_hits_to_grid[np.array(np.floor(lidar_trans_rot[:,1]*5)- ymin*5,dtype=np.int32), np.array(np.floor(lidar_trans_rot[:,0]*5)- xmin*5,dtype=np.int32)] = [0,0,255]

        r_to_grid = np.sqrt((ee-relative_location[0])**2 + (nn-relative_location[1])**2) 
        angle_to_grid = np.arctan2((ee-relative_location[0]),(nn-relative_location[1]))
        
        r_to_obst = np.sqrt((lidar_trans_rot[:,0]-relative_location[1])**2 + (lidar_trans_rot[:,1]-relative_location[0])**2)
        angle_to_obst = np.arctan2(lidar_trans_rot[:,1]-relative_location[0],lidar_trans_rot[:,0]-relative_location[1])

        diff_grid = np.ones(angle_to_grid.shape)*7 #larger than 2*pi, (larger than max_val)
        relevant_obst_range = np.zeros(angle_to_grid.shape)
        
        #loop over all obstacles, for each grid cell, find the obstacle with the closest angle from car
        for i, obst_angle in enumerate(angle_to_obst):
            change_indices = np.abs(angle_to_grid-obst_angle) < diff_grid
            diff_grid[change_indices] = np.abs(angle_to_grid-obst_angle)[change_indices]
            relevant_obst_range[change_indices] = r_to_obst[i]

        temp_grid = np.zeros(angle_to_grid.shape)
        first_con   = r_to_grid >= (np.minimum(max_dist_hit, relevant_obst_range) + alpha/2)
        second_con  = (first_con == False) & (relevant_obst_range < max_dist_hit) & (np.abs(r_to_grid - relevant_obst_range) < alpha/2)
        third_cond  = (first_con == False) & (second_con == False) & (r_to_grid <= (relevant_obst_range - alpha/2))

        temp_grid[first_con] = 0
        temp_grid[second_con] = hit_prob_logit
        temp_grid[third_cond] = miss_prob_logit

        occupancy_grid = occupancy_grid + temp_grid
        occupancy_grid = np.minimum(occupancy_grid, occ_sat_logit)  #clip to max saturation
        occupancy_grid = np.maximum(occupancy_grid, free_sat_logit) #clip to min saturation

        occupancy_grid_rgb = np.ones([occupancy_grid.shape[0], occupancy_grid.shape[1], 3], dtype=np.int32)*255 #unknown
        occupancy_grid_rgb[occupancy_grid>occ_thr_logit] = [0,0,0]                                              #occupied
        occupancy_grid_rgb[occupancy_grid<free_thr_logit] = [0,0,255]                                           #free
        #adding self location        
        occupancy_grid_rgb[self_loc_to_int[:,0], self_loc_to_int[:,1]] = [255,0,0]
        occupancy_grid_rgb[self_loc_to_int[:,0]+1, self_loc_to_int[:,1]] = [255,0,0]
        occupancy_grid_rgb[self_loc_to_int[:,0], self_loc_to_int[:,1]+1] = [255,0,0]
        occupancy_grid_rgb[self_loc_to_int[:,0]+1, self_loc_to_int[:,1]+1] = [255,0,0]
        
        trans_scatter = occupancy_plot.imshow(lidar_hits_to_grid, animated=True, origin='lower', extent=[xmin,xmax,ymin,ymax])
        # trans_scatter = occupancy_plot.imshow(occupancy_grid_rgb, animated=True, origin='lower', extent=[xmin,xmax,ymin,ymax])
        trans_occupance_scatter.append(trans_scatter)
        end = time.time()
        print("finished frame "+ str(frame_idx)+" out of "+str(num_of_frames) + f". took {end-start} seconds")
        frame_idx = frame_idx + 1

             
    frames = []
    for img, scan, occupancy in zip(cam_images, imgs_scatter, trans_occupance_scatter):
        frames.append([img, scan, occupancy])

    my_anim = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat=1)


    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    my_anim.save('im_09_04_temp.mp4', writer=writer)

    plt.show()


if __name__ == "__main__":
    basedir = '/Users/yaronaloni/git/AutonomousRobots/'
    date = '2011_09_26'
    drive = '0005'

    dataset = pykitti.raw(basedir, date, drive, frames=range(start_frame, start_frame+num_of_frames, 1))

    show_anim(dataset)