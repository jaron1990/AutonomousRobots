import matplotlib.pyplot as plt 
from matplotlib import animation
import pykitti
import numpy as np
import math
import time
import pandas as pd
import pymap3d as pm
from sklearn.neighbors import KDTree

global_pose=True
run_with_icp = False

min_occ_points_for_icp=500
ICP_iter = 200
ICP_err = 0.05
ICP_ths = 1.5
xmin = -30
ymin = -83
xmax = 61
ymax = 30
alpha = 0.2
max_dist_hit = 30
hit_prob = 0.7
miss_prob = 0.4
occ_thr = 0.75
free_thr = 0.25
occ_sat = 0.95
free_sat = 0.05

hit_prob_logit  = math.log(hit_prob/(1-hit_prob))
miss_prob_logit = math.log(miss_prob/(1-miss_prob))
occ_thr_logit   = math.log(occ_thr/(1-occ_thr))
free_thr_logit  = math.log(free_thr/(1-free_thr))
occ_sat_logit   = math.log(occ_sat/(1-occ_sat))
free_sat_logit  = math.log(free_sat/(1-free_sat))

grid_e = np.arange(xmin, xmax, alpha)
grid_n = np.arange(ymin, ymax, alpha)
nn, ee = np.meshgrid(grid_e, grid_n)

### filter lidar hits longer than max_range distance from velo
def remove_long_hits(input_lidar, max_range):
    valid_indices = (input_lidar[:,0]**2 + input_lidar[:,1]**2 < max_range**2)
    short_hits = input_lidar[valid_indices]
    return short_hits

### filter grid cells containing hits with at least min_obst_height difference between min and max height values
def filter_by_height(lidar_data, max_range, min_obst_height):
    lidar_to_grid = np.array(lidar_data[:,:2]/alpha + max_range/alpha, dtype=np.int32) 
    lidar_to_grid = np.hstack((lidar_to_grid, lidar_data[:,2].reshape(-1,1)))

    lidar_to_grid_pd = pd.DataFrame(data=lidar_to_grid, columns=['x','y','z'])    
    lidar_grid_unique = lidar_to_grid_pd.groupby(['x','y'])
    lidar_grid_unique = lidar_grid_unique.agg({'z': ['max', 'min']}).reset_index().to_numpy()

    lidar_height_diff = lidar_grid_unique[:,-2] - lidar_grid_unique[:,-1]
    lidar_filtered = lidar_grid_unique[np.where(lidar_height_diff > min_obst_height)]

    lidar_filtered[:,:2] = (lidar_filtered[:,:2] - max_range/alpha)*alpha + alpha/2
    lidar_filtered = lidar_filtered[:,:3]
    return lidar_filtered

### raw lidar to filtered lidar
def get_obstacles_from_lidar(raw_lidar, max_range, min_obst_height):
    lidar_short = remove_long_hits(raw_lidar, max_range)
    lidar_filtered = filter_by_height(lidar_short, max_range, min_obst_height)
    return lidar_filtered

### move from lidar origin to imu origin
def lidar_to_imu(lidar_raw, velo_to_imu):
    ones = np.ones((len(lidar_raw),1))
    lidar_homo = np.hstack((lidar_raw,ones))
    lidar_in_imu = np.matmul(velo_to_imu, lidar_homo.T).T
    return lidar_in_imu

### get relative location of the car based on GPS    
def get_relative_location(oxts_data):
    return pm.geodetic2ned(oxts_data[0].lat, oxts_data[0].lon, oxts_data[0].alt , dataset.oxts[0][0].lat, dataset.oxts[0][0].lon, dataset.oxts[0][0].alt)# [xNorth,yEast,zDown]

### change relative_location of the car based on the velocities
def change_relative_location(relative_location, oxts_data, ts, last_ts):
    return np.array([oxts_data[0].vn, oxts_data[0].ve, oxts_data[0].vu])*(ts-last_ts).microseconds/(10**6) + relative_location

def append_location_to_grid_path(path_in_grid, relative_location):
    relative_location_to_int = [math.floor((relative_location[0]-ymin)/alpha), math.floor((relative_location[1]-xmin)/alpha)]
    if path_in_grid is None:
        path_in_grid = np.array([relative_location_to_int])
    else:
        path_in_grid = np.vstack((path_in_grid, np.array([relative_location_to_int])))
    return path_in_grid

### convert imu_origin coordinates to ENU originated in the first car's location
def imu_to_global(imu_data, oxts_data, relative_location):
    yaw = oxts_data[0].yaw
    trans_mat = [[1, 0, 0, relative_location[1]],
                [0, 1, 0, relative_location[0]],
                [0, 0, 1, -relative_location[2]],
                [0, 0, 0, 1]
                ]
    rotation_mat = [[math.cos(yaw), -math.sin(yaw), 0, 0],
                [math.sin(yaw), math.cos(yaw), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
                ]
    
    points_global = np.matmul(trans_mat, np.matmul(rotation_mat,imu_data.T)).T
    return points_global

def get_occupied_from_grid(occupancy_grid, occ_thr_logit):
    return np.array(np.where(occupancy_grid>occ_thr_logit)).T

def ICP(X, P, ICP_iter, ICP_err): #X is reference, P is target
    err = float(10E5)
    R = np.identity(2)
    t = np.zeros(2)
    P_init = P

    for i in range(ICP_iter):
        if (err<ICP_err):
            break
        
        kd_tree = KDTree(P)
        
        d, index = kd_tree.query(X, k=1)
        P_i = P[index]

        miu_X = X.mean(axis=0)
        miu_P_i = P_i.mean(axis=0)

        X_pr = X - miu_X
        P_i_pr = P_i - miu_P_i

        U, sig, Vt = np.linalg.svd(np.matmul(X_pr.squeeze().T, P_i_pr.squeeze()),full_matrices=True)
        
        R = np.matmul(U,Vt)
        t = miu_X - R.dot(miu_P_i.T).squeeze()

        P = R.dot(P.T).T + t
        err = (np.linalg.norm(P_i_pr, axis=1)**2).sum() + (np.linalg.norm(X_pr)**2).sum() - 2 * sig.sum()
        err = err / P_i_pr.shape[0]
    t = P.mean(axis=0) - P_init.mean(axis=0)
    if(err > ICP_ths):
        P=P_init
        t=t*0
    return P, t

def show_anim(dataset):   
    num_of_frames = len(dataset.timestamps)
    frame_idx = 1

    fig = plt.figure(figsize=(9,7), dpi=160)
    grid = plt.GridSpec(5,5,hspace=0.4, wspace=0.4)
    vid_plot = fig.add_subplot(grid[0:2,0:3])
    scatter_plot = fig.add_subplot(grid[2:5,0:3])
    occupancy_plot = fig.add_subplot(grid[:,3:5])
    vid_plot.set_title('Scene Video')
    scatter_plot.set_title('Current LIDAR Sample')
    occupancy_plot.set_title('Occupancy Grid')
    occupancy_grid = np.zeros(ee.shape)

    cam_images = []
    for image in dataset.cam2:
        cam_images.append(vid_plot.imshow(image))

    my_anim = animation.ArtistAnimation(fig, cam_images, interval=100, blit=True, repeat=1)

    imu_to_velo = dataset.calib.T_velo_imu
    velo_to_imu = np.linalg.inv(imu_to_velo)

    imgs_scatter = []
    occupancy_grid_frames = []
    self_loc = None
    path_in_grid = None
    start = time.time()
    relative_location=[0,0,0]
    ts=dataset.timestamps[0]
    last_ts=dataset.timestamps[0]
    
    for timestamp, lidar_raw, oxts_data in zip(dataset.timestamps, dataset.velo, dataset.oxts):
        do=0
        #take only XYZ
        lidar_raw = lidar_raw[:,:-1] 
        #scatter raw_lidar
        lidar_in_imu = lidar_to_imu(lidar_raw, velo_to_imu)
        scatter = scatter_plot.scatter(lidar_in_imu[:,0], lidar_in_imu[:,1], c='b', s=0.2, animated=True)
        imgs_scatter.append(scatter)

        occupied = get_obstacles_from_lidar(lidar_raw, max_dist_hit, 0.3)
        occuppied_in_imu = lidar_to_imu(occupied, velo_to_imu)
        
        if(global_pose):
            relative_location = get_relative_location(oxts_data)
        else:
            ts = timestamp
            relative_location = change_relative_location(relative_location, oxts_data, ts, last_ts)
            last_ts = ts

        path_in_grid = append_location_to_grid_path(path_in_grid, relative_location)

        lidar_in_global = imu_to_global(occuppied_in_imu, oxts_data, relative_location)
        if(run_with_icp and not global_pose):
            if(np.sum(occupancy_grid>occ_thr_logit) > lidar_in_global.shape[0]*0.05):
                do = 1
                occupied_cells_indices = get_occupied_from_grid(occupancy_grid, occ_thr_logit)
                occupied_cells_global = np.array([(occupied_cells_indices[:,0])*alpha + ymin +alpha/2 , (occupied_cells_indices[:,1])*alpha + xmin +alpha/2]).T
                lidar_in_glob_NE = np.array([lidar_in_global[:,1],lidar_in_global[:,0]]).T
                occ_cells_filter_long = (occupied_cells_global[:,0] - relative_location[0])**2 + (occupied_cells_global[:,1] - relative_location[1])**2 < 30**2
                occupied_cells_global = occupied_cells_global[occ_cells_filter_long]
                lidar_in_glob_NE, t = ICP(occupied_cells_global, lidar_in_glob_NE, ICP_iter, ICP_err)
                relative_location[:2] = relative_location[:2] + t
                lidar_in_global = np.array([lidar_in_glob_NE[:,1],lidar_in_glob_NE[:,0]]).T

        dist_car_to_grid = np.sqrt((ee-relative_location[0])**2 + (nn-relative_location[1])**2) 
        angle_car_to_grid = np.arctan2((ee-relative_location[0]),(nn-relative_location[1]))
        
        dist_car_to_obst = np.sqrt((lidar_in_global[:,0]-relative_location[1])**2 + (lidar_in_global[:,1]-relative_location[0])**2)
        angle_car_to_obst = np.arctan2(lidar_in_global[:,1]-relative_location[0],lidar_in_global[:,0]-relative_location[1])

        diff_grid = np.ones(angle_car_to_grid.shape)*7 #larger than 2*pi, (larger than max_val)
        relevant_obst_range = np.zeros(angle_car_to_grid.shape)
        
        #loop over all obstacles, for each grid cell, find the obstacle with the closest angle from car
        for i, obst_angle in enumerate(angle_car_to_obst):
            change_indices = np.abs(angle_car_to_grid-obst_angle) < diff_grid
            diff_grid[change_indices] = np.abs(angle_car_to_grid-obst_angle)[change_indices]
            relevant_obst_range[change_indices] = dist_car_to_obst[i]

        temp_grid = np.zeros(angle_car_to_grid.shape)
        first_con   = dist_car_to_grid >= (relevant_obst_range + alpha/2)
        second_con  = (first_con == False) & (relevant_obst_range < max_dist_hit) & (np.abs(dist_car_to_grid - relevant_obst_range) < alpha/2)
        third_cond  = (first_con == False) & (second_con == False) & (dist_car_to_grid <= (relevant_obst_range - alpha/2))

        temp_grid[first_con] = 0
        temp_grid[second_con] = hit_prob_logit
        temp_grid[third_cond] = miss_prob_logit

        occupancy_grid = occupancy_grid + temp_grid
        occupancy_grid = np.minimum(occupancy_grid, occ_sat_logit)  #clip to max saturation
        occupancy_grid = np.maximum(occupancy_grid, free_sat_logit) #clip to min saturation

        occupancy_grid_rgb = np.ones([occupancy_grid.shape[0], occupancy_grid.shape[1], 3], dtype=np.int32)*190 #unknown
        occupancy_grid_rgb[occupancy_grid>occ_thr_logit] = [0,0,0]                                              #occupied
        occupancy_grid_rgb[occupancy_grid<free_thr_logit] = [80,80,255]                                           #free
        
        #adding self location. extended to 2x2 in order to be more visible in the plot
        occupancy_grid_rgb[path_in_grid[:,0], path_in_grid[:,1]] = [255,0,0]
        occupancy_grid_rgb[path_in_grid[:,0]+1, path_in_grid[:,1]] = [255,0,0]
        occupancy_grid_rgb[path_in_grid[:,0], path_in_grid[:,1]+1] = [255,0,0]
        occupancy_grid_rgb[path_in_grid[:,0]+1, path_in_grid[:,1]+1] = [255,0,0]
        
        occupancy_frame = occupancy_plot.imshow(occupancy_grid_rgb, animated=True, origin='lower', extent=[xmin,xmax,ymin,ymax])
        occupancy_grid_frames.append(occupancy_frame)

        end = time.time()
        print(f"finished frame {frame_idx}/{num_of_frames}. time remaining ~ {(float(end-start)*(num_of_frames-frame_idx)/frame_idx):.2f} seconds")
        frame_idx = frame_idx + 1

    frames = []
    for img, scan, occupancy in zip(cam_images, imgs_scatter, occupancy_grid_frames):
        frames.append([img, scan, occupancy])

    my_anim = animation.ArtistAnimation(fig, frames, interval=100, blit=True, repeat=1)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=10, metadata=dict(artist='Me'), bitrate=1800)
    my_anim.save(f'video_velocity_TEMP_{hit_prob:.1f}_{miss_prob:.1f}.mp4', writer=writer)
    plt.show()

    fig_png = plt.figure()
    plt.title('Occupancy Grid')
    plt.imshow(occupancy_grid_frames[-1].get_array(), animated=True, origin='lower', extent=[xmin,xmax,ymin,ymax])
    plt.savefig(f'occupancy_grid_velocity_TEMP_{hit_prob:.1f}_{miss_prob:.1f}.png')

if __name__ == "__main__":
    basedir = '/Users/yaronaloni/git/AutonomousRobots/'
    date = '2011_09_26'
    drive = '0005'
    dataset = pykitti.raw(basedir, date, drive)
    show_anim(dataset)