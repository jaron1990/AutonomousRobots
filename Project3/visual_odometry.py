import matplotlib
from numpy.lib.function_base import append; matplotlib.use('agg')
import itertools
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import pykitti
import cv2 as cv
import warnings
import copy
warnings.simplefilter("ignore")

#gets the location value in order to make the animation
def animate(name, GT_location_list, locations_list, matching_frames):
    GT_location_list = np.array(GT_location_list)
    locations_list = np.array(locations_list)
    locations_list[:,2] = locations_list[:,2]
    matching_frames.append(matching_frames[-1])
    vid = cv.VideoWriter(f'{name}.mp4',cv.VideoWriter_fourcc(*'mp4v'), 50, (800,600))
    for i in range(GT_location_list.shape[0]):
        if (i%50 ==0):
            print(f'creating animation. frame {i}/{GT_location_list.shape[0]}')
        plt.close()
        fig = plt.figure(figsize=(8,6))

        ay = fig.add_subplot(2,1,1)
        plt.title('frames with matching points')
        plt.imshow(matching_frames[i])

        ax = fig.add_subplot(2,1,2)
        ax.plot(GT_location_list[:i+1,0], GT_location_list[:i+1,2], c='k', label='GT')
        ax.plot(locations_list[:i+1,0], locations_list[:i+1,2], c='g', linestyle='dashed', label='video_odometry')
        plt.title(f'Video Odometry. frame={i}')
        min_x = min(np.min(GT_location_list[:i+1,0]), np.min(locations_list[:i+1,0])) - 2
        min_z = min(np.min(GT_location_list[:i+1,2]), np.min(locations_list[:i+1,2])) - 2
        max_x = max(np.max(GT_location_list[:i+1,0]), np.max(locations_list[:i+1,0])) + 2
        max_z = max(np.max(GT_location_list[:i+1,2]), np.max(locations_list[:i+1,2])) + 2
        
        plt.xlim(min_x, max_x)
        plt.ylim(min_z, max_z)
        plt.legend(loc='upper right')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        vid.write(cv.cvtColor(data,cv.COLOR_RGB2BGR))
    vid.release()
    plt.savefig(f'{name}.png')

def get_keypoints_matches(prev_frm, cur_frm):
    # Initiate SIFT detector
    descriptor = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    img1 = np.array(prev_frm[0])
    img2 = np.array(cur_frm[0])
    kp1, des1 = descriptor.detectAndCompute(img1,None)
    kp2, des2 = descriptor.detectAndCompute(img2,None)

    # create BFMatcher object
    bf = cv.BFMatcher()
    # Match descriptors.
    matches = bf.knnMatch(des1,des2,k=2)

    # Apply ratio test
    good, keypoints_cur, keypoints_prev = [], [], []

    for m,n in matches:
        if m.distance < 0.8*n.distance:
            good.append([m])
            keypoints_prev.append(kp1[m.queryIdx].pt)
            keypoints_cur.append(kp2[m.trainIdx].pt)
    
    # img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None)
    img0_kp = cv.drawKeypoints(img1, kp1, img2)
    # plt.imshow(img0_kp),plt.show()

    return keypoints_prev, keypoints_cur, img0_kp

def visual_odometry_main(sequence, live_scaling):
    # Change this to the directory where you store KITTI data
    basedir = '/Users/yaronaloni/git/AutonomousRobots55/odometry/dataset'

    # Load the data. Optionally, specify the frame range to load.
    # dataset = pykitti.odometry(basedir, sequence, frames=range(2300, 2600, 1))
    dataset = pykitti.odometry(basedir, sequence)

    fig = plt.figure()
    prev_frm=None

    camera_matrix = dataset.calib.K_cam0

    current_pose = np.eye(4,4)
    matching_frames = []
    i=0
    #loop over all frames
    for cur_frm, GT_pose in zip(dataset.gray, dataset.poses):
        if (i%50==0): print(f"starting frame {i}/{len(dataset.poses)}")
        i+=1
        #if this is not the first frame - calculate movement. else - continue
        if (prev_frm!=None):
            keypoints_prev, keypoints_cur, image = get_keypoints_matches(prev_frm, cur_frm)
            matching_frames.append(image)
            
            keypoints_prev = np.int32(keypoints_prev)
            keypoints_cur = np.int32(keypoints_cur)
            
            E, mask = cv.findEssentialMat(keypoints_prev, keypoints_cur, camera_matrix, prob = 0.999, threshold = 1)

            ret, R, t, mask = cv.recoverPose(E, keypoints_prev, keypoints_cur, camera_matrix)
            
            if(live_scaling):
                true_step_size = np.linalg.norm(GT_pose[:3,3].T - previous_GT_t)
                t*=true_step_size
            
            T = np.hstack([R,t])
            T = np.vstack([T,[0,0,0,1]])
            current_pose = current_pose.dot(np.linalg.inv(T))
            locations_list = np.vstack([locations_list, current_pose[:3,3]])
            GT_location_list = np.vstack([GT_location_list, GT_pose[:3,3]])
            previous_GT_t = np.array(GT_pose[:3,3])
        else:
            locations_list = np.array(current_pose[:3,3])
            GT_location_list = np.array(GT_pose[:3,3])
            previous_GT_t = np.array(GT_pose[:3,3])

        prev_frm=cur_frm

    #live scaling is calculated in runtime
    if (not live_scaling):
        scale = np.linalg.norm(GT_location_list[-1]) / np.linalg.norm(locations_list[-1])
        locations_list *= scale
        

    animate(f'visual_odometry_drive_{sequence}_{("live_scaling" if live_scaling else "global_scaling")}', GT_location_list, locations_list, matching_frames)

    GT_location_list = np.array(GT_location_list)
    locations_list = np.array(locations_list)
    RMSE = np.sqrt(((GT_location_list[:,0]-locations_list[:,0]) ** 2).mean() + ((GT_location_list[:,1]-locations_list[:,1]) ** 2).mean())
    max_error = np.max(np.linalg.norm(GT_location_list-locations_list, axis=1))

    print(f'*******')
    print(f'VO for drive {sequence} with {("live_scaling" if live_scaling else "global_scaling")} errors:')
    print(f'RMSE={RMSE:.2f}')
    print(f'max_error={max_error:.2f}')
    print()

if __name__ == "__main__":
    live_scaling=False

    for sequence in ["04","05"]:
        for live_scaling in [True, False]:
            visual_odometry_main(sequence, live_scaling)
