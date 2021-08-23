import matplotlib; matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np
import cv2
import os
from matplotlib.patches import Ellipse
import warnings
from Kalman import *
from KittiLoader import KittiLoader

warnings.simplefilter("ignore")

def draw_cov_ellipse(ax, position, sigma):
    covariance = sigma[0:2,0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)
    #get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:,max_ind]
    max_eigval = eigenvals[max_ind]
    #get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:,min_ind]
    min_eigval = eigenvals[min_ind]

    #chi-square value for sigma confidence interval
    chisquare_scale = 2.2789  

    #calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale*max_eigval)
    height = 2 * np.sqrt(chisquare_scale*min_eigval)
    angle = np.arctan2(max_eigvec[1],max_eigvec[0])

    #generate covariance ellipse
    error_ellipse = Ellipse(xy=[position[0],position[1]], 
                        width=width, height=height, 
                        angle=angle/np.pi*180, edgecolor='red',
                        facecolor='none', label='location_std')
    error_ellipse.set_alpha(0.25)
    return ax.add_patch(error_ellipse)

def animate_kalman_filter(filename, noisy_EN, kalman_xy, kalman_sig_xy, kalman_DR_xy, GT_xy):
    vid = cv2.VideoWriter(f'{filename}.mp4',cv2.VideoWriter_fourcc(*'mp4v'), 50, (800,600))
    for i in range(noisy_EN.shape[0]):
        if (i%50 ==0):
            print(f'creating animation. frame {i}/{noisy_EN.shape[0]}')
        plt.close()
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(1,1,1)
        ax.plot(GT_xy[:i+1,0], GT_xy[:i+1,1], c='k', label='GT')
        ax.scatter(noisy_EN[:i+1,0],noisy_EN[:i+1,1], c='b', s=0.1, label='measurements')
        ax.plot(kalman_DR_xy[:i+1,0], kalman_DR_xy[:i+1,1], c='r', linestyle='dashed', label='dead_reckoning')
        ax.plot(kalman_xy[:i+1,0], kalman_xy[:i+1,1], c='g', linestyle='dashed', label='kalman_filter')
        draw_cov_ellipse(ax, kalman_xy[i,:], kalman_sig_xy[i,0:3:2,0:3:2])
        plt.title(f'Kalman filter. frame={i}')
        plt.xlabel('east (m)')
        plt.ylabel('north (m)')
        plt.xlim((np.min(kalman_xy[:i+1,0]-2), np.max(kalman_xy[:i+1,0]+2)))
        plt.ylim((np.min(kalman_xy[:i+1,1]-2), np.max(kalman_xy[:i+1,1]+2)))
        plt.legend(loc='upper right')
        fig.canvas.draw()
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        vid.write(cv2.cvtColor(data,cv2.COLOR_RGB2BGR))
    vid.release()
    plt.savefig(f'{filename}.png')

def plot_GT(gt_locations_LLA, gt_locations_EN):
    plt.clf()
    plt.plot(gt_locations_LLA[:,1], gt_locations_LLA[:,0], c='k')
    plt.title('GT GPS location in LLA')
    plt.xlabel('east')
    plt.ylabel('north')
    plt.savefig('GT_LLA.png')

    plt.clf()
    plt.plot(gt_locations_EN[:,0], gt_locations_EN[:,1], c='k')
    plt.title('GT GPS location in ENU (car coordinates)')
    plt.xlabel('east (m)')
    plt.ylabel('north (m)')
    plt.savefig('GT_ENU.png')

def plot_noisy_locations(gt_locations_EN, noisy_locations):
    plt.clf()
    plt.plot(gt_locations_EN[:,0], gt_locations_EN[:,1], c='k', label='GT')
    plt.scatter(noisy_locations[:,0], noisy_locations[:,1], c='b', s=0.1, label='measurements')
    plt.title('GT GPS location in ENU (car coordinates)')
    plt.xlabel('east (m)')
    plt.ylabel('north (m)')
    plt.title(f'GPS vs noisy GPS')
    plt.legend(loc='upper right')
    plt.savefig('noisy_locations.png')

def run_kalman_filter_scenario(kitti, sig_x, sig_y, gt_locations_EN, sig_r, plot_noise = 0):
    noisy_EN = kitti.get_noisy_EN_locations(sig_x,sig_y)
    if(plot_noise):
        plot_noisy_locations(gt_locations_EN, noisy_EN)
    plt.clf()

    kalman = Kalman(noisy_EN, np.diag([sig_x**2,sig_y**2]), np.diag([10,10,10,10]), kitti.timediffs_start_0, sig_r)
    kalman.run()
    kalman_xy = kalman.get_xy()
    x_std = np.sqrt(kalman.get_pred_cov_list())[:,0,0]
    y_std = np.sqrt(kalman.get_pred_cov_list())[:,1,1]
    
    kalman_DR = Kalman(noisy_EN, np.diag([sig_x**2,sig_y**2]), np.diag([10,10,10,10]), kitti.timediffs_start_0, sig_r)
    kalman_DR.run(dead_reckoning=1)
    kalman_DR_xy = kalman_DR.get_xy()
    
    errors = kalman_xy - gt_locations_EN
    maxE = np.max(np.sum(errors, axis=1)[100:])
    RMSE = np.sqrt(np.sum(np.power(errors[100:,:], 2))/(errors.shape[0]-100))

    animate_kalman_filter(f'Kalman_filter_{sig_x}_{sig_y}', noisy_EN, kalman_xy, np.array(kalman.get_pred_cov_list()), kalman_DR_xy, gt_locations_EN)
    
    fig = plt.figure()
    plt.close()
    plt.title('x_y errors')
    plt.subplot(2, 1, 1)
    plt.scatter(range(len(x_std)), errors[:,0], s=0.2)
    plt.plot(range(len(x_std)), x_std, c='g', label='_nolegend_')
    plt.plot(range(len(x_std)), -x_std, c='g',  label='_nolegend_')
    plt.ylabel('err_x')
    plt.legend([r'$x-x_{gt}$'], loc='upper right')
    plt.subplot(2, 1, 2)
    plt.scatter(range(len(y_std)), errors[:,1], s=0.2)
    plt.plot(range(len(y_std)), y_std, c='g', label='_nolegend_')
    plt.plot(range(len(y_std)), -y_std, c='g',  label='_nolegend_')
    plt.xlabel('frame')
    plt.ylabel('err_y')

    plt.legend([r'$y-y_{gt}$'], loc='upper right')
    plt.savefig(f'Kalman_filter_x_y_errors_{sig_x}_{sig_y}.png')

    maxE = np.max(np.sum(np.abs(errors), axis=1)[100:])
    RMSE = np.sqrt(np.sum(np.power(errors[100:,:], 2))/(errors.shape[0]-100))
    print(f'maxE = {maxE:.2f}')
    print(f'RMSE = {RMSE:.2f}')

def run_EKF_scenario(kitti, sig_x, sig_y, sig_w, sig_v, gt_locations_EN, plot_yaw_velo = 0):
    noisy_EN = kitti.get_noisy_EN_locations(sig_x,sig_y)
    noisy_w = kitti.get_noisy_yaw_rate(sig_w)
    noisy_v = kitti.get_noisy_velocities(sig_v)

    plt.clf()

    if (plot_yaw_velo):
        fig = plt.figure(figsize=(2,5))
        plt.close()
        plt.subplot(3, 1, 1)
        plt.title(r'yaw values')
        plt.subplots_adjust(wspace=0.5, hspace=0.5)
        
        yaws = kitti.yaws[:,0]
        yaws[yaws>np.pi] = yaws[yaws>np.pi]-2*np.pi
        yaws[yaws<-np.pi] = yaws[yaws<-np.pi]+2*np.pi

        plt.plot(range(len(kitti.yaws[:,0])), yaws, c='g', label='_nolegend_')
        plt.ylabel('yaw [rads]')    
        plt.subplot(3, 1, 2)
        plt.title(r'yaw_rate values')
        plt.plot(range(len(kitti.yaws[:,1])), kitti.yaws[:,1], c='g', label='_nolegend_')
        plt.ylabel('yaw_rate [rads/s]')    
        plt.subplot(3, 1, 3)
        plt.title(r'forward_velocity')
        plt.plot(range(len(kitti.velocities[:])), kitti.velocities[:], c='g', label='_nolegend_')
        plt.ylabel('forward_velocity [m/sec]')   
        plt.savefig(f'yaw_vals_values.png')

    plt.clf()
    
    ext_kalman = ExtKalman(noisy_EN, np.diag([sig_x**2, sig_y**2]), noisy_w, sig_w, noisy_v, sig_v, np.diag([10,10, 10]), kitti.timediffs_start_0)
    ext_kalman.run()

    ext_kalman_xy = ext_kalman.get_xy()
    ext_kalman_theta = ext_kalman.get_yaws()
    x_std = np.sqrt(ext_kalman.get_pred_cov_list())[:,0,0]
    y_std = np.sqrt(ext_kalman.get_pred_cov_list())[:,1,1]
    theta_std = np.sqrt(ext_kalman.get_pred_cov_list())[:,2,2]
    
    ext_kalman_DR = ExtKalman(noisy_EN, np.diag([sig_x**2, sig_y**2]), noisy_w, sig_w, noisy_v, sig_v, np.diag([10,10, 10]), kitti.timediffs_start_0)
    ext_kalman_DR.run(dead_reckoning=1)
    ext_kalman_DR_xy = ext_kalman_DR.get_xy()
    
    errors = ext_kalman_xy - gt_locations_EN
    theta_errors = ext_kalman_theta - kitti.yaws[:,0]
    theta_errors[theta_errors>np.pi] = theta_errors[theta_errors>np.pi]-2*np.pi
    theta_errors[theta_errors<-np.pi] = theta_errors[theta_errors<-np.pi]+2*np.pi
    maxE = np.max(np.sum(errors, axis=1)[100:])
    RMSE = np.sqrt(np.sum(np.power(errors[100:,:], 2))/(errors.shape[0]-100))

    animate_kalman_filter(f'EKF_{sig_w}_{sig_v}', noisy_EN, ext_kalman_xy, np.array(ext_kalman.get_pred_cov_list()), ext_kalman_DR_xy, gt_locations_EN)
    
    fig = plt.figure()
    plt.close()
    plt.subplot(3, 1, 1)
    plt.title(r'x_y_$\theta$ errors')
    plt.scatter(range(len(x_std)), errors[:,0], s=0.2)
    plt.plot(range(len(x_std)), x_std, c='g', label='_nolegend_')
    plt.plot(range(len(x_std)), -x_std, c='g',  label='_nolegend_')
    plt.ylabel('err_x')    
    plt.ylim((-2.5,2.5))
    plt.legend([r'$x-x_{gt}$'], loc='upper right')
    plt.subplot(3, 1, 2)
    plt.scatter(range(len(y_std)), errors[:,1], s=0.2)
    plt.plot(range(len(y_std)), y_std, c='g', label='_nolegend_')
    plt.plot(range(len(y_std)), -y_std, c='g',  label='_nolegend_')
    plt.xlabel('frame')
    plt.ylabel('err_y')
    plt.ylim((-2.5,2.5))
    plt.legend([r'$y-y_{gt}$'], loc='upper right')
    plt.subplot(3, 1, 3)
    plt.scatter(range(len(theta_std)), theta_errors, s=0.2)
    plt.plot(range(len(theta_std)), theta_std, c='g', label='_nolegend_')
    plt.plot(range(len(theta_std)), -theta_std, c='g',  label='_nolegend_')
    plt.ylabel(r'err_$\theta$')
    plt.legend([r'$\theta-\theta_{gt}$'], loc='upper right')
    plt.ylim((-0.3,0.3))
    plt.savefig(f'EKF_x_y_errors_{sig_w}_{sig_v}.png')

    maxE = np.max(np.sum(np.abs(errors), axis=1)[100:])
    RMSE = np.sqrt(np.sum(np.power(errors[100:,:], 2))/(errors.shape[0]-100))
    print(f'maxE = {maxE:.2f}')
    print(f'RMSE = {RMSE:.2f}')

if __name__ == "__main__":
    kitti = KittiLoader('/Users/yaronaloni/git/AutonomousRobots/', '2011_09_30', '0033')
    gt_locations_EN = kitti.location_ENU[:,:2]
    gt_locations_LLA = kitti.coords_lla[:,:2]
    sig_w = 0.2
    sig_v = 0
    sig_r = 1.6

    plot_GT(gt_locations_LLA, gt_locations_EN)

    run_kalman_filter_scenario(kitti, sig_x=3, sig_y=3, gt_locations_EN=gt_locations_EN, sig_r=sig_r, plot_noise=1)
    run_kalman_filter_scenario(kitti, sig_x=5, sig_y=5, gt_locations_EN=gt_locations_EN, sig_r=sig_r)
    run_kalman_filter_scenario(kitti, sig_x=10, sig_y=10, gt_locations_EN=gt_locations_EN, sig_r=sig_r)


    run_EKF_scenario(kitti, sig_x=3, sig_y=3, sig_w=0, sig_v=0, gt_locations_EN=gt_locations_EN, plot_yaw_velo=1)
    run_EKF_scenario(kitti, sig_x=3, sig_y=3, sig_w=0.2, sig_v=0, gt_locations_EN=gt_locations_EN)
    run_EKF_scenario(kitti, sig_x=3, sig_y=3, sig_w=0, sig_v=2, gt_locations_EN=gt_locations_EN)
