
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np

def angle_diff(angle1, angle2):
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

def error_ellipse(position, sigma):

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
    error_ellipse = Ellipse(xy=[position[0],position[1]], width=width, height=height, angle=angle/np.pi*180)
    error_ellipse.set_alpha(0.25)

    return error_ellipse

def plot_state(particles, landmarks, GT_locations, running_best_particle, dec_ave):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and 
    # estimated mean landmark positions and covariances.

    draw_mean_landmark_poses = False

    map_limits = [-7, 17,-7, 17]
    
    #particle positions
    xs = []
    ys = []

    #landmark mean positions
    lxs = []
    lys = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])
        
    # ground truth landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i][0])
        ly.append(landmarks[i][1])

    # best particle
    estimated = best_particle(particles)
    robot_x = estimated['x']
    robot_y = estimated['y']
    robot_theta = estimated['theta']

    # estimated traveled path of best particle
    hist = estimated['history']
    hx = []
    hy = []

    for pos in hist:
        hx.append(pos[0])
        hy.append(pos[1])

    x_running_best = []
    y_running_best = []
    for loc in running_best_particle:
        x_running_best.append(loc[0])
        y_running_best.append(loc[1])

    x_GT = []
    y_GT = []
    for loc in GT_locations:
        x_GT.append(loc[0])
        y_GT.append(loc[1])

    x_ave = []
    y_ave = []
    for loc in dec_ave:
        x_ave.append(loc[0])
        y_ave.append(loc[1])

    # plot filter state
    plt.clf()

    plt.title(f'Particle Filter. frame {len(hx)}')

    #particles
    plt.plot(xs, ys, 'r.')
    
    if draw_mean_landmark_poses:
        # estimated mean landmark positions of each particle
        plt.plot(lxs, lys, 'b.')

    # estimated traveled path of best particle
    plt.plot(hx, hy, 'r-', label='current_best')

    #plot GT
    plt.plot(x_GT, y_GT, 'k-', label='GT')

    # #plot running_best_particle
    # plt.plot(x_running_best, y_running_best, 'g-', label='running_best')

    #plot dec_ave
    if(len(x_ave) > 1):
        plt.plot(x_ave[1:], y_ave[1:], 'b-', label='declined_average')

    # true landmark positions
    plt.plot(lx, ly, 'b+',markersize=10)

    # draw pose of best particle
    plt.quiver(robot_x, robot_y, np.cos(robot_theta), np.sin(robot_theta), angles='xy',scale_units='xy')
    
    plt.axis(map_limits)
    plt.legend(loc='upper right')
    plt.pause(0.0001)

def best_particle(particles):
    #find particle with highest weight 
    highest_weight = 0
    best_particle = None
    for particle in particles:
        if particle['weight'] > highest_weight:
            best_particle = particle
            highest_weight = particle['weight']

    return best_particle

def normalize_angle(angle):
    while(angle <= -np.pi or angle > np.pi):
        if (angle>np.pi):
            angle-=2*np.pi
        else:
            angle+=2*np.pi
    return angle

def get_weight(diff):
    # c = np.math.sqrt(np.linalg.det(2*np.pi*Q))
    exp = -0.5*diff*diff
    return (np.exp(exp))