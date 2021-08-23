from numpy import random
from read_data import read_world, read_sensor_data
from misc_tools import *
import numpy as np
import math
import copy

#plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def get_dec_ave(particles):
    x = 0
    y = 0
    theta = 0
    for particle in particles:
        x += particle['x']*particle['weight']
        y += particle['y']*particle['weight']
        theta += particle['theta']*particle['weight']
    return [x,y,theta]


def get_new_pose(cur_x, cur_y, cur_theta, t, r1, r2):
    new_x = cur_x + t*np.cos(cur_theta+r1)
    new_y = cur_y + t*np.sin(cur_theta+r1)
    new_theta = normalize_angle(cur_theta + r1 + r2)
    return new_x, new_y, new_theta


def initialize_particles(num_particles, num_landmarks):
    #initialize particle at pose [0,0,0] with an empty map

    particles = []

    for i in range(num_particles):
        particle = dict()

        #initialize pose: at the beginning, robot is certain it is at [0,0,0]
        particle['x'] = 0
        particle['y'] = 0
        particle['theta'] = 0

        #initial weight
        particle['weight'] = 1.0 / num_particles
        
        #particle history aka all visited poses
        particle['history'] = []

        #initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            #initialize the landmark mean and covariance 
            landmark['mu'] = [0,0]
            landmark['sigma'] = np.zeros([2,2])
            landmark['observed'] = False

            landmarks[i+1] = landmark

        #add landmarks to particle
        particle['landmarks'] = landmarks

        #add particle to set
        particles.append(particle)

    return particles

def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise 

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    '''your code here'''
    sigma_rot1 = 0.01
    sigma_rot2 = 0.01
    sigma_trans = 0.02

    for particle in particles:
        delta_rot1_noisy = delta_rot1 + np.random.normal(0,sigma_rot1)
        delta_rot2_noisy = delta_rot2 + np.random.normal(0,sigma_rot2)
        delta_trans_noisy = delta_trans + np.random.normal(0,sigma_trans)

        particle['x'], particle['y'], particle['theta'] = get_new_pose(particle['x'], particle['y'], particle['theta'], delta_trans_noisy, delta_rot1_noisy, delta_rot2_noisy)
    '''***        ***'''

def measurement_model(particle, landmark):
    #Compute the expected measurement for a landmark
    #and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    #calculate expected range measurement
    meas_range_exp = np.sqrt( (lx - px)**2 + (ly - py)**2 )
    meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta
    meas_bearing_exp = normalize_angle(meas_bearing_exp)

    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian H of the measurement function h 
    #wrt the landmark location
    
    H = np.zeros((2,2))
    H[0,0] = (lx - px) / h[0]
    H[0,1] = (ly - py) / h[0]
    H[1,0] = (py - ly) / (h[0]**2)
    H[1,1] = (lx - px) / (h[0]**2)

    return h, H

def save_hist(particles):
    for particle in particles:
        px = particle['x']
        py = particle['y']

        particle['history'].append([px, py])


def eval_sensor_model(sensor_data, particles):
    #Correct landmark poses with a measurement and
    #calculate particle weight

    #sensor noise
    Q_t = np.array([[0.1, 0],\
                    [0, 0.1]])

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    #update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']

        px = particle['x']
        py = particle['y']
        ptheta = particle['theta'] 
        weights = []

        #loop over observed landmarks 
        for i in range(len(ids)):

            #current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]
            
            #measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            if not landmark['observed']:
                # landmark is observed for the first time
                
                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                '''your code here'''

                landmark['mu'][0] = px + meas_range*np.cos(meas_bearing + ptheta)
                landmark['mu'][1] = py + meas_range*np.sin(meas_bearing + ptheta)

                _, H = measurement_model(particle, landmark)

                landmark['sigma'] = np.linalg.inv(H).dot(Q_t).dot(np.linalg.inv(H).T)
                '''***        ***'''

                landmark['observed'] = True

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above. 
                # calculate particle weight: particle['weight'] = ...
                '''your code here'''
                z = np.array([meas_range, meas_bearing])
                h, H = measurement_model(particle, landmark)
                Q = H.dot(landmark['sigma']).dot(H.T)+Q_t
                K = landmark['sigma'].dot(H.T).dot(np.linalg.inv(Q))
                diff = z-h
                diff[1] = normalize_angle(diff[1])
                landmark['mu'] += K.dot(diff)
                landmark['sigma'] = (np.eye(landmark['sigma'].shape[0]) - K.dot(H)).dot(landmark['sigma'])
                weights.append(get_weight(Q, diff))
                '''***        ***'''

        if len(weights) > 0:
                particle['weight'] = sum(weights) / len(weights)

    #normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer

def plot_errors(dec_ave, running_best_particle, GT_locations, last_best_particle):
    plt.show()
    plt.savefig(f'fastslam.png')

    fig = plt.figure()
    plt.close()
    plt.subplot(2, 1, 1)
    plt.title(r'x_y errors')
    plt.scatter(range(dec_ave.shape[0]), GT_locations[:,0]-dec_ave[:,0], c='g', label='declined_average_error', s=0.2)
    # plt.scatter(range(running_best_particle.shape[0]), GT_locations[:,0]-running_best_particle[:,0], c='b', label='running_best_particle_error', s=0.2)
    plt.scatter(range(last_best_particle.shape[0]), GT_locations[:,0]-last_best_particle[:,0], c='r', label='last_best_particle_error', s=0.2)
    plt.ylabel('err_x')
    plt.legend(loc='upper right')
    plt.subplot(2, 1, 2)
    plt.scatter(range(dec_ave.shape[0]), GT_locations[:,1]-dec_ave[:,1], c='g', label='declined_average_error', s=0.2)
    # plt.scatter(range(running_best_particle.shape[0]), GT_locations[:,1]-running_best_particle[:,1], c='b', label='running_best_particle_error', s=0.2)
    plt.scatter(range(last_best_particle.shape[0]), GT_locations[:,1]-last_best_particle[:,1], c='r', label='last_best_particle_error', s=0.2)
    plt.xlabel('frame')
    plt.ylabel('err_y')
    plt.legend(loc='upper right')
    plt.savefig(f'fast_slam_err.png')

def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle 
    # weights.

    new_particles = []

    # '''your code here'''

    r = np.random.rand() / len(particles)
    c = particles[0]['weight']
    i = 0

    for j in range(len(particles)):
        U = r + j / len(particles)
        while U > c:
            i+=1
            c+=particles[i]['weight']
        new_particle = copy.deepcopy(particles[i])
        new_particles.append(new_particle)
    # '''***        ***'''

    # hint: To copy a particle from particles to the new_particles
    # list, first make a copy:
    # new_particle = copy.deepcopy(particles[i])
    # ...
    # new_particles.append(new_particle)

    return new_particles

def main():

    print ("Reading landmark positions")
    landmarks = read_world("Project3/data/world.dat")

    print ("Reading sensor data")
    sensor_readings = read_sensor_data("Project3/data/sensor_data.dat")

    loops = 1
    # num_particles_list = [1,3, 5, 10, 20, 40, 60, 80, 100]
    num_particles_list = [20]
    RMSE_ave_list = []
    max_error_ave_list = []
    for num_particles in num_particles_list:
        RMSE_ave = 0
        RMSE_best = 0
        max_error_ave = 0
        for i in range(loops):
            # num_particles = 100
            num_landmarks = len(landmarks)

            #create particle set
            particles = initialize_particles(num_particles, num_landmarks)

            dec_ave = []
            running_best_particle = []
            GT_locations = []
            GT_last = np.array([0,0,0])

            #run FastSLAM
            for timestep in range(int(len(sensor_readings)/2)):
                #get new GT pose
                reading = sensor_readings[timestep,'odometry']

                x, y, theta = get_new_pose(GT_last[0], GT_last[1], GT_last[2], reading['t'], reading['r1'], reading['r2'])
                GT_last = np.array([x, y, theta])
                GT_locations.append(GT_last)
                
                #predict particles by sampling from motion model with odometry info
                sample_motion_model(sensor_readings[timestep,'odometry'], particles)

                #evaluate sensor model to update landmarks and calculate particle weights
                eval_sensor_model(sensor_readings[timestep, 'sensor'], particles)

                best = best_particle(particles)
                running_best_particle.append([best['x'], best['y'], best['theta']])
                dec_ave.append(get_dec_ave(particles))
                
                #plot filter state
                plot_state(particles, landmarks, GT_locations, running_best_particle, dec_ave)

                save_hist(particles)

                #calculate new set of equally weighted particles
                particles = resample_particles(particles)

            dec_ave = np.array(dec_ave)
            running_best_particle = np.array(running_best_particle)
            GT_locations = np.array(GT_locations)
            last_best_particle = np.array(best['history'])
            plot_errors(dec_ave, running_best_particle, GT_locations, last_best_particle)

            RMSE_ave += np.sqrt(((GT_locations[:,0]-dec_ave[:,0]) ** 2).mean() + ((GT_locations[:,1]-dec_ave[:,1]) ** 2).mean())
            max_error_ave += np.max(np.linalg.norm(GT_locations-dec_ave, axis=1))
            RMSE_best += np.sqrt(((GT_locations[:,0]-last_best_particle[:,0]) ** 2).mean() + ((GT_locations[:,1]-last_best_particle[:,1]) ** 2).mean())
        
        RMSE_ave /= loops
        max_error_ave /= loops
        RMSE_best /= loops

        print(f'num particles = {num_particles}')
        print(f'RMSE for weighted error = {RMSE_ave:.2f}')
        print(f'max_error_ave for weighted error = {max_error_ave:.2f}')
        print(f'RMSE for last_best error = {RMSE_best:.2f}')

        RMSE_ave_list.append(RMSE_ave)
        max_error_ave_list.append(max_error_ave)
    
    # fig = plt.figure()
    # plt.close()
    # plt.subplot(2,1,1)
    # plt.axis([0, 100, 0, 2])
    # plt.plot(num_particles_list, RMSE_ave_list, label='RMSE', c='b')
    # plt.title('error vs. number of particles')
    # plt.ylabel('RMSE')

    # plt.subplot(2,1,2)
    # plt.axis([0, 100, 0, 8])
    # plt.plot(num_particles_list, max_error_ave_list, label='max_err', c='b')
    # plt.xlabel('# of particles')
    # plt.ylabel('max_err')
    # plt.savefig(f'fast_slam_err_vs_num_of_particles.png')

        

if __name__ == "__main__":
    main()