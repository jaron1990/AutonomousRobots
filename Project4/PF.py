from numpy import linalg, random
from read_data import read_world, read_sensor_data
from misc_tools import *
import numpy as np
import math
import copy
import warnings
import matplotlib
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)


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
    particles = []

    for i in range(num_particles):
        particle = dict()

        #initialize pose: at the beginning, robot is randomly distributed across the room
        r = 10 * np.sqrt(np.random.rand())
        theta = (2 * np.random.rand() - 1) * np.pi
        
        particle['x'] = 5 + r * np.cos(theta)
        particle['y'] = 5 + r * np.sin(theta)
        particle['theta'] = (2 * np.random.rand() - 1) * np.pi

        #initial weight
        particle['weight'] = 1.0 / num_particles
        
        #particle history aka all visited poses
        particle['history'] = []

        #add particle to set
        particles.append(particle)

    return particles

#calculate the measured ranges to every landmark
def calculate_ranges(landmarks, GT_location):
    ranges = []
    for idx, landmark in landmarks.items():
        range = np.linalg.norm([GT_location[0]-landmark[0], GT_location[1]-landmark[1]])
        ranges.append(range)
    return ranges

def delete_outside_particles(particles):
    new_particles = []

    for particle in particles:
        if (linalg.norm([particle['x']-5, particle['y']-5]) < 10):
            new_particles.append(particle)

    return new_particles

def sample_motion_model(odometry, particles):
    # Updates the particle positions, based on old positions, the odometry
    # measurements and the motion noise 

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    '''your code here'''
    sigma_rot1 = 0.05
    sigma_rot2 = 0.05
    sigma_trans = 0.1

    for particle in particles:
        delta_rot1_noisy = delta_rot1 + np.random.normal(0,sigma_rot1)
        delta_rot2_noisy = delta_rot2 + np.random.normal(0,sigma_rot2)
        delta_trans_noisy = delta_trans + np.random.normal(0,sigma_trans)

        particle['x'], particle['y'], particle['theta'] = get_new_pose(particle['x'], particle['y'], particle['theta'], delta_trans_noisy, delta_rot1_noisy, delta_rot2_noisy)
    '''***        ***'''

def save_hist(particles):
    for particle in particles:
        px = particle['x']
        py = particle['y']

        particle['history'].append([px, py])


def eval_sensor_model(particles, landmarks, ranges):
    #Correct landmark poses with a measurement and
    #calculate particle weight

    #update landmarks and calculate weight for each particle
    for particle in particles:
        px = particle['x']
        py = particle['y']
        weights = []
        diffs = []
        #loop over observed landmarks 
        for idx, landmark  in landmarks.items():
            '''your code here'''
            #calculate range from particle to landmark
            measured_range = ranges[idx]
            part_range = linalg.norm([py-landmark[1], px-landmark[0]])
            
            diff = part_range-measured_range
            diffs.append(diff)
            weights.append(get_weight(diff))
            '''***        ***'''

        # particle['weight'] = get_weight(diff)
        particle['weight'] = sum(weights) / len(weights)

    #normalize weights
    normalizer = sum([p['weight'] for p in particles])

    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer

def plot_errors(dec_ave, running_best_particle, GT_locations, last_best_particle):
    plt.show()
    plt.savefig(f'PF.png')

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
    plt.savefig(f'PF_err.png')

def resample_particles(particles, num_particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle 
    # weights.

    new_particles = []

    # '''your code here'''

    r = np.random.rand() / num_particles
    c = particles[0]['weight']
    i = 0

    for j in range(num_particles):
        U = r + j / num_particles
        while U > c:
            i+=1
            c+=particles[i]['weight']
        new_particle = copy.deepcopy(particles[i])
        new_particles.append(new_particle)
    # '''***        ***'''
    return new_particles

def main():

    print ("Reading landmark positions")
    landmarks = read_world("AutonomousRobots55/Project4/ParticleEX1/landmarks_EX1.csv")

    print ("Reading sensor data")
    sensor_readings = read_sensor_data("AutonomousRobots55/Project4/ParticleEX1/odometry.dat")

    loops = 10
    num_particles_list = [2, 5, 10, 20, 40, 60, 80, 100]
    # num_particles_list = [100]
    RMSE_ave_list = []
    max_error_ave_list = []
    for num_particles in num_particles_list:
        RMSE_ave = 0
        RMSE_best = 0
        max_error_ave = 0
        non_convergent_loops = 0
        for i in range(loops):
            num_landmarks = len(landmarks)

            #create particle set
            particles = initialize_particles(num_particles, num_landmarks)

            dec_ave = []
            running_best_particle = []
            GT_locations = []
            GT_last = np.array([0,0,0])

            # plot_state(particles, landmarks, GT_locations, running_best_particle, dec_ave)
            # plt.show()
            # plt.savefig(f'PF_init.png')


            #run PF
            for timestep in range(int(len(sensor_readings))):
                #get new GT pose
                reading = sensor_readings[timestep,'odometry']

                x, y, theta = get_new_pose(GT_last[0], GT_last[1], GT_last[2], reading['t'], reading['r1'], reading['r2'])
                GT_last = np.array([x, y, theta])
                GT_locations.append(GT_last)
                
                #predict particles by sampling from motion model with odometry info
                sample_motion_model(sensor_readings[timestep,'odometry'], particles)

                particles = delete_outside_particles(particles)

                if(len(particles)==0):
                    print ("all particles deleted - no convergence")
                    break

                #evaluate sensor model to update landmarks and calculate particle weights
                ranges = calculate_ranges(landmarks, GT_last)

                eval_sensor_model(particles, landmarks, ranges)

                best = best_particle(particles)
                # running_best_particle.append([best['x'], best['y'], best['theta']])
                dec_ave.append(get_dec_ave(particles))
                
                #plot filter state
                # plot_state(particles, landmarks, GT_locations, running_best_particle, dec_ave)

                save_hist(particles)

                #calculate new set of equally weighted particles
                particles = resample_particles(particles, num_particles)

            if (len(particles)==0):
                non_convergent_loops += 1
                continue

            dec_ave = np.array(dec_ave)
            running_best_particle = np.array(running_best_particle)
            GT_locations = np.array(GT_locations)
            last_best_particle = np.array(best['history'])
            # plot_errors(dec_ave, running_best_particle, GT_locations, last_best_particle)

            RMSE_ave += np.sqrt(((GT_locations[:,0]-dec_ave[:,0]) ** 2).mean() + ((GT_locations[:,1]-dec_ave[:,1]) ** 2).mean())
            max_error_ave += np.max(np.linalg.norm(GT_locations-dec_ave, axis=1))
            RMSE_best += np.sqrt(((GT_locations[:,0]-last_best_particle[:,0]) ** 2).mean() + ((GT_locations[:,1]-last_best_particle[:,1]) ** 2).mean())
        
        RMSE_ave /= (loops-non_convergent_loops)
        max_error_ave /= (loops-non_convergent_loops)
        RMSE_best /= (loops-non_convergent_loops)

        print(f'num particles = {num_particles}')
        print(f'RMSE for weighted error = {RMSE_ave:.2f}')
        print(f'max_error_ave for weighted error = {max_error_ave:.2f}')
        print(f'RMSE for last_best error = {RMSE_best:.2f}')
        print(f'non_convergent_loops = {non_convergent_loops}')

        RMSE_ave_list.append(RMSE_ave)
        max_error_ave_list.append(max_error_ave)
    
    fig = plt.figure()
    plt.close()
    plt.subplot(2,1,1)
    plt.axis([0, 100, 0, 10])
    plt.plot(num_particles_list, RMSE_ave_list, label='RMSE', c='b')
    plt.title('error vs. number of particles')
    plt.ylabel('RMSE')

    plt.subplot(2,1,2)
    plt.axis([0, 100, 0, 10])
    plt.plot(num_particles_list, max_error_ave_list, label='max_err', c='b')
    plt.xlabel('# of particles')
    plt.ylabel('max_err')
    plt.savefig(f'PF_err_vs_num_of_particles.png')

    plt.show()

    # a=1
        

if __name__ == "__main__":
    main()