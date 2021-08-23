import pykitti
import numpy as np
import pymap3d as pm

class KittiLoader:
    ned_to_enu = [[0,1,0],[1,0,0],[0,0,-1]]
    def __init__(self, basedir, date, drive):
        self.dataset = pykitti.raw(basedir, date, drive)
        self.parse_data()
    def parse_data(self):
        coords_lla = []
        location_ENU = []
        yaws = []
        velocities = []
        self.timestamps = np.array(self.dataset.timestamps)
        self.timediffs_start_0 = self.timestamps - self.timestamps[0]
        for oxts_data in self.dataset.oxts:
            coords_lla.append([oxts_data[0].lat, oxts_data[0].lon, oxts_data[0].alt])
            location_ned = np.array(pm.geodetic2ned(oxts_data[0].lat, oxts_data[0].lon, oxts_data[0].alt , self.dataset.oxts[0][0].lat, self.dataset.oxts[0][0].lon, self.dataset.oxts[0][0].alt))
            location_ENU.append(location_ned.dot(self.ned_to_enu))
            yaws.append([oxts_data[0].yaw, oxts_data[0].wz])
            velocities.append(oxts_data[0].vf)
        self.coords_lla = np.array(coords_lla)
        self.location_ENU = np.array(location_ENU)
        self.yaws = np.array(yaws)
        self.velocities = np.array(velocities)
    def get_noisy_EN_locations(self, sig_x, sig_y):
        x_noisy = self.location_ENU[:,0] + np.random.normal(0, sig_x, self.location_ENU.shape[0])
        y_noisy = self.location_ENU[:,1] + np.random.normal(0, sig_y, self.location_ENU.shape[0])
        return np.vstack([x_noisy, y_noisy]).T
    def get_noisy_yaw_rate(self, sig_w):
        return np.array(self.yaws[:,1]) + np.random.normal(0, sig_w, self.yaws.shape[0])
    def get_noisy_velocities(self, sig_v):
        return np.array(self.velocities) + np.random.normal(0, sig_v, self.velocities.shape[0])
    def get_GT_ENU_locations(self):
        return self.location_ENU
    def get_GT_LLA_locations(self):
        return self.coords_lla