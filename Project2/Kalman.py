import numpy as np

class Kalman:
    C=np.array([[1, 0, 0, 0],
                [0, 0, 1, 0]])

    def __init__(self, measurements, measurement_cov, prediction_cov, timestamps, sig_r=15):
        self.meas = measurements
        self.meas_cov = measurement_cov
        self.pred_cov = prediction_cov
        self.timestamps = timestamps
        self.current_state = np.array([measurements[0,0],0,measurements[0,1],0])
        self.states = []
        self.pred_cov_list = []
        self.states.append(self.current_state)
        self.pred_cov_list.append(self.pred_cov)
        self.sigma_r = sig_r

    def calculate_A(self, dt):
        A = np.array([  [1,dt,0,0],
                        [0,1,0,0],
                        [0,0,1,dt],
                        [0,0,0,1]], dtype=np.float)
        return A

    def calculate_R(self,dt):
        return np.diag([0,dt,0,dt])*(self.sigma_r**2)

    def predict(self, dt):
        A = self.calculate_A(dt)
        R = self.calculate_R(dt)
        #predict state
        prediction = A.dot(self.current_state) ##TODO: add w
        #predict covariance
        temp_covariance_pred = A.dot(self.pred_cov.dot(A.T)) + R ## TODO: add Q
        return prediction, temp_covariance_pred

    def run(self, dead_reckoning=0):
        for ts, measurement in zip(self.timestamps, self.meas):
            if ts.microseconds==0 and ts.seconds==0:
                prev_ts = ts
                continue

            dt = ts - prev_ts
            dt_sec = float(dt.microseconds) / 1E6
            prediction, temp_covariance_pred = self.predict(dt_sec)
            nominator = temp_covariance_pred.dot(self.C.T)
            denominator = self.C.dot(temp_covariance_pred.dot(self.C.T)) + self.meas_cov
            KG = nominator.dot(np.linalg.inv(denominator))

            if (dead_reckoning and ts.seconds>=5):
                KG = KG*0
            self.current_state = prediction + KG.dot(measurement-self.C.dot(prediction))
            self.pred_cov = (np.identity(4) - KG.dot(self.C)).dot(temp_covariance_pred)
            
            self.states.append(self.current_state)
            self.pred_cov_list.append(self.pred_cov)
            prev_ts = ts

    def get_xy(self):
        return np.array(self.states)[:,::2]

    def get_pred_cov_list(self):
        return self.pred_cov_list

class ExtKalman():
    H=np.array([[1,0,0],
                [0,1,0]])

    def __init__(self, measurements, measurement_cov, yaw_rates, yaw_rates_sig, velocities, velocities_sig, prediction_cov, timestamps):
        self.meas = measurements
        self.meas_cov = measurement_cov
        self.pred_cov = prediction_cov
        self.timestamps = timestamps
        self.current_state = np.array([measurements[0,0],measurements[0,1],0])
        self.states = []
        self.pred_cov_list = []
        self.states.append(self.current_state)
        self.pred_cov_list.append(self.pred_cov)
        self.yaw_rates = yaw_rates
        self.sig_w = yaw_rates_sig
        if(self.sig_w==0):
            self.sig_w = 0.1
        self.velocities = velocities
        self.sig_v = velocities_sig
        if(self.sig_v==0):
            self.sig_v = 0.1
        self.R = np.diag([self.sig_v**2, self.sig_w**2])
    
    def predict(self, dt, velocity, yaw_rate):
        x, y, yaw = self.current_state
        v_w = velocity/yaw_rate
        x_fix = (-1)*v_w*np.sin(yaw) + v_w*np.sin(yaw+yaw_rate*dt)
        y_fix = v_w*np.cos(yaw) - v_w*np.cos(yaw+yaw_rate*dt)
        yaw_fix = yaw_rate*dt
        fix = [x_fix, y_fix, yaw_fix]
        prediction = self.current_state + fix
        Gt = np.eye(3)
        Gt[:2,2] = [-y_fix,x_fix]
        Vt = np.array([[x_fix/velocity, -x_fix/yaw_rate + v_w*np.cos(yaw+yaw_rate*dt)*dt],
                        [y_fix/velocity, -y_fix/yaw_rate + v_w*np.sin(yaw+yaw_rate*dt)*dt],
                        [0, dt]])
        temp_prediction_cov = Gt.dot(self.pred_cov.dot(Gt.T)) + Vt.dot(self.R.dot(Vt.T))
        return prediction , temp_prediction_cov

    def run(self, dead_reckoning=0):
        for ts, measurement, yaw_rate, velocity in zip(self.timestamps, self.meas, self.yaw_rates, self.velocities):
            if ts.microseconds==0 and ts.seconds==0:
                prev_ts = ts
                continue
            dt = ts-prev_ts
            dt_sec = float(dt.microseconds) / 1E6
            prediction, temp_covariance_pred = self.predict(dt_sec, velocity, yaw_rate)

            nominator = temp_covariance_pred.dot(self.H.T)
            denominator = self.H.dot(temp_covariance_pred.dot(self.H.T)) + self.meas_cov

            KG = nominator.dot(np.linalg.inv(denominator))
            if (dead_reckoning and ts.seconds>=5):
                KG = KG*0
            self.current_state = prediction + KG.dot(measurement-self.H.dot(prediction))
            self.pred_cov = (np.eye(3) - KG.dot(self.H)).dot(temp_covariance_pred)
            self.states.append(self.current_state)
            self.pred_cov_list.append(self.pred_cov)
            prev_ts = ts

    def get_xy(self):
        return np.array(self.states)[:,:2]

    def get_yaws(self):
        return np.array(self.states)[:,2]

    def get_pred_cov_list(self):
        return np.array(self.pred_cov_list)