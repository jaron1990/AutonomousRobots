function noisy_odometry = generate_noisy_odometry(odometry,sig_rot1, sig_t, sig_rot2)
%GENERATE_NOISY_ODOMETRY Summary of this function goes here
%   Detailed explanation goes here
noisy_odometry = struct;

noisy_odometry.r1 = odometry.r1 + normrnd(0,sig_rot1);
noisy_odometry.t = odometry.t + normrnd(0,sig_t);
noisy_odometry.r2 = odometry.r2 + normrnd(0,sig_rot2);

end

