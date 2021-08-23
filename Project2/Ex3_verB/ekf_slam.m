% This is the main extended Kalman filter SLAM loop. This script calls all the required
% functions in the correct order.
%
% You can disable the plotting or change the number of steps the filter
% runs for to ease the debugging. You should however not change the order
% or calls of any of the other lines, as it might break the framework.
%
% If you are unsure about the input and return values of functions you
% should read their documentation which tells you the expected dimensions.

% Turn off pagination:
more off;

% clear all variables and close all windows
figure('visible', 'on');
clear all;
close all;

% Make tools available
addpath('tools');

% Read world data, i.e. landmarks. The true landmark positions are not given to the robot
landmarks = read_world('world.dat');
% load landmarks;
% Read sensor readings, i.e. odometry and range-bearing sensor
data = read_data('sensor_data.dat');
%load data;

INF = 100;
% Get the number of landmarks in the map
N = size(landmarks,1);

%init lists
gt = zeros(3,1);
preds = repmat(0, 2*N+3, 1);
sigmas = repmat(INF, 2*N+3, 2*N+3);

% observedLandmarks is a vector that keeps track of which landmarks have been observed so far.
% observedLandmarks(i) will be true if the landmark with id = i has been observed at some point by the robot
observedLandmarks = repmat(false,1,N);

% TODO Initialize belief:
% mu: 2N+3x1 vector representing the mean of the normal distribution
% The first 3 components of mu correspond to the pose of the robot,
% and the landmark poses (xi, yi) are stacked in ascending id order.
% sigma: (2N+3)x(2N+3) covariance matrix of the normal distribution
mu = repmat(0, 2*N+3, 1);
sigma = diag(cat(1, [1;1;0.2], repmat(20, 2*N, 1)));

% toogle the visualization type
% showGui = true;  % show a window while the algorithm runs
showGui = false; % plot to files instead


out_dir = [pwd '/figures/' datestr(now,'yyyy_mm_dd__HH_MM_SS')];
mkdir(out_dir);

v = VideoWriter([out_dir '/ekf_slam.mp4'], 'MPEG-4');
v.FrameRate = 10;
open(v);


% Perform filter update for each odometry-observation pair read from the
% data file.
for t = 1:size(data.timestep, 2)
    %calculate new GT
    gt = [gt,get_next_gt(gt(:,end), data.timestep(t).odometry)];
    
    %generate noise to odometry
    noisy_odometry = generate_noisy_odometry(data.timestep(t).odometry, sqrt(0.01), sqrt(0.04), sqrt(0.01));
    
    % Perform the prediction step of the EKF
    [mu, sigma] = prediction_step(mu, sigma, noisy_odometry);

    % Perform the correction step of the EKF
    [mu, sigma, observedLandmarks] = correction_step(mu, sigma, data.timestep(t).sensor, observedLandmarks);

    preds = [preds, mu];
    sigmas = cat(3, sigmas,sigma);

    %Generate visualization plots of the current state of the filter
    plot_state(gt, preds, mu, sigma, landmarks, t, observedLandmarks, data.timestep(t).sensor, showGui);
    disp('Current state vector:')
    disp('mu = '), disp(mu)
    frame = getframe(gcf);
    writeVideo(v,frame);

end

close(v);

Errors = gt(1:3,:) - preds(1:3, :);
abs_err = abs(gt(1,:)-preds(1,:)) + abs(gt(2,:)-preds(2,:));
maxErr = max(abs_err(20:end));
RMSE_int = (gt(1,:)-preds(1,:)).^2 + (gt(2,:)-preds(2,:)).^2;
RMSE = sqrt( (1/(length(RMSE_int) - 20)) * sum(RMSE_int(20:end)));

saveas(gcf,[out_dir '/GT_vs_SLAM_err' num2str(maxErr) '_RMSE' num2str(RMSE) '.fig']);
saveas(gcf,[out_dir '/GT_vs_SLAM_err' num2str(maxErr) '_RMSE' num2str(RMSE) '.png']);


t = 1:size(Errors, 2);

%pose errors
figure
subplot(3,1,1)
title('\mu errors')
hold on
grid on
plot(t, Errors(1,:), 'b.', 'markersize', 10);
plot(t, reshape(sqrt(sigmas(1,1,:)), 1, size(sigmas,3)), 'k-')
plot(t, reshape(-sqrt(sigmas(1,1,:)), 1, size(sigmas,3)), 'k-')
ylim([-3,3])
ylabel('error[m]')
legend('x_{err}','sig_{x}')
hold off

subplot(3,1,2)
hold on
grid on
plot(t, Errors(2,:), 'b.', 'markersize', 10);
plot(t, reshape(sqrt(sigmas(2,2,:)), 1, size(sigmas,3)), 'k-')
plot(t, reshape(-sqrt(sigmas(2,2,:)), 1, size(sigmas,3)), 'k-')
ylim([-3,3])
ylabel('error[m]')
legend('y_{err}','sig_{y}')
hold off

subplot(3,1,3)
hold on
grid on
Errors(3,:) = normalize_all_angles(Errors(3,:));
plot(t, Errors(3,:), 'b.', 'markersize', 10);
plot(t, reshape(sqrt(sigmas(3,3,:)), 1, size(sigmas,3)), 'k-')
plot(t, reshape(-sqrt(sigmas(3,3,:)), 1, size(sigmas,3)), 'k-')
ylim([-pi/2,pi/2])
legend('\theta_{err}','sig_{\theta}')
hold off

saveas(gcf,[out_dir '/pose_err.fig']);
saveas(gcf,[out_dir '/pose_err.png']);

%landmark #1 errors
figure
subplot(2,1,1)
title('landmark #1 errors')
hold on
grid on
plot(t, preds(4,:) - landmarks(1,2), 'b.', 'markersize', 10);
plot(t, reshape(sqrt(sigmas(4,4,:)), 1, size(sigmas,3)), 'k-')
plot(t, reshape(-sqrt(sigmas(4,4,:)), 1, size(sigmas,3)), 'k-')
ylim([-3,3])
ylabel('error[rad]')
legend('x_{err}','sig_{x}')
hold off

subplot(2,1,2)
hold on
grid on
Errors(3,:) = normalize_all_angles(Errors(3,:));
plot(t, preds(5,:) - landmarks(1,3), 'b.', 'markersize', 10);
plot(t, reshape(sqrt(sigmas(5,5,:)), 1, size(sigmas,3)), 'k-')
plot(t, reshape(-sqrt(sigmas(5,5,:)), 1, size(sigmas,3)), 'k-')
ylim([-3,3])
ylabel('error[m]')
legend('y_{err}','sig_{y}')
hold off

saveas(gcf,[out_dir '/landmark1_err.fig']);
saveas(gcf,[out_dir '/landmark1_err.png']);

%landmark #2 errors
figure
subplot(2,1,1)
title('landmark #2 errors')
hold on
grid on
plot(t, preds(6,:) - landmarks(2,2), 'b.', 'markersize', 10);
plot(t, reshape(sqrt(sigmas(6,6,:)), 1, size(sigmas,3)), 'k-')
plot(t, reshape(-sqrt(sigmas(6,6,:)), 1, size(sigmas,3)), 'k-')
ylabel('error[m]')
ylim([-3,3])
legend('x_{err}','sig_{x}')
hold off

subplot(2,1,2)
hold on
grid on
Errors(3,:) = normalize_all_angles(Errors(3,:));
plot(t, preds(7,:) - landmarks(2,3), 'b.', 'markersize', 10);
plot(t, reshape(sqrt(sigmas(7,7,:)), 1, size(sigmas,3)), 'k-')
plot(t, reshape(-sqrt(sigmas(7,7,:)), 1, size(sigmas,3)), 'k-')
ylim([-3,3])
ylabel('error[m]')
legend('y_{err}','sig_{y}')
hold off

saveas(gcf,[out_dir '/landmark2_err.fig']);
saveas(gcf,[out_dir '/landmark2_err.png']);


disp('Final system covariance matrix:'), disp(sigma)
% Display the final state estimate
disp('Final robot pose:')
disp('mu_robot = '), disp(mu(1:3)), disp('sigma_robot = '), disp(sigma(1:3,1:3))

disp('RMSE = '), disp(RMSE)
disp('maxErr = '), disp(maxErr)