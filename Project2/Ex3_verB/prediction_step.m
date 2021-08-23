function [mu, sigma] = prediction_step(mu, sigma, u)
% Updates the belief concerning the robot pose according to the motion model,
% mu: 2N+3 x 1 vector representing the state mean
% sigma: 2N+3 x 2N+3 covariance matrix
% u: odometry reading (r1, t, r2)
% Use u.r1, u.t, and u.r2 to access the rotation and translation values


% TODO: Compute the 3x3 Jacobian Gx of the motion model
Gx = eye(3);
Gx(1,3) = -u.t*sin(mu(3)+u.r1);
Gx(2,3) = u.t*cos(mu(3)+u.r1);

% TODO: Compute the 3x3 Jacobian Rx of the motion model
[m,n] = size(sigma);
% R_tilde = diag([0.1+(0.1)^2,0.2+(0.2)^2,0.1+(0.1)^2]);
R_tilde = diag([(0.1)^2,(0.2)^2,(0.1)^2]);
Vt = [-u.t*sin(mu(3)+u.r1), cos(mu(3)+u.r1), 0;
        u.t*cos(mu(3)+u.r1), sin(mu(3)+u.r1), 0;
        1, 0, 1];
    
Rx = Vt*R_tilde*Vt';

% TODO: Compute the new mu based on the noise-free (odometry-based) motion model
% Remember to normalize theta after the update (hint: use the function normalize_angle available in tools)
Fx = zeros(3,n);
Fx(1:3,1:3) = diag([1,1,1]);
mu_temp = mu + Fx'* [u.t*cos(mu(3)+u.r1),u.t*sin(mu(3)+u.r1),u.r1+u.r2]';
mu_temp(3) = normalize_angle(mu_temp(3));
mu = mu_temp;

% % TODO: Construct the full Jacobian G
G = [Gx, zeros(3, n-3); zeros(n-3, 3), eye(n-3)];

% % TODO: Motion noise R
R = Fx'*Rx*Fx;

% Compute the predicted sigma after incorporating the motion
sigma = G*sigma*G' + R;

end
