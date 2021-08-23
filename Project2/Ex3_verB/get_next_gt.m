function [next_loc] = get_next_gt(current_loc, odometry)
%GET_NEXT_GT Summary of this function goes here
%   Detailed explanation goes here
next_loc = [current_loc(1) + odometry.t*cos(current_loc(3)+odometry.r1);
            current_loc(2) + odometry.t*sin(current_loc(3)+odometry.r1);
            current_loc(3)+odometry.r1+odometry.r2];
end

