function [angles_ret] = normalize_all_bearings(angles)
% Go over the observations vector and normalize the bearings
% The expected format of z is [range; bearing; range; bearing; ...]

for i=1:length(angles)
   angles(i) = normalize_angle(angles(i));
end
angles_ret = angles;
