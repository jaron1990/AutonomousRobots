function [phiNorm] = normalize_angle(phi)
%Normalize phi to be between -pi and pi

if (phi>pi)
	phi = phi - 2*pi;
end
if(phi<-pi)
	phi = phi + 2*pi;
end

phiNorm = phi;

end
