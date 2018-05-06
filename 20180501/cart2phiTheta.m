function PhiTheta = cart2phiTheta(n)
    theta = acos(n(:,:,3));
    theta(theta>pi/2) = theta(theta>pi/2)-pi;
    theta(isnan(theta)) = pi/2;
    phi = atan2(n(:,:,2),n(:,:,1));
%     phi = atan(n(:,:,2)./n(:,:,1));
    phi(isnan(phi)) = 0;
    PhiTheta = cat(3,phi,theta);
end