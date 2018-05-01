function PhiTheta = cart2phiTheta(n)
    
    N = n(:,:,1).^2+n(:,:,2).^2+n(:,:,3).^2;
    theta = acos(n(:,:,3)./sqrt(N));
    theta(theta>pi/2) = theta(theta>pi/2)-pi;
    theta(isnan(theta)) = pi/2;
%     phi = atan2(n(:,:,2),n(:,:,1));
    phi = atan(n(:,:,2)./n(:,:,1));
    phi(isnan(phi)) = 0;
    PhiTheta = cat(3,phi,theta);
end