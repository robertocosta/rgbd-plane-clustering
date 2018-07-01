function uv = toPinhole(cp)
% uv = toPinhole(cp) returns a matrix [:,:,2] with new uv coordinates
% cp is the camera parameters with cp.K, cp.R, cp.T
    [u,v] = meshgrid(1:cp.resolution(2),1:cp.resolution(1));
    points = double([u(:), v(:)]);
    % unpack the intrinisc matrix and the distortion coefficients
    cx = cp.K(1,3);
    cy = cp.K(2,3);
    fx = cp.K(1,1);
    fy = cp.K(2,2);
    skew = cp.K(1,2);
    k = cp.R;
    p = cp.D;
    
    % center the points
    center = [cx, cy];
    centeredPoints = bsxfun(@minus, points, center);

    % normalize the points
    yNorm = centeredPoints(:, 2) ./ fy;
    xNorm = (centeredPoints(:, 1) - skew * yNorm) ./ fx;
    
    % compute radial distortion
    r2 = xNorm .^ 2 + yNorm .^ 2;
    r4 = r2 .* r2;
    r6 = r2 .* r4;
    alpha = k(1) * r2 + k(2) * r4 + k(3) * r6;

    % compute tangential distortion
    xyProduct = xNorm .* yNorm;
    dxTangential = 2 * p(1) * xyProduct + p(2) * (r2 + 2 * xNorm .^ 2);
    dyTangential = p(1) * (r2 + 2 * yNorm .^ 2) + 2 * p(2) * xyProduct;
    
    % apply the distortion to the points
    normalizedPoints = [xNorm, yNorm];
    distortedNormalizedPoints = normalizedPoints + ...
        normalizedPoints .* [alpha, alpha] + [dxTangential, dyTangential];
    
    % convert back to pixels
    distortedPointsX = distortedNormalizedPoints(:, 1) * fx + cx + ...
        skew * distortedNormalizedPoints(:,2);
    distortedPointsY = distortedNormalizedPoints(:, 2) * fy + cy;

    uv = cat(3,reshape(distortedPointsX,size(u,1),size(u,2)),...
        reshape(distortedPointsY,size(u,1),size(u,2)));
end
