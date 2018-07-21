% Finds the major scene surfaces(planes) using ransac.
% 
% Args:
%   X - HxW matrix of X coordinates.
%   Y - HxW matrix of Y coordinates.
%   Z - HxW matrix of Z coordinates.
%   normals - Nx3 matrix of surface normals where N=H*W
%   isvalid - HxW matrix indicating whether each point is valid.
%
% Returns:
%   planes - Mx4 matrix of plane parameters where M is the number of major
%            surfaces found.
%   plane_idx - 1xM cell array of indices of each plane.
function [planes, plane_idx] = xyz2planes_ransac(X, Y, Z, normals, isvalid,...
    min_pts_per_plane)

    [H, W] = size(X);

    % inlier threshold for distance of 3D point to plane (sensor resolution is 0.003 per meter)
    distthresh = 0.0075*Z(:); 
    normthresh = 0.1; % inlier threshold for surface normal distance
    offset = 30; % horizontal/vertical offset for choosing triples of points

    X = X(:);
    Y = Y(:);
    Z = Z(:);

    npts = numel(X);
    pts = [X Y Z ones(npts, 1)];

    ui = round(offset/2):offset/2:round(W-offset/2);
    vi = round(offset/2):offset/2:round(H-offset/2);
    [ui, vi] = meshgrid(ui, vi);
    i1 = vi(:) + (ui(:)-1)*H;
    i2 = i1+(offset-(2*offset)*(ui(:)>W/2))*H;
    i3 = i1+(offset-(2*offset)*(vi(:)>H/2));
    validi = all(isvalid([i1 i2 i3]), 2);
    i1 = i1(validi);  i2 = i2(validi);  i3 = i3(validi);

    %i = randperm(numel(valid));
    niter = numel(i1);
    planes = zeros(4, niter);
    inliers = cell(niter, 1);
    count = zeros(niter, 1);

    for t = 1:niter

        % unit-norm solution
        A = pts([i1(t) i2(t) i3(t)], :);        
        [v, ~] = eig(A'*A);
        planes(:, t) = v(:, 1);
        planes(:, t) = planes(:, t) / sqrt(sum(planes(1:3, t).^2));

        % perpendicular distance of points to the plane
        dist = abs(pts*planes(:, t));
        distN = 1-abs(normals*planes(1:3, t));

        inliers{t} = find(isvalid(:) & (dist<distthresh) & (distN < normthresh)) ; 
        count(t) = numel(inliers{t});         

    end    

    [~, si] = sort(count, 'descend');
    isused = false(size(isvalid));

    plane_idx = {};

    c = zeros(niter, 1);
    for t = 1:niter        
        c(t) = sum(~isused(inliers{si(t)}));
        if c(t)>min_pts_per_plane
            err = abs(pts*planes(:, si(t)));
            tmpidx = isvalid(:) & err<distthresh/2;
            A = pts(tmpidx, :);        
            [v, ~] = eig(A'*A);
            planes(:, si(t)) = v(:, 1);                    
            planes(:, si(t)) = planes(:, si(t)) / sqrt(sum(planes(1:3, si(t)).^2));

            plane_idx{end+1} = inliers{si(t)};
            isused(inliers{si(t)}) = 1;
        end
    end

    planes = planes(:, si(c>min_pts_per_plane))';
    [planes, plane_idx] = merge_similar_planes(planes, plane_idx, pts, distthresh);

end

function [planes2, plane_idx2] = merge_similar_planes(planes, plane_idx, pts, distthresh)
    np = size(planes, 1);
    err = cell(np,1);
    for p1 = 1:np
        err{p1} = abs(pts*planes(p1, :)');
    end
    newassign = 1:np;
    for p1 = 1:np
        for p2 = p1+1:np
            idx_plane2 = plane_idx{p2};
            idx_plane1 = plane_idx{p1};
            d_points2plane1 = err{p1}(idx_plane2);
            d_points1plane2 = err{p2}(idx_plane1);
            mean_d_plane1points2 = ...
                mean(d_points2plane1 < distthresh(idx_plane2)*2);
            mean_d_plane2points1 = ...
                mean(d_points1plane2 < distthresh(idx_plane1)*2);
            if  mean_d_plane1points2 > 0.4 || mean_d_plane2points1 > 0.4
                newassign(p2) = p1;
            end
        end
    end
    uid = unique(newassign);
    for p = 1:numel(uid)
        ind = find(newassign==uid(p));
        planes2(p, :) = planes(ind(1), :);
        plane_idx2{p} = cat(1, plane_idx{ind});
    end
end

