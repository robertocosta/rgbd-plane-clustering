function bool = point_belongs_to_plane(point,plane,max_distance)
%POINT_BELONG_TO_PLANE point: Nx3, plane: 1x4, max_distance: double
%   Detailed explanation goes here
dist = abs([point,ones(size(point,1),1)]*plane')./norm(plane(1:end-1));
bool = dist<=max_distance;
end

