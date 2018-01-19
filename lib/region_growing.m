function ind_done = region_growing(xyz,plane,starting_idx,t)
%FIND_INLIERS Geographical region growing from selected inliers
%   Detailed explanation goes here
neighbors_layers = 10;
new_ind = true(5*neighbors_layers,1);
ind_done = starting_idx;
all_points = find(point_belongs_to_plane(xyz,plane,t));
while (length(new_ind)>4*neighbors_layers)
    ind_todo = ind_done;
    for i=1:neighbors_layers
        ind_todo = aggiungi_vicini(ind_todo);
    end
    new_ind = setdiff(ind_todo,ind_done);
    new_ind = intersect(new_ind,all_points);
    ind_done = union(ind_done,new_ind);
end
% idid = zeros(size(xyz,1),1);
% idid(ind) = 1;
% plot3D_labeled(xyz,idid);
% title('region growing');
