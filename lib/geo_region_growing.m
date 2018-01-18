function ind = geo_region_growing(xyz,selected)
%FIND_INLIERS Summary of this function goes here
%   Detailed explanation goes here
neighbors_layers = 10;

threshold = 1.01;
starting_points = xyz(selected,:);
[~, min_var] = find_plane(starting_points);
%final_points = xyz(final_ind,:);
condition = true(50,1);
% subsets = cell(1);
% j = 1;
% counter = 1;
ind_done = selected;
ind = selected;
while (length(condition)>1)
    ind_todo = ind;
    for i=1:neighbors_layers
        ind_todo = aggiungi_vicini(ind_todo);
    end
    %final_ind = intersect(final_ind,selected);
    new_ind = setdiff(ind_todo,ind_done);
    new_points = xyz(new_ind,:);
    vars = zeros(length(new_ind),1);
    for i=1:length(new_ind)
        [~, vars(i)] = find_plane([starting_points;new_points(i,:)]);
    end
    %condition = find(abs(vars-min_var)*length(starting_ind)<threshold);
    
    condition = find(vars<=min_var*threshold); %-sqrt(var(vars)/2)
    ind_done = union(ind_done,new_ind);
    %new_points = xyz(new_ind,:);
    ind = [ind;new_ind(condition)];
    %starting_points = xyz(starting_ind,:);
    %final_ind = starting_ind;
    %final_points = xyz(final_ind,:);
    %[~, min_var] = find_plane(final_points);
    
%     if (length(final_ind)>500 && length(condition)>4)
%         subsets{j} = starting_ind;
%         j = j+1;
%         selected = setdiff(selected,final_ind);
%         selected = union(selected,new_ind);
%         starting_ind = new_ind;
%         dbug = mod(counter,5)==0;
%         counter = counter + 1;
%         if dbug==true
%             ind = subsets{1};
%             for i=2:length(subsets)
%                 ind = union(ind,subsets{i});
%             end
%             ind = union(ind,starting_ind);
%             idid = zeros(size(xyz,1),1);
%             idid(ind) = 1;
%             plot3D_labeled(xyz,idid);
%         end
%     end
    
end
idid = zeros(size(xyz,1),1);
idid(ind) = 1;
plot3D_labeled(xyz,idid);
title('region growing');
% if ~isempty(subsets{1})
%     ind = subsets{1};
%     for i=2:length(subsets)
%         ind = union(ind,subsets{i});
%     end
%     ind = union(ind,starting_ind);
% else
%     ind = starting_ind;
% end
    