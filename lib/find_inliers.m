function ind = find_inliers(xyz,selected)
%FIND_INLIERS Summary of this function goes here
%   Detailed explanation goes here
neighbors_layers = 3;
rng('default');
% take a random set of initial points
initial_n_of_points = 30;
selected_ind = selected(randi(length(selected),initial_n_of_points,1));
% take every possible m-set of initial points
m = 5;
possible_starting_ind = combnk(1:initial_n_of_points,m);
planes = zeros(nchoosek(initial_n_of_points,m),4);
vars = zeros(nchoosek(initial_n_of_points,m),1);
for i=1:nchoosek(initial_n_of_points,m)
    starting_points = xyz(selected_ind(possible_starting_ind(i,:)),:);
    [planes(i,:),vars(i)] = find_plane(starting_points);
end
ind_min_var = find(vars==min(vars));
%min_var = vars(ind_min_var);
starting_ind = selected_ind(possible_starting_ind(ind_min_var(1),:));
starting_points = xyz(starting_ind,:);

%final_points = xyz(final_ind,:);
condition = zeros(50,1);
subsets = cell(1);
j = 1;
while (length(condition)>1)
    final_ind = starting_ind;
    for i=1:neighbors_layers
        final_ind = aggiungi_vicini(final_ind);
    end
    final_ind = intersect(final_ind,selected);
    new_ind = setdiff(final_ind,starting_ind);
    new_points = xyz(new_ind,:);
    vars = zeros(length(new_ind),1);
    for i=1:length(new_ind)
        [~, vars(i)] = find_plane([starting_points;new_points(i,:)]);
    end
    %condition = find(abs(vars-min_var)*length(starting_ind)<threshold);
    condition = find(vars<mean(vars)); %-sqrt(var(vars)/2)
    new_ind = new_ind(condition);
    %new_points = xyz(new_ind,:);
    starting_ind = [starting_ind;new_ind];
    starting_points = xyz(starting_ind,:);
    %final_ind = starting_ind;
    %final_points = xyz(final_ind,:);
    %[~, min_var] = find_plane(final_points);
    if (length(final_ind)>300 && length(condition)>4)
        subsets{j} = final_ind;
        j = j+1;
        selected = setdiff(selected,final_ind);
        selected = union(selected,new_ind);
        starting_ind = new_ind;
    end
end
if ~isempty(subsets{1})
    ind = subsets{1};
    for i=2:length(subsets)
        ind = union(ind,subsets{i});
    end
    ind = union(ind,starting_ind);
else
    ind = starting_ind;
end
    
    
    
    