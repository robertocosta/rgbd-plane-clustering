function links = run_training(labels)
%RUN_TRAINING Summary of this function goes here
%   Detailed explanation goes here
rightLabs = labels(:,1)+1;
links = cell(length(min(rightLabs):max(rightLabs)),3);
edges = min(rightLabs):max(rightLabs);
while (~isempty(rightLabs(rightLabs>0)))
    ithLab = find_most_frequent(rightLabs,edges);
    indexes = find(rightLabs == ithLab);
    links{ithLab,1} = [links{ithLab,1} find_most_frequent(labels(indexes,2),edges)];
    links{ithLab,2} = [links{ithLab,2} find_most_frequent(labels(indexes,3),edges)];
    links{ithLab,3} = [links{ithLab,3} find_most_frequent(labels(indexes,4),edges)];
    rightLabs(indexes) = 0;
end

end

