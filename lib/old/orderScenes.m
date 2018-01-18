function tab = orderScenes(scenes,ids)
%ORDERSCENES Summary of this function goes here
%   Detailed explanation goes here
tab = cell(1,2);
i = 0;
st = scenes;
scenes = scenes(ids);
while (~isempty(scenes))
    i = i+1;
    tab{i,1} = scenes{1};
    scenes(cellfun(@strcmp, scenes, ...
        repmat(scenes(1),size(scenes,1),1))==1) = [];
end
for j=1:i
    tab{j,2} = find(cellfun(@strcmp, st, ...
        repmat(tab(j,1),size(st,1),1))==1);
end

end

