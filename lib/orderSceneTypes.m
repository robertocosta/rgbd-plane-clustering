function tab = orderSceneTypes(sceneTypes)
%orderSceneTypes returns a cell array with 2 elements per row: [sceneType,
% indexes of the images of that specific Type] 
tab = cell(1,2);
i = 0;
st = sceneTypes;
while (~isempty(sceneTypes))
    i = i+1;
    tab{i,1} = sceneTypes{1};
    sceneTypes(cellfun(@strcmp, sceneTypes, ...
        repmat(sceneTypes(1),size(sceneTypes,1),1))==1) = [];
end
for j=1:i
    tab{j,2} = find(cellfun(@strcmp, st, ...
        repmat(tab(j,1),size(st,1),1))==1);
end