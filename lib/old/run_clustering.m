function labels = run_clustering(geom, ~)
%RUN_CLUSTERING 
planes = geom(:,1:4);
%normal = geom(:,5:7);
labels = zeros(size(geom,1),1);
threshold = 0.6;
stepThreshold = 0.05;
maxClusters = 10;
out = cell(1,2);
i=1;
while (~isempty(planes))
    sqDiff = (planes-repmat(planes(1,:),size(planes,1),1)).^2;
    indexes = find(sqrt(sum(sqDiff,2))<threshold);
    out{i,1} = mean(planes(indexes,:),1);
    planes(indexes,:) = [];
    i = i+1;
end
while (size(out,1)>maxClusters)
    threshold = threshold + stepThreshold;
    planes = geom(:,1:4);
    out = cell(1,2);
    i=1;
    while (~isempty(planes))
        sqDiff = (planes-repmat(planes(1,:),size(planes,1),1)).^2;
        indexes = find(sqrt(sum(sqDiff,2))<threshold);
        out{i,1} = mean(planes(indexes,:),1);
        planes(indexes,:) = [];
        i = i+1;
    end
end
for i=1:size(geom,1)
    plane = geom(i,1:4);
    distances = zeros(size(out,1),1);
    for j=1:length(distances)
        distances(j) = norm(plane-out{j,1});
    end
    %distances = norm(repmat(plane,size(out,1),1)-out{:,1});
    minD = find(distances==min(distances));
    out{minD,2} = [out{minD,2} i];
end
for i=1:size(out,1)
    for j=1:length(out{i,2})
        labels(out{i,2}(j)) = i;
    end
end
labels = uint16(reshape(labels,427,561));
