function labels = run_clustering_3(geom, color)
%RUN_CLUSTERING 
%planes = geom(:,1:4);
%normals = geom(:,5:7);
labels = zeros(size(geom,1),1);
colors = rgb2lab(color);
c2 = colors;
threshold = 15;
stepThreshold = 2;
maxClusters = 10;
out = cell(1,2);
i=1;
while (~isempty(colors))
    sqDiff = (colors-repmat(colors(1,:),size(colors,1),1)).^2;
    indexes = find(sqrt(sum(sqDiff,2))<threshold);
    out{i,1} = mean(colors(indexes,:),1);
    colors(indexes,:) = [];
    i = i+1;
end
while (size(out,1)>maxClusters)
    threshold = threshold + stepThreshold;
    colors = c2;
    out = cell(1,2);
    i=1;
    while (~isempty(colors))
        sqDiff = (colors-repmat(colors(1,:),size(colors,1),1)).^2;
        indexes = find(sqrt(sum(sqDiff,2))<threshold);
        out{i,1} = mean(colors(indexes,:),1);
        colors(indexes,:) = [];
        i = i+1;
    end
end
colors = c2;
for i=1:size(geom,1)
    c = colors(i,:);
    distances = zeros(size(out,1),1);
    for j=1:length(distances)
        distances(j) = norm(c-out{j,1});
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
