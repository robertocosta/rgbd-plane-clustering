function id = cell2idx(ar)
%CELL2IDX Get the labels from a cell array of indexes
h = 480;
w = 640;
n=h*w;
id = uint32(zeros(n,1));
for i=1:length(ar)
    id(ar{i}) = i;
end     
end

