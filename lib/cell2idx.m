function id = cell2idx(ar)
%CELL2IDX Summary of this function goes here
%   Detailed explanation goes here
[h,w] = const;
n=h*w;
id = uint32(zeros(n,1));
for i=1:length(ar)
    id(ar{i}) = i;
end     
end

