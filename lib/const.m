function [h,w,neighbors] = const
%CONST Summary of this function goes here
%   Detailed explanation goes here
h = 427;
w = 561;
indexes = reshape(1:h*w,h,w);
neighbors = uint32(zeros(h*w,8));
for j=2:w-1
    for i=2:h-1
        ind = indexes(i,j);
        neighbors(ind,:) = [indexes(i-1,j-1),indexes(i-1,j),...
            indexes(i,j+1),indexes(i,j-1),indexes(i,j+1),...
            indexes(i+1,j-1),indexes(i+1,j),indexes(i+1,j+1)];
    end
end
neighbors(1,1:3) = [2,h+1,h+2];
% left part
for i=2:h-1
    neighbors(i,1:5) = [i-1,i+1,h+i-1,h+i,h+i+1];
end
neighbors(h,1:3) = [h-1,2*h-1,2*h];
% top part
for j=2:w-1
    neighbors(h*(j-1)+1,1:5)=[indexes(1,j-1),indexes(2,j-1),...
        indexes(2,j),indexes(1,j+1),indexes(2,j+1)];
end
neighbors(h*(w-1)+1,1:3) = [h*(w-2)+1,h*(w-2)+2,h*(w-1)+2];
% bottom part
for j=2:w-1
    neighbors(indexes(h,j),1:5)=[indexes(h-1,j-1),indexes(h,j-1),...
        indexes(h-1,j),indexes(h-1,j+1),indexes(h,j+1)];
end
neighbors(h*w,1:3) = [h*(w-1)-1,h*(w-1),h*w-1];
% right part
for i=2:h-1
    neighbors(indexes(i,w),1:5)=[indexes(i-1,w-1),indexes(i,w-1),...
        indexes(i+1,w-1),indexes(i-1,w),indexes(i+1,w)];
end
if (nargout==1)
    h = neighbors;
end
end


