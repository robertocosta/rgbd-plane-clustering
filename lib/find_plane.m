function [plane,var] = find_plane(points)
%FIND_PLANE Returns the best fitted plane and error by SVD
%   Detailed explanation goes here
A = [points,ones(size(points,1),1)];

% [eigv, c] = eig(A'*A);
% plane = transpose(eigv(:,1));
% var = abs(sqrt(c(1,1)/c(2,2)));
[~,~,V] = svd(A,0);
plane = V(:,4);
errors = A*plane;
var = sum(abs(errors))/size(points,1);
% var = var/(10^(round(log10(var))));
end

