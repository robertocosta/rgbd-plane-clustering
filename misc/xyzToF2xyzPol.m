function [x,y,z] = xyzToF2xyzPol(x,y,z)
%XYZTOF2XYZPOL(x,y,z) transform from ToF ref. sys. to Pol. ref. sys.
global glob;
    R = glob.R1;
    T = glob.T1;

    siz = size(x);

    xyz1 = [x(:), y(:), z(:)] - repmat(transpose(T),numel(x),1);
    xyz2 = transpose(R \ transpose(xyz1));

    x = reshape(xyz2(:,1),siz(1),siz(2));
    y = reshape(xyz2(:,2),siz(1),siz(2));
    z = reshape(xyz2(:,3),siz(1),siz(2));

end

