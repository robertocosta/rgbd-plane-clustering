function [x,y,z] = radial2xyz(R)
%RADIAL2XYZ(R) takes the radial depth and gives the (x,y,z) coordinates
global glob;
    uv = glob.uv1;
    cx = glob.cp1.K(1,3);
    cy = glob.cp1.K(2,3);
    f = (glob.cp1.K(1,1)+glob.cp1.K(2,2))/2;
    uv = uv - repmat(reshape([cx,cy],1,1,2),size(uv,1),size(uv,2));
    z = ((R ./ sqrt(uv(:,:,1).^2+uv(:,:,2).^2+f^2))+1)*f;
    x = z.*uv(:,:,1) / f;
    y = z.*uv(:,:,2) / f;
end