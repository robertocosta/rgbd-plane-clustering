function f = plot3D(xyz)
% outputs: f (figure)
    x = xyz(:,1);
    y = xyz(:,2);
    z = xyz(:,3);
    f = figure;
    scatter3(x,y,z);
    view(45,45);
end

