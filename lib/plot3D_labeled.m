function f = plot3D_labeled(xyz,label)
    x = xyz(:,1);
    y = xyz(:,2);
    z = xyz(:,3);
    f = figure;
    n = max(max(label))+1;
    rng('default');
    colors = rand(n,3);
    lab = zeros(numel(label),3);
    for i=1:numel(label)
        lab(i,:) = colors(label(i)+1,:);
    end
    dimension = 0.5;
    hold on;
    scatter3(x,y,z,dimension*ones(numel(x),1),lab);
    %view(0,10);
    camproj perspective;
    %camtarget('auto');
    %campos('auto');
    t = mean(xyz,1);
    camtarget(t);
    distance = 30;
    campos([t(1),t(2)-distance,t(3)]);
end


