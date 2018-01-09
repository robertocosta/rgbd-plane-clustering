%addpath(genpath('lib'));
addpath('lib/toolbox_nyu_depth_v2');
addpath('lib');
%addpath('lib/indoor_scene_seg/common');
%rmpath('D:\git\3DFinalProj\lib\indoor_scene_seg\common');
%rmpath(genpath('lib/indoor_scene_seg'))
%{
whos('-file',fullfile(pwd, 'mat/nyu_depth_v2_labeled.mat'))
  Name                      Size                         Bytes  Class 

  accelData              1449x4                          23184  single     
                 roll, yaw, pitch and tilt angle
  depths                  480x640x1449              1780531200  single                      
  images                  480x640x3x1449            1335398400  uint8                       
  instances               480x640x1449               445132800  uint8                       
  labels                  480x640x1449               890265600  uint16                      
  names                   894x1                         118076  cell                        
  namesToIds                 -                               8  containers.Map              
  rawDepthFilenames      1449x1                         302934  cell                        
  rawDepths               480x640x1449              1780531200  single                      
  rawRgbFilenames        1449x1                         302938  cell                        
  sceneTypes             1449x1                         187276  cell                        
  scenes                 1449x1                         201766  cell
%}

reduced_dataset = true;
reload = false;
vars = {'accelData','depths','rawDepths','images','instances','labels','names',...
    'namesToIds','sceneTypes','scenes'};
if reduced_dataset == true
    if reload == true
        clear all;
        load(fullfile(pwd, 'mat/reduced_dataset.mat'));
    else
        clearvars('-except', vars{:});
    end
else
    if reload==true
        load(fullfile(pwd, 'mat/nyu_depth_v2_labeled.mat'),vars{:});
    else
        clearvars('-except', vars{:});
    end
end    
clearvars vars reload;


close all;
ordInd = orderSceneTypes(sceneTypes);
% ordInd is a cell tab with 2 elements per row. Each row is composed
% by: [sceneType, list of indexes of scenes of that type]
for i=1:size(ordInd,1)
    ordInd{i,3} = orderScenes(scenes,ordInd{i,2});
    % another element is added to ordInd: a cell tab with 2 elements per
    % row: each row is composed by [scene, list of indexes of that scene]
    for j = 1:size(ordInd{i,3},1)
        for k=1:length(ordInd{i,3}{j,2})
            im = images(:,:,:,ordInd{i,3}{j,2}(k));
            depth = depths(:,:,ordInd{i,3}{j,2}(k));
            %raw_depth = rawDepths(:,:,ordInd{i,3}{j,2}(k));
            lab = labels(:,:,ordInd{i,3}{j,2}(k));
            f = show_im(im,depth,lab);
            imDepth = rgb_p2rgb_w(depth);
            X = imDepth(:,:,1); Y = imDepth(:,:,2); Z = imDepth(:,:,3);
            x = X(45:471, 41:601);
            y = Y(45:471, 41:601);
            z = Z(45:471, 41:601);
            [p, n, ~] = compute_local_planes(x,y,z);
            r = im(45:471,41:601,1);
            g = im(45:471,41:601,2);
            b = im(45:471,41:601,3);
            rgb = [ r(:) g(:) b(:) ];
            nx = n(:,:,1); ny = n(:,:,2); nz = n(:,:,3);
            pa = p(:,:,1); pb = p(:,:,2); pc = p(:,:,3); pd = p(:,:,4);
            toClust = horzcat(pa(:),pb(:),pc(:),pd(:),...
                nx(:),ny(:),nz(:),x(:),y(:),z(:));
            lab1 = lab(45:471, 41:601);
            lab2 = run_clustering(toClust, rgb);
            lab3 = run_clustering_2(toClust, rgb);
            lab4 = run_clustering_3(toClust, rgb);
            labs = [lab1(:) lab2(:) lab3(:) lab4(:)];
            relations = run_training(labs);
            plot3D_labeled([x(:),y(:),z(:)],lab1(:));
            plot3D_labeled([x(:),y(:),z(:)],lab2(:));
            plot3D_labeled([x(:),y(:),z(:)],lab3(:));
            plot3D_labeled([x(:),y(:),z(:)],lab4(:));
            %{
            ex_ind = 300;
            quiver3(x(ex_ind,ex_ind),y(ex_ind,ex_ind),z(ex_ind,ex_ind),...
                normals(ex_ind,ex_ind,1),normals(ex_ind,ex_ind,2),...
                normals(ex_ind,ex_ind,3),0.5);
            %}
            hold off;
            %planeData = rgbd2planes(im, depth, raw_depth, normals);
            
            %plot3D_labeled(point_cloud,lab(:));
        end
    end
end
clearvars i j k f im depth lab;


