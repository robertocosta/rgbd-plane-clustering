%addpath(genpath('lib'));
addpath('lib/toolbox_nyu_depth_v2');
addpath('lib');
%addpath('lib/backup3');

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
            raw_depth =depth;
            %gauss_smoothed = imgaussfilt3(cat(3,im,depth));
            depth = imgaussfilt(raw_depth,0.6);
            %raw_depth = rawDepths(:,:,ordInd{i,3}{j,2}(k));
            lab = labels(:,:,ordInd{i,3}{j,2}(k));
            f = show_im(im,depth,lab);
            xyz = rgb_p2rgb_w(depth);
            N = numel(meshgrid(45:471, 41:601));
            XYZ = [reshape(xyz(45:471, 41:601, :),N,3), ones(N,1)];
            x = xyz(45:471, 41:601, 1);
            y = xyz(45:471, 41:601, 2);
            z = xyz(45:471, 41:601, 3);
            [p, n, conf] = compute_local_planes(x,y,z);
            %nn_approach = [x(:),y(:),z(:),reshape(p,N,4),double(reshape(im(45:471, 41:601, :),N,3))/255];
            %kmeans_input = [reshape(p,N,4),double(reshape(rgb2lab(im(45:471, 41:601, :)),N,3))/255];
            
            % normali
            kmeans_input = reshape(n,N,3);
            [idx1,C1,~,D1] = kmeans(kmeans_input,6);
            plot3D_labeled([x(:),y(:),z(:)],idx1);
            title("K-means Normals");
            [class_ord1, max_ind1] = sort_by_variance(D1,idx1,0.8);
            
            % piani
            kmeans_input = reshape(p,N,4);
            [idx2,C2,~,D2] = kmeans(kmeans_input,6);
            plot3D_labeled([x(:),y(:),z(:)],idx2);
            title("K-means Planes");
            [class_ord2, max_ind2] = sort_by_variance(D1,idx1,0.8);
            
            % piani, colore
            kmeans_input = [reshape(p,N,4),double(reshape(rgb2lab(im(45:471, 41:601, :)),N,3))/255];
            [idx3,C3,~,D3] = kmeans(kmeans_input,6);
            plot3D_labeled([x(:),y(:),z(:)],idx3);
            title("K-means Planes + color");
            [class_ord3, max_ind3] = sort_by_variance(D1,idx1,0.8);
            
            % normali, colore
            kmeans_input = [reshape(n,N,3),double(reshape(rgb2lab(im(45:471, 41:601, :)),N,3))/255];
            [idx4,C4,~,D4] = kmeans(kmeans_input,6);
            plot3D_labeled([x(:),y(:),z(:)],idx4);
            title("K-means Normals + color");
            [class_ord4, max_ind4] = sort_by_variance(D1,idx1,0.8);
           
            
            [planes, plane_idx] = xyz2planes_ransac(x,y,z,...
                reshape(n,N,3),abs(conf)>0.5);
            lab1 = lab(45:471, 41:601);
            lab2 = uint8(zeros(427,561));
            for ii=1:size(planes,1)
                lab2(plane_idx{ii})=ii;
            end
            plot3D_labeled([x(:),y(:),z(:)],lab2(:));
            plot3D_labeled([x(:),y(:),z(:)],lab1(:));
        end
    end
end
clearvars i j k f im depth lab;

