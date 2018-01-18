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
            depth = imgaussfilt(raw_depth,0.1);
%             depth2 = imgaussfilt(raw_depth,0.1);
%             depth_gradient = single(zeros(size(depth2,1)-1,size(depth2,2)-1,2));
%             for ii=1:size(depth_gradient,1)
%                 for jj=1:size(depth_gradient,2)
%                     depth_gradient(ii,jj,1) = depth2(ii+1,jj)-depth2(ii,jj);
%                     depth_gradient(ii,jj,2) = depth2(ii,jj+1)-depth2(ii,jj);
%                 end
%             end
%             depth_gradient = depth_gradient(45:471, 41:601, :);
            %raw_depth = rawDepths(:,:,ordInd{i,3}{j,2}(k));
            lab = labels(:,:,ordInd{i,3}{j,2}(k));
            f = show_im(im,depth,lab);
            xyz = rgb_p2rgb_w(depth);
            N = numel(meshgrid(45:471, 41:601));
            XYZ = [reshape(xyz(45:471, 41:601, :),N,3), ones(N,1)];
            x = xyz(45:471, 41:601, 1);
            y = xyz(45:471, 41:601, 2);
            z = xyz(45:471, 41:601, 3);
            xyz = [x(:),y(:),z(:)];
            [p, n, conf] = compute_local_planes(x,y,z);
            [planes, planes_idx] = xyz2planes_ransac(x,y,z,...
                reshape(n,N,3),abs(conf)>0.5);
            n_lab = length(planes_idx);
            
            lab0 = lab(45:471, 41:601);
            idx = cell2idx(planes_idx);
            plot3D_labeled([x(:),y(:),z(:)],lab0);
            title('labeled');
            plot3D_labeled([x(:),y(:),z(:)],idx);
            title("xyz2planes-ransac");
            inliers = cell(2,1);
            expanded_region = cell(2,1);
            inliers_planes = zeros(size(planes));
            vars = zeros(n_lab,1);
            inliers_percent = zeros(n_lab,1);
            for ii=1:n_lab
                planes(ii,:) = planes(ii,:)/norm(planes(i,:));
                inliers{ii} = find_inliers(xyz,find(idx==ii));
                expanded_region{ii} = geo_region_growing(xyz,inliers{ii});
                inliers_percent(ii) = length(inliers{ii})/...
                    double(length(planes_idx{ii}))*100;
                disp(strcat(num2str(inliers_percent(ii)),' % inliers'));
                if inliers_percent(ii)>70
                    lab1 = uint8(zeros(N,1));
                    lab2 = uint8(zeros(N,1));
                    lab1(idx==ii) = 1;
                    lab2(inliers{ii}) = 2;
                    plot3D_labeled([x(:),y(:),z(:)],lab1);
                    title(strcat('all points - ',num2str(ii)));
                    plot3D_labeled([x(:),y(:),z(:)],lab2);
                    title(strcat('inliers only - ',num2str(ii)));
                end
                [inliers_planes(ii,:), vars(ii)] = find_plane(xyz(inliers{ii},:));
            end
        end
    end
end
clearvars i j k f im depth lab;

