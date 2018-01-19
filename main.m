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
            min_pts_per_plane = 5000;
            var_gauss_filt_depth = 0.1;
            
            im = images(:,:,:,ordInd{i,3}{j,2}(k));
            depth = depths(:,:,ordInd{i,3}{j,2}(k));
            raw_depth =depth;
            depth = imgaussfilt(raw_depth,var_gauss_filt_depth);
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
                reshape(n,N,3),abs(conf)>0.5,min_pts_per_plane);
            
            n_lab = length(planes_idx);
            
            lab0 = lab(45:471, 41:601);
            idx = cell2idx(planes_idx);
            plot3D_labeled([x(:),y(:),z(:)],lab0);
            title('labeled');
            plot3D_labeled([x(:),y(:),z(:)],idx);
            title("xyz2planes-ransac");
            
            expanded_region = cell(2,1);
            lab1 = uint8(zeros(N,1));
            for ii=1:n_lab
                planes(ii,:) = planes(ii,:)/norm(planes(i,:));
%                 plane_found = false;
%                 n_points = length(planes_idx{ii});
%                 while (~plane_found)
%                     expanded_region{ii} = region_growing(xyz,planes(ii,:),...
%                         planes_idx{ii}(randi(n_points,round(n_points/6),1)),0.08);
%                     plane_found = length(expanded_region{ii})>n_points/2;
%                 end
                expanded_region{ii} = point_belongs_to_plane(xyz,...
                    planes(ii,:),0.08);
                if (length(planes_idx{ii})>min_pts_per_plane)
                    lab1(setdiff(find(expanded_region{ii}==1),...
                        find(lab1>0))) = ii;
                end

            end
            planes2 = zeros(n_lab,4);
            var_planes = zeros(n_lab,1);
            for ii=1:n_lab
                [planes2(ii,:), var_planes(ii)]= find_plane(xyz(lab1==ii,:));
            end
            [~, ordered] = sort(var_planes);
            for ii=1:n_lab
                %waitforbuttonpress;
                %close all;
                lab2 = zeros(N,1);
                lab2(lab1==ordered(ii)) = 1;
                if (length(planes_idx{ordered(ii)})>min_pts_per_plane)
                    plot3D_labeled([x(:),y(:),z(:)],lab2);
                    disp(num2str(ordered(ii)));
                end
            end
            plot3D_labeled([x(:),y(:),z(:)],lab1);
        end
    end
end
clearvars i j k f im depth lab;

