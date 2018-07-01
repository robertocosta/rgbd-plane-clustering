close all; restoredefaultpath;
clearvars RESTOREDEFAULTPATH_EXECUTED;
delete(gcp('nocreate'))
global glob;
glob.reloadDataset = false;
if glob.reloadDataset
    clearvars('-except', 'glob');
    load('..\mat\ds50.mat');
else
    vars = {'accelData','depths','rawDepths','images','instances',...
        'labels','names','namesToIds','sceneTypes','scenes','glob'};
    clearvars('-except', vars{:});
end

glob.verbose = true;
if glob.verbose
    f1 = figure;
    set(f1,'Position',[10,40,800,500]);
    f2 = figure;
    set(f2,'Position',[850,40,500,600]);
    f3 = figure;
    set(f3,'Position',[10,40,800,500]);
    f4 = figure;
    set(f4,'Position',[10,40,800,500]);
    f5 = figure;
    set(f5,'Position',[10,40,1300,500]);
    f6 = figure;
    set(f6,'Position',[10,40,800,500]);
    f7 = figure;
    set(f7,'Position',[1400,400,1300,500]);
    f8 = figure;
    set(f8,'Position',[10,40,800,500]);
    f9 = figure;
    set(f9,'Position',[10,40,800,500]);
    f10 = figure;
    set(f10,'Position',[10,40,800,500]);
    f11 = figure;
    set(f11,'Position',[39 39 814 573]);
    f12 = figure;
    set(f12,'Position',[531 39 814 573]);
    f13 = figure;
    set(f12,'Position',[1388 407 814 573]);
    f14 = figure;
    set(f14,'Position',[1958 446 814 573]);
    f15 = figure;
    set(f15,'Position',[2218 443 814 573]);
    f16 = figure;
    set(f16,'Position',[1712 527 814 573]);
end
pool = parpool;
for i=1:length(scenes)
    depth = depths(:,:,i);
    rgb = images(:,:,:,i);
    inst = instances(:,:,i);
    lab = labels(:,:,i);
    if glob.verbose
        set(0, 'currentfigure', f1);
        subplot(221); imagesc(rgb); colorbar; axis image; title('RGB');
        subplot(222); imagesc(depth); colorbar; axis image; title('Depth');
        subplot(223); imagesc(inst); colorbar; axis image; title('Inst');
        subplot(224); imagesc(lab); colorbar; axis image; title('Labels');
    end
    %% reprojection
    [rgb_depth, rgb_undist, pc] = project_depth_map(depth, rgb);
    
    if glob.verbose
        set(0, 'currentfigure', f2);
        subplot(211); imagesc(rgb_undist); colorbar; axis image;
        title('RGB undistorted');
        subplot(212); imagesc(rgb_depth); colorbar; axis image;
        title('Depth reprojected');
        set(0, 'currentfigure', f3);
        imagesc(get_rgb_depth_overlay(rgb_undist, rgb_depth));
        title('Depth and rgb');
        c = (pc(:,3)-min(pc(:,3)))/(max(pc(:,3))-min(pc(:,3)));
        set(0, 'currentfigure', f4);
        scatter3(pc(:,1),pc(:,3),pc(:,2),0.1,c);
        title('Point cloud'); xlabel('x'), ylabel('z'); zlabel('y');
    end
    %% normals computation
    glob.H = size(rgb,1);
    glob.W = size(rgb,2);   
    % set window size in normals computation
    glob.wx = 25;
    glob.wy = 25;
    % compute normals
    N = compute_normals(pc);
    % normalize normals
    N2 = sqrt(N(:,:,1).^2+N(:,:,2).^2+N(:,:,3).^2);
    N2(N2==0) = inf;
    n = cat(3,N(:,:,1)./N2,N(:,:,2)./N2,N(:,:,3)./N2);
    % cartesian coordinates to spherical coordinates
    phiTh = cart2phiTheta(n);
    % plots
    if glob.verbose
        set(0, 'currentfigure', f5);
        subplot(221);imagesc(n(:,:,1));title('nx');colorbar;axis image;
        subplot(222);imagesc(n(:,:,2));title('ny');colorbar;axis image;
        subplot(223);imagesc(n(:,:,3));title('nz');colorbar;axis image;
        subplot(224);imagesc(N2); title('|N|');colorbar;axis image;
        set(0, 'currentfigure', f6);
        subplot(121);imagesc(phiTh(:,:,1));title('Phi');colorbar;axis image;
        subplot(122);imagesc(phiTh(:,:,2));title('Theta');colorbar;axis image;
    end
    %% clustering with kmeans
    % Parallel toolbox settins
    stream = RandStream('mlfg6331_64');
    options = statset('UseParallel',1,'UseSubstreams',1,'Streams',stream);
    % Cartesian normals
    nx = n(:,:,1)/2+0.5;
    ny = n(:,:,2)/2+0.5;
    nz = n(:,:,3)/2+0.5;
    nCartTab = [nx(:),ny(:),nz(:)];
    [idxCart, ~, ~,~] = kmeans(nCartTab,12,'Options',options,...
        'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
        'drop', 'OnlinePhase','on');
    % Spherical normals
    phi = phiTh(:,:,1)/(2*pi)+0.5;
    theta = phiTh(:,:,2)/pi+0.5;
    nSphTab = [phi(:),theta(:)];
    [idxSph, ~, ~,~] = kmeans(nSphTab,12,'Options',options,'MaxIter',10000,...
        'Display','final','Replicates',5,'EmptyAction','drop',...
        'OnlinePhase','on');
    % Cart normals + RGB
    r = double(rgb_undist(:,:,1))/255;
    g = double(rgb_undist(:,:,2))/255;
    b = double(rgb_undist(:,:,3))/255;
    nCartRGBTab = [nCartTab,r(:),g(:),b(:)];
    [idxCartRGB, ~, ~,~] = kmeans(nCartRGBTab,12,'Options',options,...
        'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
        'drop', 'OnlinePhase','on');
    % Sph normals + RGB
    nSphRGBTab = [nSphTab,r(:),g(:),b(:)];
    [idxSphRGB, ~, ~,~] = kmeans(nSphRGBTab,12,'Options',options,...
        'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
        'drop', 'OnlinePhase','on');
    % Depth + RGB
    delta = max(rgb_depth(:))-min(rgb_depth(:));
    depthRGBTab = [(rgb_depth(:)-min(rgb_depth(:)))/delta,r(:),g(:),b(:)];
    [idxDepthRGB, ~, ~,~] = kmeans(depthRGBTab,12,'Options',options,...
        'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
        'drop', 'OnlinePhase','on');
    % Depth + RGB + Cart
    
    depthRGBnCartTab = [(rgb_depth(:)-min(rgb_depth(:)))/delta,...
        r(:),g(:),b(:),nCartTab];
    [idxDepthRGBnCart, ~, ~,~] = kmeans(depthRGBnCartTab,12,'Options',options,...
        'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
        'drop', 'OnlinePhase','on');
    
    % Plots
    if glob.verbose
        set(0, 'currentfigure', f9);
        idxIm = reshape(idxCart,size(n,1),size(n,2));
        imshow(label2rgb(idxIm));title('K-means labels - cart');
        pause(0.1);
        set(0, 'currentfigure', f10);
        idxIm = reshape(idxSph,size(n,1),size(n,2));
        imshow(label2rgb(idxIm));title('K-means labels - sph');
        
        set(0, 'currentfigure', f13);
        idxIm = reshape(idxCartRGB,size(n,1),size(n,2));
        imshow(label2rgb(idxIm));title('K-means labels - cart + RGB');
        
        set(0, 'currentfigure', f14);
        idxIm = reshape(idxSphRGB,size(n,1),size(n,2));
        imshow(label2rgb(idxIm));title('K-means labels - sph + RGB');
        
        set(0, 'currentfigure', f15);
        idxIm = reshape(idxDepthRGB,size(n,1),size(n,2));
        imshow(label2rgb(idxIm));title('K-means labels - RGB + D');
        
        set(0, 'currentfigure', f16);
        idxIm = reshape(idxDepthRGBnCart,size(n,1),size(n,2));
        imshow(label2rgb(idxIm));title('K-means labels - RGB + D + cart');
    end
    %% quantization 
    % quantization of the normals in the cartesian space
    % 9 levels: [-1, -0.7, -0.5, -0.3, -0.1, 0.1 0.3, 0.5, 0.7, 1]
    nLevelsCart = 8;
    from = -1;
    to = 1;
    qnx = quantization(n(:,:,1),from,to,nLevelsCart);
    qny = quantization(n(:,:,2),from,to,nLevelsCart);
    qnz = quantization(n(:,:,3),from,to,nLevelsCart);
    % quantization of the normals in the phi-theta space
    nLevelsSph = 12;
    from = -pi/2;
    to = pi/2;
    qphi = quantization(phiTh(:,:,1),from*2,to*2,nLevelsSph);
    qtheta = quantization(phiTh(:,:,2),from,to,nLevelsSph);
    
    if glob.verbose
        set(0, 'currentfigure', f7);
        subplot(131);imshow(label2rgb(qnx));title('Quantized nx');
        subplot(132);imshow(label2rgb(qny));title('Quantized ny');
        subplot(133);imshow(label2rgb(qnz));title('Quantized nz');
        set(0, 'currentfigure', f8);
        subplot(121);imshow(label2rgb(qphi));title('Quantized phi');
        subplot(122);imshow(label2rgb(qtheta));title('Quantized theta');
    end
    
    %% initial clusters
    % Cartesian coordinates quantized normals
    labCart = zeros(nLevelsCart^3,3);
    initialClustersCart = cell(nLevelsCart^3,1);
    ind = 1;
    for ii=1:nLevelsCart
        for j=1:nLevelsCart
            for k=1:nLevelsCart
                labCart(ind,:) = [ii,j,k];
                initialClustersCart{ind} = intersect(find(qnx==ii),...
                    intersect(find(qny==j),find(qnz==k)));
                ind = ind + 1;
            end
        end
    end
    nonEmptyClusters = find(~cellfun(@isempty,initialClustersCart));
    labCart = labCart(nonEmptyClusters,:);
    clustersCart = initialClustersCart(nonEmptyClusters);
    idxQuantCart = ones(size(n,1)*size(n,2),1);
    for ii=1:length(clustersCart)
        idxQuantCart(clustersCart{ii}) = ii;
    end
    idxQuantCartIm = reshape(idxQuantCart,size(n,1),size(n,2));
    if glob.verbose
        set(0, 'currentfigure', f11);
        imshow(label2rgb(idxQuantCartIm));
        title('Quantized normals - cart');
    end
    
    % Spherical coordinates quantized normals
    labSph = zeros(nLevelsSph^2,2);
    initialClustersSph = cell(nLevelsSph^2,1);
    ind = 1;
    for ii=1:nLevelsSph
        for j=1:nLevelsSph
            labSph(ind,:) = [ii,j];
            initialClustersSph{ind} = intersect(...
                find(qphi==ii),find(qtheta==j));
            ind = ind + 1;
        end
    end
    nonEmptyClusters = find(~cellfun(@isempty,initialClustersSph));
    labSph = labSph(nonEmptyClusters,:);
    clustersSph = initialClustersSph(nonEmptyClusters);
    idxQuantSph = ones(size(n,1)*size(n,2),1);
    for ii=1:length(clustersSph)
        idxQuantSph(clustersSph{ii}) = ii;
    end
    idxQuantSphIm = reshape(idxQuantCart,size(n,1),size(n,2));
    if glob.verbose
        set(0, 'currentfigure', f12);
        imshow(label2rgb(idxQuantSphIm));
        title('Quantized normals - spherical');
    end
    %{
    To Do:
    - examine the neighborhood in the grid of each initial cluster and
    merge the cluster in which the average (non-quantized) normal
    orientation falls below a threshold
    - keeping track of the merges done and merge close clusters
    - cluster the points in each cluster w.r.t. the distances of each point
    from the origin
    - (?) train supervised learning algorithm
            labels: {4: ceiling, 11: floor, 21: wall}
        input:
            rgb2LAB
            cartNormals, z
            rgb2LAB, cartNormals, z
            sphNormals, z
    %}
    pause(1);
    disp('Next image');
end

