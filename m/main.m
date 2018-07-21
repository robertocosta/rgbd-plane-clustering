close all; restoredefaultpath;
clearvars RESTOREDEFAULTPATH_EXECUTED;
delete(gcp('nocreate'))
global glob;
glob.reloadDataset = false;
glob.compute_kmeans = true;
if glob.reloadDataset
    clearvars('-except', 'glob');
    load('reduced_dataset.mat');
else
    vars = {'accelData','depths','rawDepths','images','instances',...
        'labels','names','namesToIds','sceneTypes','scenes','glob'};
    clearvars('-except', vars{:});
end
f = struct;
glob.verbose = true;
if glob.verbose
    f.f0 = figure;
    set(f.f0,'Position',[10,40,800,500]);
    f.f0q = figure;
    set(f.f0q,'Position',[10,40,800,500]);
    f.f1 = figure;
    set(f.f1,'Position',[10,40,800,500]);
    f.f2 = figure;
    set(f.f2,'Position',[10,40,850,250]);
    f.f3 = figure;
    set(f.f3,'Position',[10,40,800,500]);
%     f.f4 = figure;
%     set(f.f4,'Position',[10,40,800,500]);
    f.f5 = figure;
    set(f.f5,'Position',[10,40,850,500]);
    f.f6 = figure;
    set(f.f6,'Position',[10,40,850,250]);
    f.f7 = figure;
    set(f.f7,'Position',[50,100,1300,500]);
    f.f8 = figure;
    set(f.f8,'Position',[10,40,800,500]);
    f.f11 = figure;
    set(f.f11,'Position',[39 39 814 573]);
    f.f12 = figure;
    set(f.f12,'Position',[31 39 814 573]);
    if glob.compute_kmeans
        f.f9 = figure;
        set(f.f9,'Position',[10,40,800,500]);
        f.f10 = figure;
        set(f.f10,'Position',[10,40,800,500]);
        f.f13 = figure;
        set(f.f13,'Position',[188 40 814 573]);
        f.f14 = figure;
        set(f.f14,'Position',[258 40 814 573]);
        f.f15 = figure;
        set(f.f15,'Position',[318 40 814 573]);
        f.f16 = figure;
        set(f.f16,'Position',[212 127 814 573]);
        f.f17 = figure;
        set(f.f17,'Position',[312 127 814 573]);
        f.f18 = figure;
        set(f.f18,'Position',[188 40 814 573]);
        f.f19 = figure;
        set(f.f19,'Position',[258 40 814 573]);
        f.f20 = figure;
        set(f.f20,'Position',[318 40 814 573]);
        f.f21 = figure;
        set(f.f21,'Position',[212 127 814 573]);
        f.f22 = figure;
        set(f.f22,'Position',[312 127 814 573]);
        f.f23 = figure;
        set(f.f23,'Position',[312 127 814 573]);
    end
end
if glob.compute_kmeans
    pool = parpool;
end
% for i=[1,2,5,15,19:20]%length(scenes)]
for i=2
    depth = depths(:,:,i);
    rgb = images(:,:,:,i);
    inst = instances(:,:,i);
    lab = labels(:,:,i);
    if glob.verbose
        set(0, 'currentfigure', f.f1);
        subplot(221); imagesc(rgb); colorbar; axis image; title('RGB');
        subplot(222); imagesc(depth); colorbar; axis image; title('Depth');
%         subplot(223); imagesc(inst); colorbar; axis image; title('Inst');
        subplot(2,2,[3,4]); imagesc(lab); colorbar; axis image; title('Labels');
        f.f1.Name = sprintf('%02d_0_original',i);
        save_figures({f.f1},'dataset');
    end
    %% reprojection
    [rgb_depth, rgb_undist, pc] = project_depth_map(depth, rgb);
    pc = -pc;
    if glob.verbose
        set(0, 'currentfigure', f.f2);
        f.f2.Name = sprintf('%02d_1_reprojection',i);
        subplot(121); imagesc(rgb_undist); colorbar; axis image;
        title('RGB undistorted');
        subplot(122); imagesc(rgb_depth); colorbar; axis image;
        title('Depth reprojected');
        set(0, 'currentfigure', f.f3);
        f.f3.Name = sprintf('%02d_1_reprojection',i);
        imagesc(get_rgb_depth_overlay(rgb_undist, rgb_depth));
        title('Depth and rgb');
%         c = (pc(:,3)-min(pc(:,3)))/(max(pc(:,3))-min(pc(:,3)));
%         set(0, 'currentfigure', f.f4);
%         scatter3(pc(:,1),pc(:,3),pc(:,2),0.1,c);
%         title('Point cloud'); xlabel('x'), ylabel('z'); zlabel('y');
        save_figures({f.f2},'separated');
        save_figures({f.f3},'overlay');
    end

    %% parameters
    glob.H = size(rgb,1);
    glob.W = size(rgb,2);   
    % set window size in normals computation
    glob.wx = 25;
    glob.wy = 25;
    % number of classes for K-means
    n_cl = 6;
    % number of levels for quantization
    nLevelsCart = 6;
    nLevelsSph = 6;
    
    %% normals computation
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
        set(0, 'currentfigure', f.f5);
        f.f5.Name = sprintf('%02d_2_nCart',i);
        subplot(221);imagesc(n(:,:,1));title('nx');colorbar;axis image;
        subplot(222);imagesc(n(:,:,2));title('ny');colorbar;axis image;
        subplot(223);imagesc(n(:,:,3));title('nz');colorbar;axis image;
        subplot(224);imagesc(N2); title('|N|');colorbar;axis image;
        set(0, 'currentfigure', f.f6);
        f.f6.Name = sprintf('%02d_2_nSph',i);
        subplot(121);imagesc(phiTh(:,:,1));title('Phi');colorbar;axis image;
        subplot(122);imagesc(phiTh(:,:,2));title('Theta');colorbar;axis image;
        save_figures({f.f5},'unquantized');
        save_figures({f.f6},'unquantized');
    end
    
    %% quantization 
    % quantization of the normals in the cartesian space
    % 9 levels: [-1, -0.7, -0.5, -0.3, -0.1, 0.1 0.3, 0.5, 0.7, 1]
    from = -1;
    to = 1;
    [qnx, qnxLab] = quantization(n(:,:,1),from,to,nLevelsCart);
    [qny, qnyLab] = quantization(n(:,:,2),from,to,nLevelsCart);
    [qnz, qnzLab] = quantization(n(:,:,3),from,to,nLevelsCart);
    % quantization of the normals in the phi-theta space
    from = -pi/2;
    to = pi/2;
    [qphi, qphiLab] = quantization(phiTh(:,:,1),from*2,to*2,nLevelsSph);
    [qtheta, qthetaLab] = quantization(phiTh(:,:,2),from,to,nLevelsSph);
    
    %% clustering with SVD
    min_pts_per_plane = 2000;
    pcx = reshape(pc(:,1),glob.H,glob.W);
    pcy = reshape(pc(:,2),glob.H,glob.W);
    pcz = reshape(pc(:,3),glob.H,glob.W);
    [p, n2, conf] = compute_local_planes(pcx,pcy,pcz);%n(:,:,1),n(:,:,2),n(:,:,3));
    n2x = reshape(n2(:,:,1),numel(n2(:,:,1)),1);
    n2y = reshape(n2(:,:,2),numel(n2(:,:,1)),1);
    n2z = reshape(n2(:,:,3),numel(n2(:,:,1)),1);
    [qn2x, ~] = quantization(n2(:,:,1),from,to,nLevelsCart);
    [qn2y, ~] = quantization(n2(:,:,2),from,to,nLevelsCart);
    [qn2z, ~] = quantization(n2(:,:,3),from,to,nLevelsCart);
    tic;
    [planes, planes_idx] = xyz2planes_ransac(pcx,pcy,pcz,...
        [n2x(:),n2y(:),n2z(:)],abs(conf)>0.5,min_pts_per_plane);
    toc;
    tic;
    [planes2, planes_idx2] = xyz2planes_ransac(pcx,pcy,pcz,...
        [qn2x(:),qn2y(:),qn2z(:)],abs(conf)>0.5,min_pts_per_plane);
    toc;
    idx = cell2idx(planes_idx);
    idxIm = reshape(idx,size(n,1),size(n,2));
    set(0, 'currentfigure', f.f0);
    f.f0.Name = sprintf('%02d_3_SVD',i);
    imshow(label2rgb(idxIm));
    title('SVD');
    save_figures({f.f0},'planes');

    idx2 = cell2idx(planes_idx2);
    idxIm2 = reshape(idx2,size(n,1),size(n,2));
    set(0, 'currentfigure', f.f0q);
    f.f0q.Name = sprintf('%02d_3_SVD_quantized',i);
    imshow(label2rgb(idxIm2));
    title('SVD quantized');
    save_figures({f.f0q},'planes');
    
    if glob.compute_kmeans
        %% clustering with kmeans
        % Parallel toolbox settins
        stream = RandStream('mlfg6331_64');
        options = statset('UseParallel',1,'UseSubstreams',1,'Streams',stream);
        % Cartesian normals, nomalized between 0 and 1
        nx = n(:,:,1)/2+0.5;
        ny = n(:,:,2)/2+0.5;
        nz = n(:,:,3)/2+0.5;
        nCartTab = [nx(:),ny(:),nz(:)];
        [idxCart, ~, ~,~] = kmeans(nCartTab,n_cl,'Options',options,...
            'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
            'drop', 'OnlinePhase','on');
        % Spherical normals, normalized between 0 and 1
        phi = phiTh(:,:,1)/(2*pi)+0.5;
        theta = phiTh(:,:,2)/pi+0.5;
        nSphTab = [phi(:),theta(:)];
        [idxSph, ~, ~,~] = kmeans(nSphTab,n_cl,'Options',options,'MaxIter',10000,...
            'Display','final','Replicates',5,'EmptyAction','drop',...
            'OnlinePhase','on');
        % Cart normals + RGB normalized
        r = double(rgb_undist(:,:,1))/255;
        g = double(rgb_undist(:,:,2))/255;
        b = double(rgb_undist(:,:,3))/255;
        nCartRGBTab = [nCartTab,r(:),g(:),b(:)];
        [idxCartRGB, ~, ~,~] = kmeans(nCartRGBTab,n_cl,'Options',options,...
            'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
            'drop', 'OnlinePhase','on');
        % Sph normals + RGB
        nSphRGBTab = [nSphTab,r(:),g(:),b(:)];
        [idxSphRGB, ~, ~,~] = kmeans(nSphRGBTab,n_cl,'Options',options,...
            'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
            'drop', 'OnlinePhase','on');

        % Depth + RGB
        delta = max(rgb_depth(:))-min(rgb_depth(:));
        depthRGBTab = [(rgb_depth(:)-min(rgb_depth(:)))/delta,r(:),g(:),b(:)];
        [idxDepthRGB, ~, ~,~] = kmeans(depthRGBTab,n_cl,'Options',options,...
            'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
            'drop', 'OnlinePhase','on');
        % Depth + RGB + Cart
        depthRGBnCartTab = [(rgb_depth(:)-min(rgb_depth(:)))/delta,...
            r(:),g(:),b(:),nCartTab];
        [idxDepthRGBnCart, ~, ~,~] = kmeans(depthRGBnCartTab,n_cl,'Options',options,...
            'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
            'drop', 'OnlinePhase','on');
         % Depth + RGB + Sph
        depthRGBnSphTab = [(rgb_depth(:)-min(rgb_depth(:)))/delta,...
            r(:),g(:),b(:),nSphTab];
        [idxDepthRGBnSph, ~, ~,~] = kmeans(depthRGBnSphTab,n_cl,'Options',options,...
            'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
            'drop', 'OnlinePhase','on');
        % Plots
        if glob.verbose
            set(0, 'currentfigure', f.f9);
            idxIm = reshape(idxCart,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));
            set(f.f9,'Name',sprintf('%02d_nCart',i));
            pause(0.1);
            set(0, 'currentfigure', f.f10);
            idxIm = reshape(idxSph,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));title('K-means labels - sph');
            f.f10.Name = sprintf('%02d_nSph',i);

            set(0, 'currentfigure', f.f13);
            idxIm = reshape(idxCartRGB,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));title('K-means labels - cart + RGB');
            f.f13.Name = sprintf('%02d_nCart-RGB',i);

            set(0, 'currentfigure', f.f14);
            idxIm = reshape(idxSphRGB,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));title('K-means labels - sph + RGB');
            f.f14.Name = sprintf('%02d_nSph-RGB',i);

            set(0, 'currentfigure', f.f15);
            idxIm = reshape(idxDepthRGB,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));title('K-means labels - RGB + D');
            f.f15.Name = sprintf('%02d_Depth-RGB',i);

            set(0, 'currentfigure', f.f16);
            idxIm = reshape(idxDepthRGBnCart,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));title('K-means labels - RGB + D + cart');
            f.f16.Name = sprintf('%02d_Depth-RGB-nCart',i);

            set(0, 'currentfigure', f.f17);
            idxIm = reshape(idxDepthRGBnSph,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));title('K-means labels - RGB + D + sph');
            f.f17.Name = sprintf('%02d_Depth-RGB-nSph',i);
        end
        pause(1);
        set(0, 'currentfigure', f.f9);
        title('K-means labels - cart');
        % saving plots
        save_figures({f.f9,f.f10,f.f13,f.f14,f.f15,f.f16,f.f17},'unquantized-Kmeans');



        %% clustering with the quantized normals
        % Cartesian normals, normalized between 0 and 1
        nCartQTab = [qnx(:),qny(:),qnz(:)]/2+0.5;
        [idxCartQ, nCartQC, ~,~] = kmeans(nCartQTab,n_cl,'Options',options,...
            'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
            'drop', 'OnlinePhase','on');
        % Spherical normals, normalized between 0 and 1
        nphi = qphi/(2*pi)+0.5;
        ntheta = qtheta/pi+0.5;
        nSphQTab = [nphi(:),ntheta(:)];
        [idxSphQ, nSphQC, ~,~] = kmeans(nSphQTab,n_cl,'Options',options,'MaxIter',10000,...
            'Display','final','Replicates',5,'EmptyAction','drop',...
            'OnlinePhase','on');
        % Cart normals + RGB normalized
        nCartQRGBTab = [nCartQTab,r(:),g(:),b(:)];
        [idxCartQRGB, ~, ~,~] = kmeans(nCartQRGBTab,n_cl,'Options',options,...
            'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
            'drop', 'OnlinePhase','on');
        % Sph normals + RGB
        nSphQRGBTab = [nSphQTab,r(:),g(:),b(:)];
        [idxSphQRGB, ~, ~,~] = kmeans(nSphQRGBTab,n_cl,'Options',options,...
            'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
            'drop', 'OnlinePhase','on');
        % Depth + RGB + Cart
        depthRGBnCartQTab = [(rgb_depth(:)-min(rgb_depth(:)))/delta,...
            r(:),g(:),b(:),nCartTab];
        [idxDepthRGBnCartQ, ~, ~,~] = kmeans(depthRGBnCartQTab,n_cl,'Options',options,...
            'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
            'drop', 'OnlinePhase','on');
         % Depth + RGB + Sph
        depthRGBnSphQTab = [(rgb_depth(:)-min(rgb_depth(:)))/delta,...
            r(:),g(:),b(:),nSphTab];
        [idxDepthRGBnSphQ, ~, ~,~] = kmeans(depthRGBnSphQTab,n_cl,'Options',options,...
            'MaxIter',10000,'Display','final','Replicates',5,'EmptyAction',...
            'drop', 'OnlinePhase','on');    

        if glob.verbose
    %         set(0, 'currentfigure', f.f7);
    %         subplot(131);imshow(label2rgb(qnx));title('Quantized nx');
    %         subplot(132);imshow(label2rgb(qny));title('Quantized ny');
    %         subplot(133);imshow(label2rgb(qnz));title('Quantized nz');
    %         f.f7.Name = sprintf('%02d_nCart_raw',i);
    %         
    %         set(0, 'currentfigure', f.f8);
    %         subplot(121);imshow(label2rgb(qphi));title('Quantized phi');
    %         subplot(122);imshow(label2rgb(qtheta));title('Quantized theta');
    %         f.f8.Name = sprintf('%02d_nSph_raw',i);

            set(0, 'currentfigure', f.f18);
            idxIm = reshape(idxCartQ,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));title('K-means labels - cart quantized');
            set(f.f18,'Name',sprintf('%02d_nCart_raw',i));

            set(0, 'currentfigure', f.f19);
            idxIm = reshape(idxSphQ,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));title('K-means labels - sph quantized');
            f.f19.Name = sprintf('%02d_nSph',i);

            set(0, 'currentfigure', f.f20);
            idxIm = reshape(idxCartQRGB,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));title('K-means labels - cart quantized + RGB');
            f.f20.Name = sprintf('%02d_nCart-RGB',i);

            set(0, 'currentfigure', f.f21);
            idxIm = reshape(idxSphQRGB,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));title('K-means labels - sph quantized + RGB');
            f.f21.Name = sprintf('%02d_nSph-RGB',i);

            set(0, 'currentfigure', f.f22);
            idxIm = reshape(idxDepthRGBnCartQ,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));title('K-means labels - RGB + D + cart quantized');
            f.f22.Name = sprintf('%02d_Depth-RGB-nCart',i);

            set(0, 'currentfigure', f.f23);
            idxIm = reshape(idxDepthRGBnSphQ,size(n,1),size(n,2));
            imshow(label2rgb(idxIm));title('K-means labels - RGB + D + sph quantized');
            f.f23.Name = sprintf('%02d_Depth-RGB-nSph',i);
        end
        save_figures({f.f18,f.f19,f.f20,f.f21,f.f22,f.f23},'quantized-Kmeans');
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
                initialClustersCart{ind} = intersect(find(qnxLab==ii),...
                    intersect(find(qnyLab==j),find(qnzLab==k)));
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
    % Spherical coordinates quantized normals
    labSph = zeros(nLevelsSph^2,2);
    initialClustersSph = cell(nLevelsSph^2,1);
    ind = 1;
    for ii=1:nLevelsSph
        for j=1:nLevelsSph
            labSph(ind,:) = [ii,j];
            initialClustersSph{ind} = intersect(...
                find(qphiLab==ii),find(qthetaLab==j));
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
    % plots
    idxQuantCartIm = reshape(idxQuantCart,size(n,1),size(n,2));
    if glob.verbose
        set(0, 'currentfigure', f.f11);
        imshow(label2rgb(idxQuantCartIm));
        title('Quantized normals - cart');
        f.f11.Name = sprintf('%02d_nCart_clusters',i);
    end
    idxQuantSphIm = reshape(idxQuantSph,size(n,1),size(n,2));
    if glob.verbose
        set(0, 'currentfigure', f.f12);
        imshow(label2rgb(idxQuantSphIm));
        title('Quantized normals - spherical');
        f.f12.Name = sprintf('%02d_nSph_clusters',i);
    end
%     save_figures({f.f7,f.f8,f.f11,f.f12},'quantized-normals');
    save_figures({f.f11,f.f12},'quantized-normals');

    
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
%     waitforbuttonpress;
    disp('Next image');
end

