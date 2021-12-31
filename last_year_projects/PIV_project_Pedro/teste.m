load('calib_asus.mat');
depth1 = load('depth_1.mat');
depth2 = load('depth_2.mat');
im1 = imread('rgb_image_1.png');
im2 = imread('rgb_image_2.png');

depthm1 = depth1.depth_array;
depthm2 = depth2.depth_array;

depthm1(isnan(depthm1)) = 0;
xyz_im1=get_xyzasus(depthm1(:),[480 640],1:640*480,Depth_cam.K,1,0);
%Compute "virtual image" aligned with depth
rgbd1=get_rgbd(xyz_im1,im1,R_d_to_rgb,T_d_to_rgb,RGB_cam.K);

depthm2(isnan(depthm2)) = 0;
xyz_im2=get_xyzasus(depthm2(:),[480 640],1:640*480,Depth_cam.K,1,0);
%Compute "virtual image" aligned with depth
rgbd2=get_rgbd(xyz_im2,im2,R_d_to_rgb,T_d_to_rgb,RGB_cam.K);

im1=rgbd1;
im2=rgbd2;


%% Detect Corresponding points
    points1 = detectSURFFeatures(rgb2gray(im1));
    [features1, valid_points1] = extractFeatures(rgb2gray(im1), points1);
    points2 = detectSURFFeatures(rgb2gray(im2));
    [features2, valid_points2] = extractFeatures(rgb2gray(im2), points2);
    [indexPairs,matchmetric] = matchFeatures(features1,features2);
    matchedPoints1 = valid_points1(indexPairs(:,1),:);
    matchedPoints2 = valid_points2(indexPairs(:,2),:);
    p1=matchedPoints1.Location;
    p2=matchedPoints2.Location;
    
    p1=round(p1);
    p2=round(p2);
    
    figure(1);
    imagesc([im1 im2]);
    hold on;
    plot([p1(:,1)';p2(:,1)'+size(im1,2)],[p1(:,2)' ;p2(:,2)']);

     % p(1,:) colunas
     % p(2,:) linhas
    inds1 =(p1(:,1)-1)*3+p1(:,2);
    inds2 =(p2(:,1)-1)*3+p2(:,2);
    
    xyz1 = xyz_im1(inds1,:);
    xyz2 = xyz_im2(inds2,:);
    
%    xyz1 =  xyz1(xyz1(:,3)~=0,:);
%    xyz2 =  xyz2(xyz2(:,3)~=0,:);

    xyz = horzcat(xyz1,xyz2);

    a = xyz(:,3)~=0;
    b = xyz(:,6)~=0;
    inds = a.*b;
   
    xyz = xyz(inds(:)==1,:);
    
    p1 = p1(inds(:)==1,:);
    p2 = p2(inds(:)==1,:);

    figure(2);
    imagesc([im1 im2]);
    hold on;
    plot([p1(:,1)';p2(:,1)'+size(im1,2)],[p1(:,2)' ;p2(:,2)']);

    xyz1 = xyz(:,1:3);
    xyz2 = xyz(:,4:6);
    
    I1=pointCloud(xyz_im1);
    I2=pointCloud(xyz_im2);

    figure
    showPointCloud(I1)
    figure
    showPointCloud(I2)

    [P_1,P_2]=ransac(xyz1,xyz2);
     
%% Procrustes Problem
    
     P_1_centroid = mean(P_1,1);
     P_2_centroid = mean(P_2,1);
   
     A = zeros(length(P_1),3);
     B = zeros(length(P_2),3);
     A(:,1) = P_1(:,1) - P_1_centroid(1);
     A(:,2) = P_1(:,2) - P_1_centroid(2);
     A(:,3) = P_1(:,3) - P_1_centroid(3);
     B(:,1) = P_2(:,1) - P_2_centroid(1);
     B(:,2) = P_2(:,2) - P_2_centroid(2);
     B(:,3) = P_2(:,3) - P_2_centroid(3);
 
     At = A';
    Bt = B';
    
     [U,S,V] = svd(At*(Bt'));
 
     R = U*(V');
     
    T = P_1_centroid' - R*(P_2_centroid');

%     [d, Z, transform] = procrustes(P_1,P_2);
%     
%     if det(transform.T)  < 0
%       transform.T(:,3) = transform.T(:,3)*(-1);
%     end
%     
%     R = transform.b*transform.T;
%     t = transform.c(1,:);
%     
%     P_trans1 = transform.b*xyz_im2;
%     P_trans2 = P_trans1*transform.T ;

    P_trans1 = R*xyz_im2';
    
    for i = 1:length(P_trans2(:,1))
    P_trans(i,:) = (P_trans2(i,:))' + T;
    end
    
    
%     P = [R' t']
%     P = [P;0 0 0 1]
%     P = P'
%     
%     xyz_im2 = horzcat(xyz_im2,ones(size(xyz_im2,1),1));
%     P_trans = P * xyz_im2';
    
    
    
    aux = zeros(length(xyz_im1(:,1)),1);
    
    for i = 1 : length(xyz_im1(:,1))
        aux(i) = norm([P_trans(i,1) P_trans(i,2) P_trans(i,3)] - [xyz_im1(i,1) xyz_im1(i,2) xyz_im1(i,3)]);
    end
        avg = mean(aux);
        
    red = zeros(length(xyz_im1),3,'uint8');
    red(:,1) = 255;
    blue = zeros(length(xyz_im1),3,'uint8');
    blue(:,3) = 255;
    
    pp2=pointCloud(P_trans,'Color',reshape(rgbd1,length(xyz_im1),3));
    pp1=pointCloud(xyz_im1,'Color',reshape(rgbd2, length(xyz_im1),3));
    figure(6)
    showPointCloud(pp1)
    hold on
    showPointCloud(pp2)
    