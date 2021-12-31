load('calib_asus.mat');
depth1 = load('depth_1.mat');
depth2 = load('depth_2.mat');
im1 = imread('rgb_image_1.png');
im2 = imread('rgb_image_2.png');

xyz_im1=get_xyzasus(depth1(:),[480 640],1:640*480,Depth_cam.K,1,0);
%Compute "virtual image" aligned with depth
rgbd=get_rgbd(xyz,im1,R_d_to_rgb,T_d_to_rgb,RGB_cam.K);

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

    
    depth1 = depth1.depth_array;
    depth2 = depth2.depth_array;
    
    xyz_im1=get_xyzasus(depth1(:),[480 640],1:640*480,Depth_cam.K,1,0);
    xyz_im2=get_xyzasus(depth2(:),[480 640],1:640*480,Depth_cam.K,1,0);

    % p(1,:) colunas
    % p(2,:) linhas
    inds1 =(p1(:,1)-1)*3+p1(:,2);
    inds2 =(p2(:,1)-1)*3+p2(:,2);
    
    xyz1 = xyz_im1(inds1,:);
    xyz2 = xyz_im2(inds2,:);
    
    point1=pointCloud(xyz1);
    point2=pointCloud(xyz2);

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
    
    if det(R)  < 0
      R(:,3) = R(:,3)*(-1);
    end
    
        
    RP = R*(xyz_im2');
    
    P(1,:) = RP(1,:) + T(1);
    P(2,:) = RP(2,:) + T(2);
    P(3,:) = RP(3,:) + T(3);
     
    red = zeros(length(xyz_im1),3,'uint8');
    red(:,1) = 255;
    blue = zeros(length(xyz_im1),3,'uint8');
    blue(:,3) = 255;
    
    pp2=pointCloud(P','Color',red);
    pp1=pointCloud(xyz_im1,'Color',blue);
    figure(6)
    showPointCloud(pp1)
    hold on
    showPointCloud(pp2)
    