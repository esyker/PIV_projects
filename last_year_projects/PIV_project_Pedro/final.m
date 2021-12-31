

imglistrgb = {'room1/rgb_0000.jpg','room1/rgb_0001.jpg','room1/rgb_0002.jpg','room1/rgb_0003.jpg','room1/rgb_0004.jpg','room1/rgb_0005.jpg','room1/rgb_0006.jpg'};
imglistdepth = {'room1/depth_0000.mat','room1/depth_0001.mat','room1/depth_0002.mat','room1/depth_0003.mat','room1/depth_0004.mat','room1/depth_0005.mat','room1/depth_0006.mat'};

cam_params = load('cam_params.mat');
%function [tforms]  = part2(imglistdepth,imglistrgb,cam_params)
num_im = length(imglistrgb);

images_vec = cell(1,num_im);
depth_vec = cell(1,num_im);


for a = 1:num_im
    images_vec{1,a} = imread(imglistrgb{a});
    depth_vec{1,a} = load(imglistdepth{a});
end
tforms = cell(num_im);
error = cell(num_im);

for i = 1:(num_im-1)
    
    for j= (i+1):num_im
        
        
        depth1 = depth_vec{i};
        depth2 = depth_vec{j};
        im1 = images_vec{i};
        im2 = images_vec{j};
        flag=0;
        
        depthK = cam_params.Kdepth;
        RGBK = cam_params.Krgb;
        R_d_to_rgb = cam_params.R;
        T_d_to_rgb = cam_params.T;
        
        depthm1 = depth1.depth_array;
        depthm2 = depth2.depth_array;
        
        if  sum(sum(isnan(depthm1))) >= 1
            flag=1;
        end
        depthm1(isnan(depthm1)) = 0;
        if flag ==0
            xyz_im1=get_xyzasus(depthm1(:),[480 640],1:640*480,depthK,1,0);
        elseif flag ==1
            xyz_im1=get_xyzasus(depthm1(:),[480 640],1:640*480,depthK,1,0)*1000;
        end
        %Compute "virtual image" aligned with depth
        rgbd1=get_rgbd(xyz_im1,im1,R_d_to_rgb,T_d_to_rgb,RGBK);
        
        depthm2(isnan(depthm2)) = 0;
        if flag==0
            xyz_im2=get_xyzasus(depthm2(:),[480 640],1:640*480,depthK,1,0);
        elseif flag==1
            xyz_im2=get_xyzasus(depthm2(:),[480 640],1:640*480,depthK,1,0)*1000;
        end
        %Compute "virtual image" aligned with depth
        rgbd2=get_rgbd(xyz_im2,im2,R_d_to_rgb,T_d_to_rgb,RGBK);
        
        im1=rgbd1;
        im2=rgbd2;
        
        
        %% Detect Corresponding points
        points1 = detectSURFFeatures(rgb2gray(im1),'NumOctaves',1,'MetricThreshold',500);
        [features1, valid_points1] = extractFeatures(rgb2gray(im1), points1);
        points2 = detectSURFFeatures(rgb2gray(im2),'NumOctaves',1,'MetricThreshold',500);
        [features2, valid_points2] = extractFeatures(rgb2gray(im2), points2);
        [indexPairs,matchmetric] = matchFeatures(features1,features2);
        matchedPoints1 = valid_points1(indexPairs(:,1),:);
        matchedPoints2 = valid_points2(indexPairs(:,2),:);
        p1=matchedPoints1.Location;
        p2=matchedPoints2.Location;
        
        p1=round(p1);
        p2=round(p2);
        
        
                
%         figure(1);
%         imagesc([im1 im2]);
%         hold on;
%         plot([p1(:,1)';p2(:,1)'+size(im1,2)],[p1(:,2)' ;p2(:,2)']);
%         
        % p(1,:) colunas
        % p(2,:) linhas
        inds1 =(p1(:,1)-1)*480+p1(:,2);
        inds2 =(p2(:,1)-1)*480+p2(:,2);
        
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
        
%         figure(2);
%         imagesc([im1 im2]);
%         hold on;
%         plot([p1(:,1)';p2(:,1)'+size(im1,2)],[p1(:,2)' ;p2(:,2)']);
        
        xyz1 = xyz(:,1:3);
        xyz2 = xyz(:,4:6);
        
        if length(p1) <10
            continue
        end 
        
        [P_1,P_2,P1,P2,flag]=ransac(xyz1,xyz2,p1,p2);
        
        if flag == 1
            continue
        end
%         figure;
%         imagesc([im1 im2]);
%         hold on;
%         plot([P1(:,1)';P2(:,1)'+size(im1,2)],[P1(:,2)' ;P2(:,2)']);
%         
        
        
        %% Procrustes Problem
       
        
        [d, Z, transform] = procrustes(P_1,P_2,'scaling',false,'reflection',false);
        
        if det(transform.T)  < 0
            transform.T(:,3) = transform.T(:,3)*(-1);
        end
        
        R = transform.T;
        t = transform.c(1,:);
        
        tforms{j,i}.R = R;
        tforms{j,i}.T = t';
         
        P_trans = transformpointcloud(xyz_im2,R,t);
        
        P_check = transformpointcloud(P_2,R,t);
      
     
        aux = zeros(length(P_1(:,1)),1);
        
        for k = 1 : length(P_1(:,1))
            aux(k) = norm([P_check(k,1) P_check(k,2) P_check(k,3)] - [P_1(k,1) P_1(k,2) P_1(k,3)]);
        end
        error{j,i} = mean(aux);
        
    end
end

%% Complete Tranformation matrix


    
for j= 3:num_im

    if isempty(tforms{j,1})
        
        tforms{j,1}.R = tforms{j-1,1}.R*tforms{j,j-1}.R;
        tforms{j,1}.T = tforms{j-1,1}.R*tforms{j,j-1}.T + tforms{j-1,1}.T;
        
    end

end
    %% imagem1
    flag=0;
    depth = depth_vec{1};
    im = images_vec{1};
    depthm = depth.depth_array;
    if  sum(sum(isnan(depthm))) >= 1
            flag=1;
    end
        depthm(isnan(depthm)) = 0;
        if flag ==0
            xyz_im=get_xyzasus(depthm(:),[480 640],1:640*480,depthK,1,0);
        elseif flag ==1
            xyz_im=get_xyzasus(depthm(:),[480 640],1:640*480,depthK,1,0)*1000;
        end
    rgbd=get_rgbd(xyz_im,im,R_d_to_rgb,T_d_to_rgb,RGBK);
        
 %% resto das imagens
for i = 2:num_im
    flag=0;
    depth1 = depth_vec{i};
    im1 = images_vec{i};
    depthm1 = depth1.depth_array;
    if  sum(sum(isnan(depthm1))) >= 1
            flag=1;
    end
        depthm1(isnan(depthm1)) = 0;
        if flag ==0
            xyz_im1=get_xyzasus(depthm1(:),[480 640],1:640*480,depthK,1,0);
        elseif flag ==1
            xyz_im1=get_xyzasus(depthm1(:),[480 640],1:640*480,depthK,1,0)*1000;
        end
        
    rgbd_final=get_rgbd(xyz_im1,im1,R_d_to_rgb,T_d_to_rgb,RGBK);
    
    P_trans = transformpointcloud(xyz_im1,tforms{i,1}.R,tforms{i,1}.T);
    
    
    pp2=pointCloud(P_trans,'Color',reshape(rgbd_final,480*640,3));
    pp1=pointCloud(xyz_im,'Color',reshape(rgbd,480*640,3));
    
    figure;
 showPointCloud(pp1);
 hold on
 showPointCloud(pp2);

end

        

% red = zeros(length(xyz_im1),3,'uint8');
% red(:,1) = 255;
% blue = zeros(length(xyz_im1),3,'uint8');
% blue(:,3) = 255;
% 
% pp2=pointCloud(P_trans,'Color',reshape(rgbd1,480*640,3));
% pp1=pointCloud(xyz_im1,'Color',reshape(rgbd1,480*640,3));
% pp3=pointCloud(xyz_im2,'Color',reshape(rgbd2,480*640,3));
% 
% %     pp2=pointCloud(P_trans,'Color',red);
% %     pp1=pointCloud(xyz_im1,'Color',blue);
% %     pp3=pointCloud(xyz_im1,'Color',reshape(rgbd2, length(xyz_im2),3));
% 
% figure(6)
% showPointCloud(pp1)
% hold on
% showPointCloud(pp2)
% title('IM 1/2')
% 
% figure(7)
% showPointCloud(pp1)
% title('IM 1')
% figure(8)
% showPointCloud(pp3)
% title('IM 2')