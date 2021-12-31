

%imglistrgb = {'room1/rgb_0000.jpg','room1/rgb_0001.jpg','room1/rgb_0002.jpg','room1/rgb_0003.jpg','room1/rgb_0004.jpg','room1/rgb_0005.jpg','room1/rgb_0006.jpg'};
%imglistdepth = {'room1/depth_0000.mat','room1/depth_0001.mat','room1/depth_0002.mat','room1/depth_0003.mat','room1/depth_0004.mat','room1/depth_0005.mat','room1/depth_0006.mat'};

imglistrgb = {'labpiv/rgb_image_1.png','labpiv/rgb_image_2.png','labpiv/rgb_image_3.png','labpiv/rgb_image_4.png','labpiv/rgb_image_5.png','labpiv/rgb_image_6.png','labpiv/rgb_image_7.png','labpiv/rgb_image_8.png','labpiv/rgb_image_9.png','labpiv/rgb_image_10.png','labpiv/rgb_image_11.png','labpiv/rgb_image_12.png','labpiv/rgb_image_13.png','labpiv/rgb_image_14.png','labpiv/rgb_image_15.png','labpiv/rgb_image_16.png','labpiv/rgb_image_17.png','labpiv/rgb_image_18.png','labpiv/rgb_image_19.png','labpiv/rgb_image_20.png','labpiv/rgb_image_21.png','labpiv/rgb_image_22.png','labpiv/rgb_image_23.png'};
 imglistdepth = {'labpiv/depth_1.mat','labpiv/depth_2.mat','labpiv/depth_3.mat','labpiv/depth_4.mat','labpiv/depth_5.mat','labpiv/depth_6.mat','labpiv/depth_7.mat','labpiv/depth_8.mat','labpiv/depth_9.mat','labpiv/depth_10.mat','labpiv/depth_11.mat','labpiv/depth_12.mat','labpiv/depth_13.mat','labpiv/depth_14.mat','labpiv/depth_15.mat','labpiv/depth_16.mat','labpiv/depth_17.mat','labpiv/depth_18.mat','labpiv/depth_19.mat','labpiv/depth_20.mat','labpiv/depth_21.mat','labpiv/depth_22.mat','labpiv/depth_23.mat'};

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
        
        
                
        figure;      
        imagesc([im1 im2]);
        hold on;
        plot([p1(:,1)';p2(:,1)'+size(im1,2)],[p1(:,2)' ;p2(:,2)']);
        title('First Match');
        
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
        
        figure;
        imagesc([im1 im2]);
        hold on;
        plot([p1(:,1)';p2(:,1)'+size(im1,2)],[p1(:,2)' ;p2(:,2)']);
         title('After RGBD');
         
        xyz1 = xyz(:,1:3);
        xyz2 = xyz(:,4:6);
        
        if length(p1) <10
            continue
        end 
        e = 0.025;
        [P_1,P_2,P1,P2,flag]=ransac(xyz1,xyz2,p1,p2,e);
        
        if flag == 1 && abs(i-j) ~= 1
            continue
        elseif flag == 1 && abs(i-j) == 1
            while flag == 1
            e = e+0.1;
            [P_1,P_2,P1,P2,flag]=ransac(xyz1,xyz2,p1,p2,e);
            end
        end
        figure;
        imagesc([im1 im2]);
        hold on;
        plot([P1(:,1)';P2(:,1)'+size(im1,2)],[P1(:,2)' ;P2(:,2)']);
         title('Ransack Match');
        
        
        
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
    pp1=pointCloud(xyz_im,'Color',reshape(rgbd,480*640,3));
    figure(1);
    showPointCloud(pp1);
    hold on
    
 %% resto das imagens
for i = 2:num_im
    flag=0;
    depth = depth_vec{i};
    im = images_vec{i};
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
        
    rgbd_final=get_rgbd(xyz_im,im1,R_d_to_rgb,T_d_to_rgb,RGBK);
    
    R = tforms{i,1}.R;
    t = tforms{i,1}.T;
    P_trans = transformpointcloud(xyz_im,R,t);
    
    figure(1)
    pp2=pointCloud(P_trans,'Color',reshape(rgbd_final,480*640,3));    
    showPointCloud(pp2);
    hold on
    figure(2)
    pp=pointCloud(xyz_im,'Color',reshape(rgbd_final,480*640,3));    
    showPointCloud(pp);
    hold on

end

        