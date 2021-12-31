%% Create array of images



image_list = {'viana1.jpg','viana2.jpg','viana3.jpg','viana4.jpg'};
%function [H]  = part1(image_list,match_list)

num_im = length(image_list);

images_vec = cell(num_im,1);

nsize = 480;


for a = 1:num_im
   im = imread(image_list{a});
   images_vec{1,a} = imresize(im,[960 NaN]);
    
end 
H = cell(num_im);

for k = 1:(num_im - 1)
    
    %     if k+2 > num_im
    
   for j = (k+1):(num_im)
        
        im1 = images_vec{1,k};
        im2 = images_vec{1,j};
        
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
        
       
        
        if length(matchmetric) < 100 && k+1 ~= j
            H{k,j} = zeros(3);
        else
            H_temp = estimateGeometricTransform(matchedPoints1,matchedPoints2,'projective','Confidence',99.9,'MaxNumTrials',2000);
            
            A = estimateGeometricTransform(matchedPoints1,matchedPoints2,'affine','Confidence',99.9,'MaxNumTrials',2000);
             
             aux_c1 = ones(length(p1),1);
             aux_c2 = ones(length(p2),1);
             p1_c = cat(2, p1, aux_c1);
             p2_c = cat(2, p2, aux_c2);
             p1_h = p1_c*H_temp.T;
             p1_a = p1_c*A.T;
             
             diff_h = p2_c - p1_h;
             diff_a = p2_c - p1_a;
             err_h=sqrt(sum(diff_h.^2,2));
             err_a=sqrt(sum(diff_a.^2,2));
             SSEH = sum(err_h.^2);
             SSEA = sum(err_a.^2);
             
             if SSEH > SSEA
                 H{k,j} = A;
             else
                H{k,j} = H_temp;
                 
             end
            
        end
    end
    
end
%     else
%         for j = (k+1):(k+2)
%
%             im1 = imread(image_list{k});
%             im2 = imread(image_list{j});
%
%             %% Detect Corresponding points
%             points1 = detectSURFFeatures(rgb2gray(im1));
%             [features1, valid_points1] = extractFeatures(rgb2gray(im1), points1);
%             points2 = detectSURFFeatures(rgb2gray(im2));
%             [features2, valid_points2] = extractFeatures(rgb2gray(im2), points2);
%             [indexPairs,matchmetric] = matchFeatures(features1,features2);
%             matchedPoints1 = valid_points1(indexPairs(:,1),:);
%             matchedPoints2 = valid_points2(indexPairs(:,2),:);
%             p1=matchedPoints1.Location;
%             p2=matchedPoints2.Location;
%
%             if length(matchmetric) < 30
%                 H{k,j} = zeros(3);
%             else
%                 H_temp = estimateGeometricTransform(matchedPoints1,matchedPoints2,'projective','Confidence',99.9,'MaxNumTrials',2000);
%
%                 A = estimateGeometricTransform(matchedPoints1,matchedPoints2,'affine','Confidence',99.9,'MaxNumTrials',2000);
%
%                 aux_c1 = ones(length(p1),1);
%                 aux_c2 = ones(length(p2),1);
%                 p1_c = cat(2, p1, aux_c1);
%                 p2_c = cat(2, p2, aux_c2);
%                 p1_h = p1_c*H_temp.T;
%                 p1_a = p1_c*A.T;
%
%                 diff_h = p2_c - p1_h;
%                 diff_a = p2_c - p1_a;
%                 err_h=sqrt(sum(diff_h.^2,2));
%                 err_a=sqrt(sum(diff_a.^2,2));
%                 SSEH = sum(err_h);
%                 SSEA = sum(err_a);
%
%                 if SSEH > SSEA
%                     H{k,j} = A;
%                 else
%                     H{k,j} = H_temp;
%
%                 end
%
%             end
%         end
%     end


for i = 1:num_im
    
    for j = 1:num_im
        
        if(i == j)
            
            H{i,j} = projective2d(eye(3));
        end
        
        if (i > j)
            
            H{i,j} = projective2d(inv(H{j,i}.T)) ;
        end
        
        if (i < j)  &&  isequal(H{i,j},zeros(3))
           
             
            H{i,j} = projective2d(H{i,j-1}.T*H{j-1,j}.T);
        end
        
    end
    
end

 middle = round((num_im+1)/2);
isize = [960 480 3] ;
    
 %my_panorama([H{:,middle}],isize , images_vec);
%% Plot matching points
% figure(1);imagesc([im1 im2]);
% hold on;plot([p1(:,1)';p2(:,1)'+size(im1,2)],[p1(:,2)' ;p2(:,2)']);

%%