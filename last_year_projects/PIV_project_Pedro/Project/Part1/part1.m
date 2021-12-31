function [H worldpoints_x worldpoints_y ]=part1(image_list, match_list,K,pointsw)
num_im = length(image_list);

images_vec = cell(num_im,1);


for a = 1:num_im
   im = imread(image_list{a});
   images_vec{1,a} = imresize(im,[1024 NaN]);
    
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
        
        if length(matchmetric) < 200 && k+1 ~= j
            H{k,j} = zeros(3);
        else
            H_temp = estimateGeometricTransform(matchedPoints1,matchedPoints2,'projective','Confidence',99.9,'MaxNumTrials',2000);
            
            H{k,j} = H_temp.T;
            
        end
    end
    
end


for i = 1:num_im
    
    for j = 1:num_im
        
        if(i == j)
            
            H{i,j} = eye(3);
        end
        
        if (i > j)
            
            H{i,j} = inv(H{j,i}) ;
        end
        
        if (i < j)  &&  isequal(H{i,j},zeros(3))
           
             
            H{i,j} = H{i,j-1}*H{j-1,j};
        end
        
    end
    
end

K1 = inv(K);
points=cat(2,pointsw,ones(4,1));

worldpoints_x = zeros(num_im, 4);
worldpoints_y = zeros(num_im, 4);

for k = 1:num_im

    b = H{1,k};
    worldpoints = K1*b*points';

    w1 = worldpoints(1,:);
    w2 = worldpoints(2,:);
    w3 = worldpoints(3,:);
    
    x = w1./w3;
    y = w2./w3;

    worldpoints_x(k,:) = x;
    worldpoints_y(k,:) = y;
end
end