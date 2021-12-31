function [inliers1 inliers2] = ransac(xyz1,xyzk)
%%%%%%%%%%%%%%ransac for 3D planes

errorthresh=0.25;
niter=100;
RTs = [];
maxinliers = 0
numinliers = [] ;
projxyz = horzcat(xyzk,ones(size(xyzk,1),1));
%generate sets of 4 points (randomly selected)
aux=fix(rand(4*niter,1)*length(xyz1))+1;
for i=1:niter-3,
    pts = xyzk(aux(4*i:4*i+3),:);
    y = xyz1(aux(4*i:4*i+3),:);
    %pseudoinversa
    %A = horzcat(pts,ones(size(pts,1),1));
    %atemp = A'*A;
    %RT=(A'/atemp)*y;
    %RT=inv(atemp) * A' * y
    [d,Z,transform]=procrustes(y,pts)
    %RTs=[RTs RT];
    %dist = (xyz1-projxyz*RT);
    R = transform.b*transform.T;
    t = transform.c(1,:);
    P = [R' t'];
    P = [P;0 0 0 1];
    P = P';
    
    RTs = [RTs P];
    
    P_trans = transform.b*xyzk*transform.T + transform.c(1,:);
    
    dist = xyz1 - P_trans;
    
    %%dist = dist.^2;
    %erro = sqrt(sum(dist,2));
    for j = 1 : length(dist(:,1))
        erro(j) = norm(dist(j,:));
    end
    
    inds = find(erro<errorthresh);
    
    if length(inds) > maxinliers
        maxinliers = length(inds);
        bestinds = inds;
    end
    %numinliers=[numinliers length(find(erro<errorthresh))];
end
%%

% [~,ind]=max(numinliers);
% ite = (ind-1)*3 + 1;
% RT=RTs(:,ite:ite+2);
% %dist = (xyz1-projxyz*RT);
% %dist = dist.^2;
% for j = 1 : length(dist(:,1))
%         erro(j) = norm(dist(j,:));
% end
% %erro= sqrt(sum(dist,2));
% inds=find(erro<errorthresh);


inliers1 = zeros(length(bestinds),3);
inliers2 = zeros(length(bestinds),3);

    for j = 1:length(bestinds)
        
        inliers1(j,:) = xyz1(bestinds(j),:);
        inliers2(j,:) = xyzk(bestinds(j),:);
        
       
    end


%end