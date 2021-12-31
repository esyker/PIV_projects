function [inliers1 inliers2,P1,P2,noinliers] = ransac2(xyz1,xyzk,p1,p2)
%%%%%%%%%%%%%%ransac for 3D planes

errorthresh=0.025;
niter=100;
RTs = [];
maxinliers = 0;
noinliers = 0;
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
    [d,Z,transform]=procrustes(y,pts,'scaling',false,'reflection',false);
    %RTs=[RTs RT];
    %dist = (xyz1-projxyz*RT);
    R = transform.T;
    t = transform.c(1,:);
    P = [R' t'];
    P = [P;0 0 0 1];
    P = P';
    
    RTs = [RTs P];
    
    
    P_trans1 = xyzk;
    P_trans2 = P_trans1*R ;
    P_trans(:,1) = P_trans2(:,1) + t(1);
    P_trans(:,2) = P_trans2(:,2) + t(2);
    P_trans(:,3) = P_trans2(:,3) + t(3);
    
    dist = xyz1 - P_trans;
    
    dist = dist.^2;
    erro = sqrt(sum(dist,2));
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
if maxinliers<4
    
    noinliers = 1;
    
end
    
inliers1 = zeros(length(bestinds),3);
inliers2 = zeros(length(bestinds),3);
P1 = zeros(length(bestinds),2);
P2 = zeros(length(bestinds),2);
    for j = 1:length(bestinds)
        
        inliers1(j,:) = xyz1(bestinds(j),:);
        inliers2(j,:) = xyzk(bestinds(j),:);
        P1(j,:) = p1(bestinds(j),:);
        P2(j,:) = p2(bestinds(j),:);
        
       
    end


end