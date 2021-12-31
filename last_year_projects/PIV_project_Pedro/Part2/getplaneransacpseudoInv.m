%%


%%%%%%%%%%%%%%ransac for 3D planes
errorthresh=0.25;
niter=100;
%generate sets of 4 points (randomly selected)
aux=fix(rand(4*niter,1)*length(xyz))+1;
%%
planos=[];
numinliers=[];
for i=1:niter-3
    pts=xyz(aux(4*i:4*i+3),:);
    %pseudoinversa
    A = horzcat(pts,ones(size(2,pts),1))
    plano=inv(A'*A)*A'*pts(:,3);
    planos=[planos plano];
    erro=abs(xyz(:,3)-[xyz(:,1:2) ones(length(xyz),1)]*plano);
    inds=find(erro<errorthresh);
    numinliers=[numinliers length(find(erro<errorthresh))];
end
%%
figure(2);
[mm,ind]=max(numinliers);
fprintf('Maximum num of inliers %d \n',mm);
plano=planos(:,ind);
erro=abs(xyz(:,3)-[xyz(:,1:2) ones(length(xyz),1)]*plano);
inds=find(erro<errorthresh);
A=[xyz(inds,1:1) ones(length(inds),1)];
planofinal= inv(A'*A)*A'*xyz(inds,3);
pc2=pointCloud(xyz(inds,:),'Color',uint8(ones(length(inds),1)*[255 0 0]));
showPointCloud(pc);
hold; 
showPointCloud(pc2);

