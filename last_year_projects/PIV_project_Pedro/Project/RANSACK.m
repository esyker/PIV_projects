

load('p1.mat')
load('p2.mat')


nsize = 480;

P = 0.99;
p= 0.6;
n=3;
k = log(1-P)/log(1-p^n);

errorthresh=0.2;
niter=100;

aux=fix(rand(4*niter,1)*length(p1))+1;

Hs=[];
numinliers=[];
uv=[];






        
   for i=1:niter-3
    pts = p1(aux(4*i:4*i+3),:);
    pts2= p2(aux(4*i:4*i+3),:);
    
    y = reshape(pts2',[],1);
    %pseudoinversa
    a = horzcat(pts(:,:),ones(size(pts,1),1));
    line = reshape(a,1,[]);
    a1 = cat(1,line,zeros(1,length(line)));
    a2 = cat(1,zeros(1,length(line)),line);
    left = reshape(a1,8,[]);
    middle = reshape(a2,8,[]);
    up = -pts.*[pts2(:,1) pts2(:,1)];
    down = -pts.*[pts2(:,2) pts2(:,2)];
    up1 = reshape(up,1,[]);
    down1 = reshape(down,1,[]);
    imp = cat(1,up1,down1);
    right=reshape(imp,8,[]);
    A = horzcat(left,middle,right);
    At = A'*A;
    H_vec = (A'\At)*y;
    H=reshape([H_vec;1],3,3)';
    Hs=[Hs H_vec];
    x1 = horzcat(p1(:,:),ones(size(p1,1),1))';
    x2 = p2';
    uvd = H*x1;
    
    uv1 = uvd(1,:)./uvd(3,:);
    uv2 = uvd(2,:)./uvd(3,:);
    uv = [uv1;uv2];
    erro = sqrt(sum((x2-uv).^2,1));
    inds=find(erro<errorthresh);
    numinliers=[numinliers length(find(erro<errorthresh))];
   end





%end