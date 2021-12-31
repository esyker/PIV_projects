%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                  --- Processamento de Imagem e Visao ---                
%                     -- "3D scanning" com um Kinect --
%
% Autora: Ana Rita Coias - 78192
% Professor: Joao Paulo Costeirs
%                                                 Epoca Especial 2017/2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [pcloud transforms] = imagematching(image_name, depth_cam, rgb_cam, Rdtrgb, Tdtrgb)
%   Descricao: funcao que ao receber um conjunto de imagens RGB e em
%   profundidade faz a reconstrucao do cenario tridimencional
%   correspondente
%
%   Input: 
%   - image_name - array de estruturas com os nomes das imagens RGB
%   e em profundidade.
%       - image_name(k).depth - string com o path do ficheiro .mat com os
%       dados da imagem em profundidade
%       - image_name(k).rgb - string com o path de um ficheiro jpe/png com
%       a imagem rgb
%   - depth_cam - estrutura com os parametro intrinsecos da camara em
%   profundidade
%       - depth_cam.k - matriz 3x3 com os parametros intrinsecos 
%       - depth_cam.DistCoef - vector 1x5 com os coeficientes de distorcao
%       das lentes
%   - rgb_cam - estrutura com os parametros intrinsecos da camara rgb
%       - rgb_cam.k - matriz 3x3 com os parametros intrinsecos 
%       - rgb_cam.DistCoef - vector 1x5 com os coeficientes de distorcao
%       das lentes
%   - Rdtrgb - matriz de rotacao 3x3
%   - Tdtrgb - matriz de rotacao 3x1
%   
%   Output: 
%   - pcloud - matriz Nx6 com os pontos 3D e dados RGB de cada
%   ponto representado no referencial do mundo
%   - transforms - array de estruturas com o mesmo comprimento que
%   image_name onde cada elemento contem a tranformacao entre a frame da
%   camara de profundidade e o referencial do mundo para a imagem k.
%       - transforms(k).R - Matriz de rotacao ente a frame da camara de
%       profundidade e o mundo da imagem k
%       - transforms(k).T - vector de translacao ente a frame da camara de
%       profundidade e o mundo da imagem k

%flag que permite o lancamento de figures
plt = 0;

for n = 1:length(image_name)-1
    
    %======================================================================
    %                               Imagem 1
    %======================================================================
   
    %Load da imagem depth
    depth_x1 = load(image_name(n).depth);
    depth_x1 = depth_x1.depth_array;

    %correccao da depth image com os parametros intrinsecos da camara
    xyz1 = get_xyzasus(depth_x1(:),[480 640],1:640*480,depth_cam.K,1,0);

    %Le imagem RGB 
    im1 = imread(image_name(n).rgb);

    %Calcula a "imagem virtual" alinhada com a profundidade
    rgbd1 = get_rgbd(xyz1,im1,Rdtrgb,Tdtrgb,rgb_cam.K);

    %Point cloud com cor em cada pixel
    cl = reshape(rgbd1,480*640,3);
    p1 = pointCloud(xyz1,'Color',cl);

    %Graficos
    if plt
        %visualizacao da imagem depth
        imagesc(depth_x1);
        title('Imagem Depth 1');

        %Display point cloud
        figure();
        p_depth1 = pointCloud(xyz1);
        showPointCloud(p_depth1);
        title('Point cloud 1');
        
        %visualizacao da imagem RGB
        figure();
        imagesc(im1);
        title('Imagem 1 RGB');

        %Imagem RGB e RGB alinhada com depth
        figure();
        imagesc([im1; rgbd1]);
        title('Imagem 1 RGB e alinhada com depth');

        %Point Cloud com cor
        figure();
        showPointCloud(p1);
        title('Point cloud 1');
    end

    %======================================================================
    %                               Imagem 2
    %======================================================================
   
    %Load da imagem depth
    depth_x2 = load(image_name(n+1).depth);
    depth_x2 = depth_x2.depth_array;

    %correccao da depth image com os parametros intrinsecos da camara
    xyz2 = get_xyzasus(depth_x2(:),[480 640],1:640*480,depth_cam.K,1,0);

    %Le imagem RGB 
    im2 = imread(image_name(n+1).rgb);

    %Calcula a "imagem virtual" alinhada com a profundidade
    rgbd2 = get_rgbd(xyz2,im2,Rdtrgb,Tdtrgb,rgb_cam.K);

    %Point cloud com cor em cada pixel
    c2 = reshape(rgbd2,480*640,3);
    p2 = pointCloud(xyz2,'Color',c2);

    %Graficos
    if plt
        %visualizacao da imagem depth
        imagesc(depth_x2);
        title('Imagem Depth 2');
        
        %Display point cloud
        figure();
        p_depth2 = pointCloud(xyz2);
        showPointCloud(p_depth2);
        title('Point cloud 2');

        %visualizacao da imagem RGB
        figure();
        imagesc(im2);
        title('Imagem 2 RGB');

        %Imagem RGB e RGB alinhada com depth
        figure();
        imagesc([im2; rgbd2]);
        title('Imagem 2 RGB e alinhada com depth');

        %Point Cloud com cor
        figure();
        showPointCloud(p2);
        title('Point cloud 2');
    end
    
    %======================================================================
    %                        Determinacao de Features
    %======================================================================
   
    %FindFeatures
    [xa,ya,xb,yb] = FindFeatures(rgbd1,rgbd2,0);

    % Converte as coordenadas dos centro da sift frame em indices do vector com
    % coordenadas xyz
    idx1 = sub2ind([480 640],round(ya),round(xa));
    idx2 = sub2ind([480 640],round(yb),round(xb));
    
    %======================================================================
    %                               RANSAC
    %======================================================================

    transform = Ransac(idx1, idx2, xyz1, xyz2);
    
    %======================================================================
    %   Transformacao dos pontos para o referencial e propagacao da 
    % transformacao
    %======================================================================
    
    if n == 1
        
        %transformacao para a primeira imagem e a identidade (referencial
        %da primeira camara e o referencial do mundo)
        transforms(n).R = [1 0 0; 0 1 0; 0 0 1];
        transforms(n).T = [0 0 0];
        
        %Concatenacao para a matriz de transformacao
        R = transform.b*transform.T;
        t = transform.c(1,:);
        transforms(n+1).R = R;
        transforms(n+1).T = t;
        P = [R' t'];
        P = [P; 0 0 0 1]; 
        
        %Transformar pontos da imagem2 para o referencial do mundo
        xyz_trans = transform.b*xyz2*transform.T+transform.c(1,:);
        
    else
        %Concatenacao para a matriz de transformacao
        R_new = transform.b*transform.T;
        t_new = transform.c(1,:);
        P_new = [R_new' t_new'];
        P_new = [P_new; 0 0 0 1];
        
        %Propagacao de transformacao
        P = P*P_new;
        
        %Separacao em matriz de rotacao e vector de translacao
        R = P(1:3,1:3);
        R = R';
        transforms(n+1).R = R;
        t = P(1:3,4);
        t = t';
        transforms(n+1).T = t;
        
        %Tranformacao dos pontos da imagem para o referencial do mundo
        xyz_trans = xyz2*R+t;
        
    end    
    
    %======================================================================
    %                        Merge das Point Clouds
    %======================================================================

    p_trans = pointCloud(xyz_trans,'Color',c2);
    
    if n == 1
        p_final = pcmerge(p1,p_trans,0.01);
    else
        p_final = pcmerge(p_final,p_trans,0.01);
    end
    
    if plt
        figure(n);
        showPointCloud(p_final);
        title(['Point cloud total im ', num2str(n), ' e im ', num2str(n+1)]);
    end
    
end
    if plt
        figure();
        showPointCloud(p_final);
        title('Point cloud total');
    end
    
    pcloud = p_final.Location;
    pcloud = [pcloud double(p_final.Color)];

end
