function [xa,ya,xb,yb] = FindFeatures(im1,im2,plt)
%   Descricao: funcao que tem como objectivo determina quais as features
%   que serao usadas para calculo das transformacoes entre duas imagens.
%   Realiza extraccao de features, remocao de features sem interesse, match
%   de features e filtragem das mesmas pela seleccao de features
%   correspondentes com uma distancia a baixo de um valor limiar.
%
%   Input: - im1 - Imagem rgb alinha com a respectiva imagem em
%   profundidade da imagem 1 - im2 - Imagem rgb alinha com a respectiva
%   imagem em profundidade da imagem 2 - plt - flag que permite o
%   lancamento das figuras relativas as features encontradas e ao match das
%   features quanto tem valor 1 (nao ha lancamento de figuras se a flag
%   estiver a zero)

   %single precision gray scale image
    Ia = im2single(rgb2gray(im1));
    Ib = im2single(rgb2gray(im2));

    %Compute SIFT frames (keypoints) e os descritores
    [fa, da] = vl_sift(Ia);
    [fb, db] = vl_sift(Ib);
    
    %Remocao de features sem interesse
    [fa, da] = RemoveNearZeros(Ia,fa,da);
    [fb, db] = RemoveNearZeros(Ib,fb,db);

    %Visualizacao da amostra de features
    if plt
        
        %Seleccao de 50 features aleatorias
        sz1 = size(fa,2);
        perm1 = randperm(sz1);
        sel1 = perm1(1:50);
        sz2 = size(fb,2);
        perm2 = randperm(sz2);
        sel2 = perm2(1:50);
        
        figure();
        imagesc(im1);
        h1 = vl_plotframe(fa(:,sel1));
        set(h1,'color','y','linewidth',3);

        figure();
        imagesc(im2);
        h2 = vl_plotframe(fb(:,sel2));
        set(h2,'color','y','linewidth',3);
    end

    %Match entre features
    [matches, scores] = vl_ubcmatch(da, db); %Valores observados 10.000 20.000 50.000 80.000 100.000
    
    % (Filtragem) Retira matches com um score acima de xx - considerados
    % maus
    index=find(scores<20000);
    matches(:,index) = [];

    %Amostras das features correspondentes
    samples = 0;
    if samples
        sz = size(matches,2);
        perm = randperm(sz);
        sel = perm(1:10);
        matches = matches(:,sel);
    end

    %Features que fazem match
    xa = fa(1,matches(1,:));
    ya = fa(2,matches(1,:));
    xb = fb(1,matches(2,:));
    yb = fb(2,matches(2,:));

    %visualizacao do match entre features
    if plt 

        figure(); clf;
        imagesc(cat(2, im1, im2));  

        hold on ;
        h = line([xa ; xb+640], [ya ; yb]);
        set(h,'linewidth', 1, 'color', 'b');

        vl_plotframe(fa(:,matches(1,:)));
        fb(1,:) = fb(1,:) + size(Ia,2);
        vl_plotframe(fb(:,matches(2,:)));
        hold off;
    end    

end

