function transform = Ransac(idx1, idx2, xyz1, xyz2)
%   Descricao: funcao na qual esta implementado o algoritmo RANSAC que
%   permite estimar melhores paramentros, matriz de rotacao e vector de
%   translacao, para realizar a transformacao entre os pontos 3D da imagem
%   2 para o referencial da imagem 1.
%
%   Input: 
%   - idx1 - indices que determinam quais os pontos 3D das features da 
%   imagem 1 correspondentes as features resultantes do match
%   - idx2 - indices que determinam quais os pontos 3D das features da 
%   imagem 2 correspondentes as features resultantes do match
%   - xyz1 - vector de coordenada 3D de cada pontos da imagem 1
%   - xyz2 - vector de coordenada 3D de cada pontos da imagem 2
%
%   Output: 
%   - transform - estrutura com a transformacao optima entre duas imagens 

    k = 1000; %numero total de iteracoes 
    iterations = 0; %numero de iteracoes realizadas
    threshold = 0.25; %valor limite para o erro - Threshold testados 0.1 0.25 0.4 0.6 1
    max_inliers = 4; %numero maximo de inliers -> melhor modelo

    while iterations < k
        
        %--colecao de 4 pontos para calcular parametros do modelo
        
        %seleccao aleatoria de 4 pontos para determinar modelo
        sz = size(idx1,2);
        perm = randperm(sz);
        sel = perm(1:4);

        maybeinliers.a = xyz1(idx1(sel),:);
        maybeinliers.b = xyz2(idx2(sel),:);

        %calculo dos parametros do modelo
        [d, Z, transform] = procrustes(maybeinliers.a,maybeinliers.b);
            
        %--Teste do modelo para todos pontos
        alsoinliers = []; %estrutura que vai armazenar quais os pontos considerados inliers
        n_inliers = 0; %numero de inliers detectados na presente iteracao
        n_outliers = 0; %numero de outliers detectados na presente iteracao
        
        for i = 1:size(idx1,2)
            %transformacao do ponto
            xyz2_trans = transform.b*xyz2(idx2(i),:)*transform.T+transform.c(1,:);
            
            %determinacao dos inliers
            if sqrt(sum((xyz1(idx1(i),:)-xyz2_trans).^2)) < threshold
               alsoinliers.a(i-n_outliers,:) = xyz1(idx1(i),:);
               alsoinliers.b(i-n_outliers,:) = xyz2(idx2(i),:);

               n_inliers = n_inliers+1;
            else
               n_outliers = n_outliers+1; 
            end

        end
       
        %armazenamento dos parametros do modelo optimo encontrado com base
        %no modelo que possui um maior numero de inliers
        if n_inliers > max_inliers 
            inliers = alsoinliers;
            max_inliers = n_inliers;
        end
        
        iterations = iterations+1; %incrementacao da iteracoes
      
    end
    
    %calculo da tranformacao final da imagem2 para a imagem1
    [d, Z, transform] = procrustes(inliers.a,inliers.b);
    
end

