function [ f,d ] = RemoveNearZeros( I,f,d )
%   Descricao: Funcao usada para remover features, detectadas pela funcao
%   vl_sift, que correspondam a pixeis em torno dos quais existam pixeis
%   com valor zero (sem deteccao de profundidade).
%
%   Input:
%   - I - Imagem em gray scale com precisao single
%   - f - SIFT frame (keypoints) da imagem I
%   - d - Descritores de cada SIFT frame
%
%   Output:
%   - f - SIFT frame (keypoints) da imagem I apos a remocao das features
%   sem interesse
%   - d - Descritores de cada SIFT frame apos a remocao das features


    %Remocao das features sem interesse
    ix = [];
    for i = 1:size(f,2)
        x = round(f(1,i));
        y = round(f(2,i));
        
        if x == 1 && y == 1
            if I(y,x) <= 10^-6 || I(y,x+1) <= 10^-6 || I(y+1,x) <= 10^-6 || I(y+1,x+1) <= 10^-6
               if size(ix,1) == 0
                   ix = i;
               else
                   ix = [ix, i];
               end
            end
        elseif x == 1 && y == 480
            if I(y,x) <= 10^-6 || I(y,x+1) <= 10^-6 || I(y-1,x) <= 10^-6 || I(y-1,x+1) <= 10^-6
               if size(ix,1) == 0
                   ix = i;
               else
                   ix = [ix, i];
               end
            end
        elseif x == 640 && y == 1
            if I(y,x) <= 10^-6 || I(y,x-1) <= 10^-6 || I(y+1,x) <= 10^-6 || I(y+1,x-1) <= 10^-6
               if size(ix,1) == 0
                   ix = i;
               else
                   ix = [ix, i];
               end
            end            
        elseif x == 640 && y == 480
            if I(y,x) <= 10^-6 || I(y,x-1) <= 10^-6 || I(y-1,x) <= 10^-6 || I(y-1,x-1) <= 10^-6
               if size(ix,1) == 0
                   ix = i;
               else
                   ix = [ix, i];
               end
            end
        elseif x == 1
            if I(y,x) <= 10^-6 || I(y,x+1) <= 10^-6 || I(y+1,x) <= 10^-6 || I(y-1,x) <= 10^-6 || I(y-1,x+1) <= 10^-6 || I(y+1,x+1) <= 10^-6
               if size(ix,1) == 0
                   ix = i;
               else
                   ix = [ix, i];
               end
            end            
        elseif x == 640
            if I(y,x) <= 10^-6 || I(y,x-1) <= 10^-6 || I(y+1,x) <= 10^-6 || I(y-1,x) <= 10^-6 || I(y-1,x-1) <= 10^-6 || I(y+1,x-1) <= 10^-6
               if size(ix,1) == 0
                   ix = i;
               else
                   ix = [ix, i];
               end
            end     
        elseif y == 1
            if I(y,x) <= 10^-6 || I(y+1,x) <= 10^-6 || I(y,x+1) <= 10^-6 || I(y,x-1) <= 10^-6 || I(y+1,x+1) <= 10^-6 || I(y+1,x-1) <= 10^-6
               if size(ix,1) == 0
                   ix = i;
                else
                   ix = [ix, i];
               end
            end     
        elseif y == 480
            if I(y,x) <= 10^-6 || I(y-1,x) <= 10^-6 || I(y,x+1) <= 10^-6 || I(y,x-1) <= 10^-6 || I(y-1,x+1) <= 10^-6 || I(y-1,x-1) <= 10^-6
               if size(ix,1) == 0
                   ix = i;
               else
                   ix = [ix, i];
               end
            end     
        else
            if I(y,x) <= 10^-6 || I(y,x+1) <= 10^-6 || I(y,x-1) <= 10^-6 || I(y-1,x) <= 10^-6 || I(y+1,x) <= 10^-6 || I(y-1,x+1) <= 10^-6 || I(y-1,x-1) <= 10^-6 || I(y+1,x+1) <= 10^-6 || I(y+1,x-1) <= 10^-6
               if size(ix,1) == 0
                   ix = i;
               else
                   ix = [ix, i];
               end
            end                 
        end
    end

    cnt = 0;

    for i = 1:size(ix,2)
        f(:,ix(i)-cnt) = [];
        d(:,ix(i)-cnt) = [];
        cnt = cnt+1;
    end
end

