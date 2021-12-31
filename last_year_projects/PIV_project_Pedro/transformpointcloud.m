function P = transformpointcloud(P1, R, t)


        temp = P1*R ;
        P(:,1) = temp(:,1) + t(1);
        P(:,2) = temp(:,2) + t(2);
        P(:,3) = temp(:,3) + t(3);



end 