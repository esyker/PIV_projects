P_1=[1 2 3; 4 5 6; 7 8 9];
P_2 =[3 7 6;8 9 4; 6 5 1];

Size1 = size(P_1);
Size2 = size(P_2);

P_1_centroid = mean(P_1,1)
P_2_centroid = mean(P_2,1)

P_1aux = P_1 - P_1_centroid
P_2aux = P_2 - P_2_centroid;

[U,S,V] = svd(P_1aux.*P_2aux');

R = U*V.';
Test = R*P_2aux
T = P_1_centroid - R.*P_2_centroid;

XD = R.* P_2 + T;





