function [y_output]=knn(k,X_train,Y_train,x_input)
Y2=Y_train';
m=size(X_train,1);
n=size(x_input,1);
d=size(X_train,2);
% a=repmat(x_input',m,1);
% b=(reshape(a,[d,m*n]))';
b=kron(x_input,ones(m,1));
X1=repmat(X_train,n,1);
dis=sum((b-X1).^2,2);
[distance,index]=sort(reshape(dis,[m,n]));
y_output=(sign(sum(Y2(index(1:k,:)),1)))';




