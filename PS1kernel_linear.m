%load data 
train_data =load('train.txt'); 
test_data=load('test.txt');
validation_train(:,:,1)=load('CrossValidation/Fold1/cv-train.txt');
validation_test(:,:,1) =load('CrossValidation/Fold1/cv-test.txt');
validation_train(:,:,2)=load('CrossValidation/Fold2/cv-train.txt');
validation_test(:,:,2) =load('CrossValidation/Fold2/cv-test.txt');
validation_train(:,:,3)=load('CrossValidation/Fold3/cv-train.txt');
validation_test(:,:,3) =load('CrossValidation/Fold3/cv-test.txt');
validation_train(:,:,4)=load('CrossValidation/Fold4/cv-train.txt');
validation_test(:,:,4) =load('CrossValidation/Fold4/cv-test.txt');
validation_train(:,:,5)=load('CrossValidation/Fold5/cv-train.txt');
validation_test(:,:,5) =load('CrossValidation/Fold5/cv-test.txt');
X1=train_data(:,1:end-1);
    Y1=train_data(:,end);
    X2=test_data(:,1:end-1);
    Y2=test_data(:,end);

%intialize the parameters
c=[];
for i=-4:2
    c=[c,10^(i)];
end

%calculating the error
validation_err=zeros(1,7);
train_err=zeros(1,7);
test_err=zeros(1,7);

    for k=1:7
        sumerr=0;
    for h=1:5
        data1=validation_train(:,:,h);
        X3=data1(:,1:end-1);
        Y3=data1(:,end);
        data2=validation_test(:,:,h);
        X4=data2(:,1:end-1);
        Y4=data2(:,end);
        [b,Z_SV,X_SV,Y_SV]=SVM(X3,Y3,c(k),'linear',0);
        K=kernel(X_SV,X4,0,'linear');
        y_hat=sign((Z_SV.*Y_SV)'*K+b)';
        sumerr=sumerr+classification_error(y_hat,Y4);      
    end
     validation_err(k)=sumerr/5;
    
     [b2,Z_SV2,X_SV2,Y_SV2]=SVM(X1,Y1,c(k),'linear',0);
      K1=kernel(X_SV2,X1,0,'linear');
      y_hat1=sign((Z_SV2.*Y_SV2)'*K1+b2)';
    
     train_err(k) = classification_error(y_hat1,Y1);
    
     K2=kernel(X_SV2,X2,0,'linear');
     y_hat2=sign((Z_SV2.*Y_SV2)'*K2+b2)';
        
    test_err(k)= classification_error(y_hat2,Y2);

    
    
    end
    [validation_min,num]=min(validation_err);
    c_min=c(num);
   
%plot errors
semilogx(c,train_err,'r*-');
hold on 
semilogx(c,test_err,'b*-');
hold on
semilogx(c,validation_err,'ko-');
title('PS1 Kernel Linear Errors');

