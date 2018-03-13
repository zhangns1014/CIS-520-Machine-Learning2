%load data 
fread = fopen('train.txt', 'r');
C = textscan(fread, '%f %f %f','delimiter', ',');
train_data = [C{:}];
fclose(fread);
fread = fopen('test.txt', 'r');
D = textscan(fread, '%f %f %f','delimiter', ',');
test_data = [D{:}];
fclose(fread);

fread = fopen('CrossValidation/Fold1/cv-train.txt', 'r');
C = textscan(fread, '%f %f %f','delimiter', ',');
validation_train(:,:,1) = [C{:}];
fclose(fread);
fread = fopen('CrossValidation/Fold1/cv-test.txt', 'r');
D = textscan(fread, '%f %f %f','delimiter', ',');
validation_test(:,:,1) = [D{:}];
fclose(fread);

fread = fopen('CrossValidation/Fold2/cv-train.txt', 'r');
C = textscan(fread, '%f %f %f','delimiter', ',');
validation_train(:,:,2) = [C{:}];
fclose(fread);
fread = fopen('CrossValidation/Fold2/cv-test.txt', 'r');
D = textscan(fread, '%f %f %f','delimiter', ',');
validation_test(:,:,2) = [D{:}];
fclose(fread);

fread = fopen('CrossValidation/Fold3/cv-train.txt', 'r');
C = textscan(fread, '%f %f %f','delimiter', ',');
validation_train(:,:,3) = [C{:}];
fclose(fread);
fread = fopen('CrossValidation/Fold3/cv-test.txt', 'r');
D = textscan(fread, '%f %f %f','delimiter', ',');
validation_test(:,:,3) = [D{:}];
fclose(fread);

fread = fopen('CrossValidation/Fold4/cv-train.txt', 'r');
C = textscan(fread, '%f %f %f','delimiter', ',');
validation_train(:,:,4) = [C{:}];
fclose(fread);
fread = fopen('CrossValidation/Fold4/cv-test.txt', 'r');
D = textscan(fread, '%f %f %f','delimiter', ',');
validation_test(:,:,4) = [D{:}];
fclose(fread);

fread = fopen('CrossValidation/Fold5/cv-train.txt', 'r');
C = textscan(fread, '%f %f %f','delimiter', ',');
validation_train(:,:,5) = [C{:}];
fclose(fread);
fread = fopen('CrossValidation/Fold5/cv-test.txt', 'r');
D = textscan(fread, '%f %f %f','delimiter', ',');
validation_test(:,:,5) = [D{:}];
fclose(fread);

X1=train_data(:,1:end-1);
Y1=train_data(:,end);
X2=test_data(:,1:end-1);
Y2=test_data(:,end);

%kNN
k=[1,5,9,49,99];
train_err=zeros(1,5);
test_err=zeros(1,5);
val_err=zeros(1,5);
for i=1:5
    y_train=knn(k(i),X1,Y1,X1);
    train_err(i)=classification_error(y_train,Y1);
    y_test=knn(k(i),X1,Y1,X2);
    test_err(i)=classification_error(y_test,Y2);
    
    sum_err=0;
    for j=1:5
        data1=validation_train(:,:,j);
        X3=data1(:,1:end-1);
        Y3=data1(:,end);
        data2=validation_test(:,:,j);
        X4=data2(:,1:end-1);
        Y4=data2(:,end);
        y_val=knn(k(i),X3,Y3,X4);
        sum_err=sum_err+classification_error(y_val,Y4);
    end
    val_err(i)=sum_err/5;
end

%plot
semilogx(k,train_err,'b*-');
hold on
semilogx(k,test_err,'ro-');
hold on
semilogx(k,val_err,'k*-');
title('k-NN Errors Figure');

