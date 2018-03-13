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

%1NN
y_label_train=knn(1,X1,Y1,X1);
train_err=classification_error(y_label_train,Y1);
y_label_test=knn(1,X1,Y1,X2);
test_err=classification_error(y_label_test,Y2);

