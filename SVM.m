
function [b,z_i,X_i,Y_i]=SVM(X,Y,C,kernel_type,kernel_para)
if strcmpi(kernel_type,'linear')
    H=(Y*Y').*kernel(X,X,kernel_para,'linear');   
else 
    if strcmpi(kernel_type,'polynomial')
        H=(Y*Y').*kernel(X,X,kernel_para,'polynomial');        
    else 
        if strcmpi(kernel_type,'rbf')
        H=(Y*Y').*kernel(X,X,kernel_para,'rbf');  
            
        
        end 
    end 
end
    m=size(X,1);    
    f=-ones(m,1);
    Aeq=Y';
    beq=0;
    lb = zeros(m,1);
    ub=C*ones(m,1);
    z = quadprog(H,f,[],[],Aeq,beq,lb,ub);
    
    Z_SV=z((z>1e-8 & z<C-1e-8),:);
    X_SV=X((z>1e-8 & z<C-1e-8),:);
    Y_SV=Y((z>1e-8 & z<C-1e-8),:);
    SV=size(Z_SV,1);
    
    X_i=X(z>1e-8,:);
    Y_i=Y(z>1e-8,:);
    z_i=z(z>1e-8,:);

   b=1/SV*sum(Y_SV-(kernel(X_SV,X_i,kernel_para,kernel_type)*(z_i.*Y_i)));
    
    
end
