function K=kernel(x1,x2,para,type)
if strcmpi(type,'linear')
    K=x1*x2';
else
    if strcmpi(type,'polynomial')
        K=(x1*x2'+1).^para;
    else
        if strcmpi(type,'rbf')
             for h=1:size(x1,1)
                for g=1:size(x2,1)
                    K(h,g)=exp(-para*(norm(x1(h,:)-x2(g,:)))^2);
                end 
             end
        end
    end
end
            
end