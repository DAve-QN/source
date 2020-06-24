function g = grad_fn(x, ix, X, y, lambda)

%%% gradient of logistic regression with regularization parameter lambda

Xin = X(ix,:);
yin = y(ix,:);
v = (Xin*x).*yin;
e = exp(v);
g = 1./(1+e);
g = Xin'*(-yin.*g);
g=g/length(ix)+ lambda*x;

end