clc;
close all;
clear

%%% You should have LIBSVM library installed.
%%% please check the website: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

%% Loading Data
[label,X]=libsvmread('../dataset/mnist');
N = length(label);
d = size(X,2);
maxX = max(X);
minX = min(X);
delta = maxX - minX;
% normalize data
for i = 1:d
    if(delta(i)~=0)
        X(:,i) =(X(:,i)-minX(i))/delta(i);
    end
end
%

%% Global Parameters
numIter = 40;
lambda = 1;
numProcesses = 4;
freq = 1; % frequency of computing objective function value

%% Problem Definition(logistic regression)
% define objective function here and the gradient in "grad_fn"
obj_fn = @(w,ix) ( sum(log(1+exp(-label(ix).*(X(ix,:)*w))))/length(ix) +  lambda/2 * norm(w)^2 );

%% Initialization
% Bw: B for workers is a 3 dimesional matrix
% with 3rd dimension being the processor index.
% Binv: Binv for master
% Nproc: number of samples per processor
% Ind: indices of data for each processor(evenly distributed)

Nproc = floor(N/numProcesses);
x = zeros(d,1);
u = zeros(d,1);
z = zeros(d, numProcesses);
g = zeros(d, 1);

for i = 1:numProcesses
    Bw(:,:,i) = eye(d);
    u = u + Bw(:,:,i) * x;
    Ind(i,:) = (1 : Nproc) + (i-1) * Nproc;
    g = g + grad_fn(x, Ind(i,:), X, label, lambda);
end

Binv = eye(d)/numProcesses;

% take one step of GD for better initialization
x = x - 0.001 * g/numProcesses;
xi = repmat(x,1, numProcesses); % workers initial x


% array containing each processor update
% each sends 3 vectors(du, yi, qi) and two scalars(alpha, beta)
update = zeros(3*d + 2, numProcesses);
obj_vals = [];
for t = 0:numIter-1

    %%% worker's computation
    for i = 1:numProcesses
        % get current values
        zi = z(:,i);
        indices = Ind(i,:);
        Bi = Bw(:,:,i);
        x = xi(:,i);
        
        % main computation
        si = x - zi;
        yi = grad_fn(x, indices, X, label, lambda ) - grad_fn(zi, indices, X, label, lambda  );
        qi = Bi*si;
        alpha = yi'*si;
        beta = si'*(Bi*si);
        ui = Bi*zi;
        Bi = Bi + (yi*yi')/alpha - (qi*qi')/beta;
        du = Bi*x - ui;
        
        % replace previous values
        z(:,i) = x;
        Bw(:,:,i) = Bi;
        
        update(:,i) = [du;yi;qi;alpha;beta];
    
    
    %%% master's computation
        du = update(1:d,i);
        y = update(d+1:2*d,i);
        q = update(2*d+1:3*d,i);
        alpha = update(3*d+1,i);
        beta = update(3*d+2,i);
        
        u = u + du;
        g = g + y;
        v = Binv * y;
        U = Binv - v*v'/(alpha + v'*y);
        w = U * q;
        Binv = U + w*w'/(beta - q'*w);
        x = Binv * (u - g);
        
        % send x back to worker:
        xi(:,i) = x;
        
        %
        if (mod(t, freq) == 0)
            obj = obj_fn(x, 1:N);
            obj_vals = [obj_vals obj];
        end
    end
    
end
figure()
plot((1:length(obj_vals))*freq, obj_vals, 'LineWidth', 3)
xlabel 'iteration'
ylabel 'value'
title 'objective function value over iterations'