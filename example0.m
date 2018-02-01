% Sparse inverse covariance selection with Gaussian samples
%Generate problem data
clear all;
randn('seed', 0);
rand('seed', 0);

TOL=1e-6; 
toll=1e-7;


n = 100;   % number of features
N = 10*n;  % number of samples
% generate a sparse positive definite inverse covariance matrix
Sinv      = diag(abs(ones(n,1)));
idx       = randsample(n^2, 0.001*n^2);
Sinv(idx) = ones(numel(idx), 1);
Sinv = Sinv + Sinv';   % make symmetric
if min(eig(Sinv)) < 0  % make positive definite
    Sinv = Sinv + 1.1*abs(min(eig(Sinv)))*eye(n);
end
S = inv(Sinv);

% generate Gaussian samples
D = mvnrnd(zeros(1,n), S, N);
results=[];

for kkk =0.9:0.2:1.9
  q=1;
  para.alpha = kkk;
for sigma=[0.008 0.01 0.03 0.05 0.06 0.07 0.08 0.1 0.3 0.5];
 para.tau = 1.001*q*((2+para.alpha)/4);
 para.s = 0;

t_start = tic;
 [S_PID_GSADMMI, history,iter] = PID_LSADMM_I(D, sigma, TOL,toll,para);  

results=[results; para.tau para.alpha sigma iter-1 toc(t_start) history.equ(end) history.error(end)];

end
end



