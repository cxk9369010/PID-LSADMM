% Sparse inverse covariance selection with Gaussian samples
%Generate problem data
clear all;
randn('seed', 0);
rand('seed', 0);

results=[];
for method = 1:5; %%%  1--- PID_LSADMM_II;  2---GSADMMIII; 3---PJALM;  4---TADMM; 5---HTY
TOL=1e-10; 
toll=1e-9;

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




t_start = tic;
%%
if method ==1
%     q=2;
%     sigma=0.05;
%     para.alpha = 1.99;
%     para.tau = 1.001*q*((2+para.alpha)/4);
%     [S_PID_GSADMMII, history,iter] = PID_LSADMM_I(D, sigma, TOL,toll,para); %iter  best
    q=2;
    sigma=0.12;
    para.alpha = 1.7;
    para.tau = 1.001*q*((2+para.alpha)/4);
    [S_PID_GSADMMII, history,iter] = PID_LSADMM_II(D, sigma, TOL,toll,para); %iter  best
elseif method ==2
    sigma=0.05;
    [S_GSADMMIII, history,iter] = GSADMMIII(D, sigma, TOL,toll,para); %iter  best
elseif method ==3
    sigma=0.05;
    [S_PJALM, history,iter] = PJALM(D, 0.05, TOL,toll);
elseif method ==4
    sigma=0.05;
    [S_TADM, history,iter] = TADMM(D, 0.05, TOL,toll);
elseif method ==5
    sigma=0.05;
    [S_HTY, history,iter] = HTY(D, 0.05, TOL,toll);
end    



results=[results; para.tau para.alpha sigma iter-1 toc(t_start) history.equ(end) history.error(end)];
end


