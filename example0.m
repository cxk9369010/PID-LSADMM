% Sparse inverse covariance selection with Gaussian samples
%Generate problem data
clear all;
randn('seed', 0);
rand('seed', 0);
%% Table 3
%%
% TOL=1e-3; 
%     toll=1e-7;
%     toll=1e-12;
%%
% TOL=1e-6; 
%     toll=1e-8;
%     toll=1e-14;
%%
% TOL=1e-9; 
%     toll=1e-7;
%     toll=1e-15;
%% Fig5.1-5.2
TOL=1e-7; 
toll=1e-15;

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

%for kkk =0.1:0.2:1.9
  q=2;  
 para.alpha = 0.6;
% para.tau = 1.001*((2+para.alpha)/4);
%for sigma=[0.008 0.01 0.03 0.05 0.06 0.07 0.08 0.1 0.3 0.5];
% para.tau = 1.001*2;
 para.tau = 1.001*q*((2+para.alpha)/4);
 %for sigma=[0.008 0.03 0.05 0.07 0.08 0.09 0.1 0.11 0.3 0.5];
sigma=0.1;
para.s = 0;

t_start = tic;
%%
%[S_GSADMMIII, history,iter] = GSADMMIII(D, 0.05, TOL,toll,para); %iter  best
%[S_PID_GSADMM, history,iter] = PID_GSADMM(D, sigma, TOL,toll,para); %iter  best
[S_PID_GSADMMII, history,iter] = PID_GSADMM_II(D, sigma, TOL,toll,para); %iter  best

results=[results; para.tau para.alpha sigma iter-1 toc(t_start) history.equ(end) history.error(end)];

%end



%[S_PID_GSADMMII, history_PID_GSADMMII] = PID_GSADMM_II(D, 0.05, TOL,toll,para); %iter  best
%[S_PJALM, history_PJALM] = PJALM(D, 0.05, TOL,toll);
%[S_HTY, history_HTY] = HTY(D, 0.05, TOL,toll);
%[S_Bai, history_Bai] = GR_PPA(D, TOL,toll);
%[S_TADM, history_TADM] = TADMM(D, 0.05, TOL,toll);
