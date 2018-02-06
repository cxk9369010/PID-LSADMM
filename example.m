% Sparse inverse covariance selection with Gaussian samples
% Optimal linearized ADMM for multi-block separable convex programming
% Written by Xiaokai Chang email:15293183303@163.com
clear all;
randn('seed', 0);
rand('seed', 0);

results=[];
for method = 1:5; %%%  1--- LSADMM_1-2;  2---GSADMMIII; 3---PJALM;  4---TADMM; 5---HTY
TOL=1e-12; 
toll=1e-11;

n = 100;   % number of features
N = 10*n;  % number of samples
%% generate a sparse positive definite inverse covariance matrix
Sinv      = diag(abs(ones(n,1)));
idx       = randsample(n^2, 0.001*n^2);
Sinv(idx) = ones(numel(idx), 1);
Sinv = Sinv + Sinv';   % make symmetric
if min(eig(Sinv)) < 0  % make positive definite
    Sinv = Sinv + 1.1*abs(min(eig(Sinv)))*eye(n);
end
S = inv(Sinv);

%% generate Gaussian samples
D = mvnrnd(zeros(1,n), S, N);

%%  different starting points

% para.X = eye(n,n);para.S =2*eye(n,n); para.L = eye(n,n); para.lambda = zeros(n); %table 3 
%para.X = zeros(n);para.S =zeros(n); para.L = zeros(n); para.lambda = zeros(n);
 para.X = eye(n,n);para.S = eye(n,n);  para.L = zeros(n);  para.lambda = zeros(n); %feasible point
% para.X = eye(n,n);para.S =4*eye(n,n); para.L = 3*eye(n,n); para.lambda = zeros(n);


t_start = tic;
%%
if method ==1
%      q=1;
%      sigma=0.07;
%      para.alpha = 1.9;
%      para.tau = 1.001*q*((2+para.alpha)/4);
%      [S_PID_GSADMMI, history,iter] = PID_LSADMM_I(D, sigma, TOL,toll,para);
   q=2;
   sigma=0.12;
   para.alpha = 1.7;
   para.tau = 1.001*q*((2+para.alpha)/4);
   [PID_GSADMMII_history,iter] = PID_LSADMM_II(D, sigma, TOL,toll,para); 
elseif method ==2
    sigma=0.05;
    [GSADMMIII_history,iter] = GSADMMIII(D, sigma, TOL,toll,para);
elseif method ==3
    sigma=0.05;
    [PJALM_history,iter] = PJALM(D, sigma, TOL,toll,para);
elseif method ==4
    sigma=0.05;
    [TADM_history,iter] = TADMM(D,sigma, TOL,toll,para);
elseif method ==5
    sigma=0.05;
    [HTY_history,iter] = HTY(D, sigma, TOL,toll,para);
end    

%results=[results; para.tau para.alpha sigma iter-1 toc(t_start) history.equ(end) history.error(end)];
end
figure(1)
 plot(PID_GSADMMII_history.equ,'r');
 hold on
 plot(GSADMMIII_history.equ,'g');  plot(PJALM_history.equ,'b'); plot(TADM_history.equ,'k'); plot(HTY_history.equ,'m');
 xlabel('Iteration');
 ylabel('Infeasibility (IER)')
 legend('LSADMM-1-2','GS-ADMM-III','PJALM','TADMM','HTY')
 figure(2)
 plot(PID_GSADMMII_history.error,'r');
 hold on
 plot(GSADMMIII_history.error,'g');  plot(PJALM_history.error,'b'); plot(TADM_history.error,'k'); plot(HTY_history.error,'m');
 xlabel('Iteration');
 ylabel('Iteration error (RelChg)')
 legend('LSADMM-1-2','GS-ADMM-III','PJALM','TADMM','HTY')
 




