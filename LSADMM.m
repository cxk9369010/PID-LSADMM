function [history] = LSADMM(C, beta, para)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file implements the LSADMM algorithm described in 
% "Optimal linearized ADMM for multi-block separable convex programming"
% by Xiaokai Chang, Jianchao Bai, Sanyang Liu and Dunjiang Song, for solving 
% Latent Variable Gaussian Graphical Model Selection  
% min <R,SigmaO> - logdet(R) + alpha ||S||_1 + beta Tr(L) 
% s.t. R = S - L,  R positive definte,  L positive semidefinite 
% 
% Authors: Xiaokai Chang, Jianchao Bai, Sanyang Liu and Dunjiang Song
% covsel  Sparse inverse covariance selection via LSADMM
% written by Xiaokai Chang Email: xkchang@lut.cn 
% Date: Nov 25, 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fprintf('Algorithm:  LSADMM\n');

%% Global constants and defaults
QUIET    = 1;
PRINT_FINAL = 1 ;
MAX_ITER = para.maxiter;

%% Data preprocessing

n = size(C,1);
%%%%%%%%%%%%%%%%%%%%%%%%
%%
 
alpah = para.alpha;  
tau   = para.tau;
TOL   = para.TOL;
tol1  =  para.tol1;
r_L = 1.001*beta;
r_S = 1.001*beta;
 
%% two parameters
v=para.v; 
nu=para.nu;


%% ADMM inital values
EY= eye(n,n); %identity matrix

%X = eye(n,n);S =2*eye(n,n); L = eye(n,n); lambda = zeros(n); %table 3 
% X = zeros(n);S =zeros(n); L = zeros(n); lambda = zeros(n);
 X = eye(n,n);S = eye(n,n);  L =zeros(n);  lambda = zeros(n); %feasible point
% X = eye(n,n);S =4*eye(n,n); L = 3*eye(n,n); lambda = zeros(n);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

history.equ(1) = norm(X -S + L);
for k = 2:MAX_ITER
    X_old= X;
    S_old= S;
    L_old= L;

    % X-update
    [Q,P] = eig( C + beta*(L - S) - lambda );
    es = diag(P);
    xi = (-es + sqrt(es.^2 + 4*beta))./(2*beta);
    X = Q*diag(xi)*Q';X = (X+X')/2;
    
    % lambda_mid-update
    lambda_mid = lambda - alpah*beta*(X - S + L);
    
    % S-update with relaxation  
    S_tem = S-1/(tau*r_S)*(lambda_mid);
    v_tem = v/(tau*r_S);
    S = soft_shrink(S_tem, v_tem);S = (S+S')/2;

    %L-update
    
     L_tem = L + 1/(tau*r_L)*(lambda_mid - nu*EY);
    [W,T] = eig(L_tem);
    eigL = max(diag(T),0);
    L = W*diag(eigL)*W';L = (L+L')/2;
    
    % lambda_k+1-update
    LL_OLD = L-L_old;
    SS_OLD = S-S_old;
    lambda = lambda_mid  - beta* (LL_OLD-SS_OLD);
    lambda = (lambda+lambda')/2;
    equ = X -S + L;
    %compute the norm of equality constraints
   
    history.equ(k) = norm(equ);
    %%%%% 
    history.obj(k)  = objective(C, X, xi, S, eigL, v, nu);
    
    S_error=norm(SS_OLD,'fro')/(1+norm(S_old,'fro'));
    X_error=norm(X- X_old,'fro')/(1+norm(X_old,'fro'));
    L_error=norm(LL_OLD,'fro')/(1+norm(L_old,'fro'));
    history.error(k)  = max(max(S_error,X_error),L_error);
    
%     if mod(k,200)==1
%         fprintf('%3d\t %0.4e\t%0.4e\n', k,...,
%         history.equ(k), history.error(k));
%     end
     if para.continuation && mod(k,para.num_continuation)==0; 
         beta = min(beta*para.eta,para.muf) ;
         r_L = 1.001*beta;
         r_S = 1.001*beta;
     end; 

    if (history.equ(k) <= tol1 && history.error(k)<= TOL)
        history.iter = k-1;
        history.objval=history.obj(k);
         break;
    end
end
% if PRINT_FINAL
%    fprintf('%3d\t %4s\t %4s %e\n', k-1, ...
%             history.equ(k), history.error(k), history.obj(k));
%    toc(t_start);
% end
%fprintf('Fk: %4.16f\n',history.objval(k));
end


function obj = objective(C, X, xi, S, eigL, v, nu)
    %obj = trace(C'*X) - log(det(X)) + v*norm(S(:), 1) + nu*trace(L);
    obj = sum(sum(C.*X)) - sum(log(xi)) + v*sum(abs(S(:))) + nu*sum(eigL);
end

function x = soft_shrink(z,tau)
x = sign(z).*max(abs(z)-tau,0);
end


