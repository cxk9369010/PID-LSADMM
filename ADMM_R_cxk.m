function out = ADMM_R_cxk(SigmaO,alpha,beta,opts) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This file implements the PGADM algorithm described in 
% "Alternating Direction Methods for Latent Variable Gaussian Graphical 
% Model Selection", appeared in Neural Computation, 2013,
% by Ma, Xue and Zou, for solving 
% Latent Variable Gaussian Graphical Model Selection  
% min <R,SigmaO> - logdet(R) + alpha ||S||_1 + beta Tr(L) 
% s.t. R = S - L,  R positive definte,  L positive semidefinite 
% 
% Authors: Shiqian Ma, Lingzhou Xue and Hui Zou 
% Copyright (C) 2013 Shiqian Ma, The Chinese University of Hong Kong  
% Date: Jan 25, 2013

% rewritten by chang xiaokai 
% Date: Nov 25, 2018
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% initialization 
QUIET    = 1;
ABSTOL   = 1e-5;
RELTOL   = 1e-5;
TOL = opts.TOL;
toll =opts.tol1;
n = size(SigmaO,1); 
R = eye(n,n); S = R; L = zeros(n,n); Lambda = zeros(n,n); 
mu = opts.mu; eta = opts.eta; tau = opts.tau; 

for iter = 1: opts.maxiter
    % update R
    R_old = R;
    B = mu*SigmaO - mu*Lambda - S + L; 
   % [U,D] = mexeig(B);
   [U,D] = eig(B);
    d = diag(D); 
    eigR = (-d + sqrt(d.^2+4*mu))/2; 
    R = U*diag(eigR)*U'; R = (R+R')/2; 
    
    S_old = S; 
    L_old = L;
    
    % update S and L 
    Gradpartial = S - L - R + mu*Lambda; 
    G = S - tau * Gradpartial; H = L + tau * Gradpartial; 
    S = soft_shrink(G, tau*mu*alpha); S = (S+S')/2; 
%     [U,D] = mexeig(H); 
    [U,D] = eig(H); 
    d = diag(D)-tau*mu*beta; eigL = max(d,0); 
    L = U*diag(eigL)*U'; L = (L+L')/2; 
    
    % update Lambda 
    Lambda = Lambda - (R-S+L)/mu; Lambda = (Lambda + Lambda')/2; 
    % diagnostics, reporting, termination checks
    k = iter; 
    history.objval(k)  = objective(R,SigmaO,eigR,S,eigL,alpha,beta); 

    history.r_norm(k)  = norm(R-S+L,'fro');
    history.s_norm(k)  = norm(-([S;L] - [S_old;L_old])/mu);

    history.eps_pri(k) = sqrt(3*n*n)*ABSTOL + RELTOL*max(norm(R,'fro'), norm([S;-L],'fro'));
    history.eps_dual(k)= sqrt(3*n*n)*ABSTOL + RELTOL*norm(Lambda,'fro');

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\t%10.4f\t%10.4f\t%10.2f\n', k, ...
            history.r_norm(k), history.eps_pri(k), ...
            history.s_norm(k), history.eps_dual(k), history.objval(k));
    end
 
    % print stats
    resid = norm(R-S+L,'fro')/max([1,norm(R,'fro'),norm(S,'fro'),norm(L,'fro')]); 

    %%%%%  
    S_error=norm(S -S_old,'fro')/(1+norm(S_old,'fro'));
    R_error=norm(R- R_old,'fro')/(1+norm(R_old,'fro'));
    L_error=norm(L- L_old,'fro')/(1+norm(L_old,'fro'));
    history.error(k)  = max(max(S_error,R_error),L_error);
    
%     if mod(k,200)==1
%         fprintf('%3d\t %0.4e\t%0.4e\n', k,...,
%         history.r_norm(k), history.error(k));
%     end
    
    obj = history.objval(k);  
    % check stop 
    %if resid < opts.stoptol 
    if (history.r_norm(k) <= toll && history.error(k)<= TOL)
        out.R = R; out.S = S; out.L = L; out.obj = obj; out.eigR = eigR; out.eigL = eigL; out.resid = resid; out.iter = iter; 
        out.equ = history.r_norm(k);out.error = history.error(k);return; 
    end
    if opts.continuation && mod(iter,opts.num_continuation)==0; mu = max(mu*eta,opts.muf) ;
    end;  
end
out.R = R; out.S = S; out.L = L; out.obj = obj; out.eigR = eigR; out.eigL = eigL; out.resid = resid; out.iter = iter;



function x = soft_shrink(z,tau)
x = sign(z).*max(abs(z)-tau,0); 

function obj = objective(R,SigmaO,eigR,S,eigL,alpha,beta)
obj = sum(sum(R.*SigmaO)) - sum(log(eigR)) + alpha*sum(abs(S(:))) + beta*sum(eigL);
