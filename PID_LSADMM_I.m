function [S, history,k] = PID_LSADMM_I(D, beta, TOL,toll,para)

% covsel  Sparse inverse covariance selection via GSADMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Algorithm:  PID_LSADMM_I\n');

%% Global constants and defaults
QUIET    = 1;
PRINT_FINAL = 1 ;
MAX_ITER = 1000;

%% Data preprocessing
C = cov(D);
n = size(C,1);
rho_1= 1.001;  %>p-1=1

bar_rho_1= rho_1+1;

%%%%%%%%%%%%%%%%%%%%%%%%
%%
 
%alpah=1.6;  s=0; %%% better set alpha+s=1.6 and tau = 0.8

alpha = para.alpha;  
tau   = para.tau;

r_L = 1.001*beta;
 
%% two parameters
v=0.005; nu=0.05;


%% ADMM inital values
EY= eye(n,n); %identity matrix

X = eye(n,n);S =2*eye(n,n); L = eye(n,n); lambda = zeros(n); %table 3 
% X = eye(n,n);S = eye(n,n);  L =zeros(n);  lambda = zeros(n); %feasible point
% X = eye(n,n);S =4*eye(n,n); L = 3*eye(n,n); lambda = zeros(n);

%%
if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\n', 'iter',  'equ norm', 'error''ROB');
end
history.objval(1)  = objective(C, X, S, L, v, nu);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

history.equ(1) = norm(X -S + L);
t_start = tic;
for k = 2:MAX_ITER
    X_old= X;
    S_old= S;
    L_old= L;

    % X-update
    [Q,P] = eig( C + beta*(L - S) - lambda- rho_1*beta*X );
    es = diag(P);
    xi = (-es + sqrt(es.^2 + 4* bar_rho_1 *beta))./(2* bar_rho_1 *beta);
    X = Q*diag(xi)*Q';

    % S-update with relaxation
    S_tem = ( X_old + L+ rho_1*S- lambda/beta )/bar_rho_1;
    v_tem = v/(bar_rho_1 *beta);
    S = shrinkage(S_tem, v_tem);
    
    % lambda_mid-update
    lambda_mid = lambda - alpha*beta*(X - S +L);
    
    %L-update
   % L_tem = (sigmma_2*L+ S+ lambda_mid/beta- X- nu*EY/beta)/bar_sigmma_2;
    
     L_tem = L + 1/(tau*r_L)*(lambda_mid)- nu*EY/beta;
   
    [W,T] = eig(L_tem);
    L = W*diag(max(diag(T),0))*W';
    
    %equ = X -S + L_old;
    % lambda_k+1-update
    lambda = lambda_mid  - beta* (L-L_old);
    equ = X -S + L;
    %compute the norm of equality constraints
   
    history.equ(k) = norm(equ);
    %%%%% 
    history.objval(k)  = objective(C, X, S, L, v, nu);
    
   S_error=norm(S -S_old,'fro')/(1+norm(S_old,'fro'));
    X_error=norm(X- X_old,'fro')/(1+norm(X_old,'fro'));
    L_error=norm(L- L_old,'fro')/(1+norm(L_old,'fro'));
    history.error(k)  = max(max(S_error,X_error),L_error);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\n', k,...,
        history.equ(k), history.error(k));
    end
    
    if mod(k,200)==1
        fprintf('%3d\t %0.4e\t%0.4e\n', k,...,
        history.equ(k), history.error(k));
    end

    %if (history.equ(k) <= 1e-4 && history.ROB(k)<= toll && history.error(k)<= TOL)
    if (history.equ(k) <= toll && history.error(k)<= TOL)
         break;
    end
end
if PRINT_FINAL
   fprintf('%3d\t%4s\t%4s\n', k-1, ...
            history.equ(k), history.error(k));
   toc(t_start);
end
%%fprintf('Fk: %4.16f\n',history.objval(k));
end


function obj = objective(C, X, S, L, v, nu)
    obj = trace(C'*X) - log(det(X)) + v*norm(S(:), 1) + nu*trace(L);
end

function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end


