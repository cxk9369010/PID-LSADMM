function [history,k] = PJALM(D, beta, TOL,toll,para)

% covsel  Sparse inverse covariance selection via GSADMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Algorithm:  PJALM\n');

%% Global constants and defaults
QUIET    = 1;
PRINT_FINAL = 1 ;
MAX_ITER = 1000;

%% Data preprocessing
C = cov(D);
n = size(C,1);
sigmma=2;  %>=3-1=2
%% two parameters
v=0.005; nu=0.05;


%% ADMM inital values
EY= eye(n,n); %identity matrix

X = para.X; S = para.S; L = para.L; lambda = para.lambda;


if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\n', 'iter',  'equ norm', 'error''ROB');
end
history.objval(1)  = objective(C, X, S, L, v, nu);
% Fk= 32.0337449377591170;
% 
% history.ROB(1)=abs((history.objval(1)-Fk)/Fk);
history.equ(1) = norm(X -S + L);
t_start = tic;
for k = 2:MAX_ITER
    X_old=X;
    S_old=S;
    L_old=L;

    % X-update
    [Q,P] = eig( C + beta*(L - S) - lambda- sigmma*beta*X );
    es = diag(P);
    xi = (-es + sqrt(es.^2 + 4*(sigmma+1)*beta))./(2*(sigmma+1)*beta);
    X = Q*diag(xi)*Q';

    % S-update with relaxation
    S_tem = ( X_old + L+sigmma*S-lambda/beta )/(sigmma+1);
    v_tem = v/((sigmma+1)*beta);
    S = shrinkage(S_tem, v_tem);
    
    %L-update
    L_tem = (L+S_old+lambda/beta-X_old-nu*EY/beta)/(sigmma+1);
    [W,T] = eig(L_tem);
    L = W*diag(max(diag(T),0))*W';
    
    % lambda_k+1-update
    lambda = lambda - beta*(X-S+L);
    
    %compute the norm of equality constraints
    equ = X -S + L;
    history.equ(k) = norm(equ);
    %%%%% 
    history.objval(k)  = objective(C, X, S, L, v, nu);
%     history.ROB(k)=abs((history.objval(k)-Fk)/Fk);
    
%     S_error=norm(S-S_old,'inf');
%     X_error=norm(X-X_old,'inf');
%     L_error=norm(L-L_old,'inf');
   S_error=norm(S -S_old,'fro')/(1+norm(S_old,'fro'));
    X_error=norm(X- X_old,'fro')/(1+norm(X_old,'fro'));
    L_error=norm(L- L_old,'fro')/(1+norm(L_old,'fro'));
    history.error(k)  = max(max(S_error,X_error),L_error);

    if ~QUIET
        fprintf('%3d\t%10.4f\t%10.4f\n', k,...,
        history.equ(k), history.error(k));
    end

%     if (history.equ(k) <= 1e-4 && history.ROB(k)<=toll && history.error(k)<=TOL)
 if (history.equ(k) <= toll && history.error(k)<= TOL)
         break;
    end
end
if PRINT_FINAL
   fprintf('%3d\t%4s\t%4s\n', k-1, ...
            history.equ(k), history.error(k));
   toc(t_start);
end
%fprintf('Fk: %4.16f\n',history.objval(k));
end

function obj = objective(C, X, S, L, v, nu)
    obj = trace(C*X) - log(det(X)) + v*norm(S(:), 1) + nu*trace(L);
end

function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end




