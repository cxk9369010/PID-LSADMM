function [S, history,k] = TADMM(D, beta, TOL,toll)

% covsel  Sparse inverse covariance selection via TADMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
fprintf('Algorithm:  TADMM_II\n');

%% Global constants and defaults
QUIET    = 1;
PRINT_FINAL = 1 ;
MAX_ITER = 1000;

%% Data preprocessing
C = cov(D);
n = size(C,1);
%% two parameters in the problem
v=0.005; nu=0.05;

%% 
%%%----- proximal paratemer in TADMM by their experiments( M=bar_v* EY )
bar_v=beta;

%% inital iterative values
EY= eye(n,n); %identity matrix

%X = eye(n,n);S =2*eye(n,n); L = eye(n,n); lambda = zeros(n); %table 3
X = zeros(n);S =zeros(n); L = zeros(n); lambda = zeros(n);
% X = eye(n,n);S = eye(n,n);  L =zeros(n);  lambda = zeros(n); %feasible point
% X = eye(n,n);S =4*eye(n,n); L = 3*eye(n,n); lambda = zeros(n);

%% corection facor
bar_a=1.6;

if ~QUIET
    fprintf('%3s\t%10s\t%10s\t%10s\n', 'iter',  'equ norm', 'error''ROB');
end
history.objval(1)  = objective(C, X, S, L, v, nu);
% Fk= 32.0271931399763550;
% 
% history.ROB(1)=abs((history.objval(1)-Fk)/Fk);
history.equ(1) = norm(X -S + L);
t_start = tic;
for k = 2:MAX_ITER
    X_old=X;
    S_old=S;
    L_old=L;
    lambda_old=lambda;

    % X-update
    [Q,P] = eig( C + beta*(L - S) - lambda_old );
    es = diag(P);
    xi = (-es + sqrt(es.^2 + 4*beta))./(2*beta);
    X = Q*diag(xi)*Q';
    
    % lambda_mid-update
    lambda_mid= lambda_old- beta*( X- S_old+ L_old );
    diff_lambda= lambda_mid- lambda_old;
    
    % S-update with relaxation
    S_tem = ( beta*(X + L_old) -lambda_mid+ bar_v*S_old )/(beta+ bar_v);
    v_tem = v/(beta+ bar_v);
    S = shrinkage(S_tem, v_tem);
    diff_S= S- S_old;
       
    %L-update
    L_tem = ( beta*(S_old- X)+ lambda_mid+ bar_v*L_old- nu*EY )/(beta+ bar_v);
    [W,T] = eig(L_tem);
    L = W*diag(max(diag(T),0))*W';
    diff_L= L- L_old;
    
    % correction step
    S= S_old+  bar_a*diff_S;
    L= L_old+  bar_a*diff_L;
    lambda= lambda_old+ bar_a*diff_lambda;
    
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
    obj = trace(C'*X) - log(det(X)) + v*norm(S(:), 1) + nu*trace(L);
end

function y = shrinkage(a, kappa)
    y = max(0, a-kappa) - max(0, -a-kappa);
end


