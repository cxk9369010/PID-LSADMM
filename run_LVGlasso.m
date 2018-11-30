clc; 
clear all; 
close all;
results=[];
%% load data
problem = 1;

if problem ==1
   rho = 0.0313; alpha = rho; beta = 5*rho;
   dataformat = ' Rosetta'; %%  Rosetta ;Iconix; rand-1;rand-K 
   input_dim = 100; 
   SigmaO = getdata(dataformat,input_dim); 
   opts.TOL=1e-5;
   opts.tol1=1e-6;
else
    randn('seed', 0);
    rand('seed', 0);
   alpha = 0.005; 
   beta = 0.05;
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
   SigmaO = cov(D);
   opts.TOL=1e-7;
   opts.tol1=1e-6;
end


n = size(SigmaO,1);


do_LSADMM = 1; do_ADMM_R_cxk = 1; 

opts.maxiter = 500; 

if do_LSADMM
    opts.v=alpha;
    opts.nu=beta;
    q=2;
    if problem == 1
       sigma=1/n;
       opts.alpha = 1;
       opts.eta = 4;
    elseif problem ==2
       sigma=0.05;
       opts.alpha = 1.99;
       opts.eta = 1;
    end
        
    opts.tau = 1.001*q*((2 + opts.alpha)/4);
    opts.continuation = 1;  opts.num_continuation = 10;  opts.muf = 1e+6;
    tic; out_LSADMM = LSADMM(SigmaO, sigma, opts);solve_LSADMM = toc;
    fprintf('LSADMM: obj: %e, iter: %d, cpu: %3.1f \n',out_LSADMM.objval,out_LSADMM.iter,solve_LSADMM); 
end

if do_ADMM_R_cxk
    opts.tau = 0.6;  
    opts.continuation = 1; opts.mu = n; opts.num_continuation = 10; opts.eta = 1/4; opts.muf = 1e-6;
        tic; out_R = ADMM_R_cxk(SigmaO,alpha,beta,opts); solve_R = toc; 
    fprintf('ADMM_R: obj: %e, iter: %d, cpu: %3.1f, resid:%e \n',out_R.obj,out_R.iter,solve_R,out_R.resid);
end

%results=[results; out_R.iter solve_R out_R.equ out_R.error out_R.obj];
