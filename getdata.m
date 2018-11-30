function [SigmaO] = getdata(dataformat,input_dim)

randn('seed', 2018);
rand('seed', 2018);

if strcmp(dataformat,'rand-1')
    %     alpha = 0.025; beta = 0.021;
    p = input_dim+round(input_dim*0.05); po = input_dim; ph = p-po;
    W = zeros(p,p); spa = 0.025; picks = randperm(p*p); picks = picks(1:round(p*p*spa));
    W(picks) = randn(size(picks));
    C = W' * W;
    C(1:po,po+1:p) = C(1:po,po+1:p) + 0.5*randn(po,ph); C = (C+C')/2;
    d = diag(C); C = max(min(C-diag(d),1),-1);
    K = C + max(-1.2*min(eig(C)),0.001)*eye(p);
    KO = K(1:po,1:po); KOH = K(1:po,po+1:p); KHO = K(po+1:p,1:po); KH = K(po+1:p,po+1:p);
    KOtilde = KO - KOH/KH*KHO;
    N = 5*po;
    EmpCov = inv(KOtilde); EmpCov = (EmpCov + EmpCov')/2;
    data = mvnrnd(zeros(N,po),EmpCov);
    SigmaO = (1/N)*data'*data;
elseif strcmp(dataformat,'rand-K')
    p = input_dim+round(input_dim*0.05); po = input_dim; ph = p-po;
    nnzr = 0.01*p*p;          % Number of nonzero coefficients in A
    % Generate A, the original inverse covariance, with random sparsity pattern...
    A=eye(p);
    for i = 1:nnzr
        A(ceil(p*rand),ceil(p*rand))=sign(rand()-.5);
    end
    K=A*A'+1e-6*eye(p); % A is the gound truth inverse covariance matrix
    KO = K(1:po,1:po); KOH = K(1:po,po+1:p); KHO = K(po+1:p,1:po); KH = K(po+1:p,po+1:p);
    KOtilde = KO - KOH/KH*KHO;
    N = 5*po; 
    B = inv(KOtilde); % B is the ground-truth covariance matrix
    B=(B+B')/2; % B = B + 1e-6*eye(p); B = (B+B')/2;
    data = mvnrnd(zeros(N,po),B);
    SigmaO = (1/N)*data'*data;
elseif strcmp(dataformat,'Rosetta')
    load Rosetta.mat; % rho = 0.0313; alpha = rho; beta = 5*rho;
    Tmp = isnan(Gene);
    Gene1 = Gene;
    for i = 1:size(Gene,1)
        for j = 1:size(Gene,2)
            if (Tmp(i,j)==1); Gene1(i,j)=0; end;
        end
    end
    n = input_dim;
    vv = var(Gene1');
    [vvsort,idx] = sort(vv);
    vvsort = vvsort(end:-1:1); idx = idx(end:-1:1);
    %% select n variables with highest variances
    index = idx(1:n);
    Sigma = cov(Gene1(index,:)');
    Sigma = 0.5 * (Sigma + Sigma'); SigmaO = Sigma;
    
elseif strcmp(dataformat,'Iconix');
    %load GSE8858.mat;
    load GSE8858_GPL5424_series_matrix_1.mat
    Gene = data; % rho  = 0.0853; alpha = rho; beta = 5*rho;
    Tmp = isnan(Gene);
    Gene1 = Gene;
    for i = 1:size(Gene,1)
        for j = 1:size(Gene,2)
            if (Tmp(i,j)==1); Gene1(i,j)=0; end;
        end
    end
    n = input_dim;
    vv = var(Gene1');
    [vvsort,idx] = sort(vv);
    vvsort = vvsort(end:-1:1); idx = idx(end:-1:1);
    %% select n variables with highest variances
    index = idx(1:n);
    Sigma = cov(Gene1(index,:)');
    Sigma = 0.5 * (Sigma + Sigma'); SigmaO = Sigma;
end
end

