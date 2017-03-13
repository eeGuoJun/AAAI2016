function [Omega] = DADL(Y,W,H,lamda1,lamda2,lamda3,sigma,T) 
% 'DADL.m' implements DADL in Algorithm 1
% Input:
%       Y       -each column is a training sample
%       W       -the weighting matrix (local topology)
%       H       -each column is a target code (reflects y's catergory)
%       lamda1	-regularization parameter for code consistent term
%       lamda2	-regularization parameter for local topology term
%       lamda3	-regularization parameter for ||Omega||_F^2
%       sigma   -a parameter of Gaussian kernel
%       T       -sparsity of x_i, ||X||_0 <= T
% Output:
%       Omega	-analysis dictionary


X = H;  % let Xinit = H
p = ones(1,size(Y,2));
P = diag(p);
q = ones(1,size(Y,2));
% Q = diag(q);
R = ones(size(Y,2));
iterOut = 6;	% Outer Loop
iterIn = 3;     % Inner Loop
for iOut = 1:1:iterOut
    L = getL(W,R); % update Laplacian Matrix
    % First : update Omega and X
    temp = lamda1*(q./p);
    for iIn = 1:1:iterIn % inner iteration can run just one round 
        Omega = (X*P*Y')/(Y*(P+lamda2*L)*Y'+lamda3*eye(size(Y,1))); 
        X_head = (Omega*Y+H.*repmat(temp,size(H,1),1))./repmat(1+temp,size(H,1),1);
        X = hard_thr(X_head,T); 
        % Sometimes, ignoring the sparsity can achieve better performance  
    end
    % Next : update P, Q, R    
    p = sum((X-Omega*Y).^2);
    p = exp(-p/(sigma^2));
    P = diag(p);  % update P
    q = sum((X-H).^2);
    q = exp(-q/(sigma^2));
%     Q = diag(q);  % update Q
    r = EuDist2(Y'*Omega',[],0);
    R = exp(-r/(sigma.^2));  % update R
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function L = getL(W,R) 
% 'getL' aims to obtain the Laplacian matrix in each iteration 
d = sum(W.*R,2);
if min(diag(d))>0  % Normalized Laplacian Matrix
    L = diag(1./sqrt(d))*(diag(d)-W)*diag(1./sqrt(d)); 
else  % Unnormalized Laplacian Matrix
    L = diag(d)-W; 
end
L = full(max(L,L')); % guarantee symmetry