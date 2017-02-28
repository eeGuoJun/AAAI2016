function [Omega] = DADL(Y,W,H,lamda1,lamda2,lamda3,sigma,T) 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Descripion: 
% This function implements DADL in Algorithm 1 of our paper
% Jun Guo, Yanqing Guo, Xiangwei Kong, Man Zhang, and Ran He, 
% "Discriminative Analysis Dictionary Learning," In AAAI 2016.
%   -Source code version 1.0  2016/03/01 by Jun Guo
%   -The complete code accompanying a journal version 
%    of this paper will be released after tidying up.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note: 
% Users need to generate the matrixes Y, W and H in advance.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Please cite our work if you find the code helpful.
% @inproceedings{JunGuo_AAAI_2016,
%   author = {J. Guo and Y. Guo and X. Kong and M. Zhang and R. He},
%   title = {Discriminative Analysis Dictionary Learning},
%   booktitle = {Proc. AAAI Conf. Artificial Intell. (AAAI)},
%   address = {Phoenix, Arizona},
%   pages = {1617-1623},
%   month = {Feb.},
%   year = {2016}
% }
% If you have problems, please contact us at 
% eeguojun@outlook.com  or  guoyq@dlut.edu.cn
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


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
        X = hard_thr(X_head,T); % Sometimes, ignoring the sparsity can achieve better performance  
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
% getL aims to obtain the Laplacian matrix in each iteration 
d = sum(W.*R,2);
if min(diag(d))>0  % Normalized Laplacian Matrix
    L = diag(1./sqrt(d))*(diag(d)-W)*diag(1./sqrt(d)); 
else  % Unnormalized Laplacian Matrix
    L = diag(d)-W; 
end
L = full(max(L,L')); % guarantee symmetry

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function X = hard_thr(Y,T)
% hard_thr performs hard thresholding operation on each column of Y
X = Y;
Yabs = abs(Y);
[matYabs,~] = sort(Yabs,1,'descend');
med = matYabs(T+1,:);
Yabs = Yabs - repmat(med,size(Y,1),1);
X(Yabs<=0) = 0;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function D = EuDist2(fea_a,fea_b,bSqrt)
% EuDist2 efficiently computes the Euclidean distance matrix
%   D = EuDist(fea_a,fea_b)
%   fea_a:    nSample_a * nFeature
%   fea_b:    nSample_b * nFeature
%   D:      nSample_a * nSample_a
%       or  nSample_a * nSample_b
%   Written by Deng Cai (dengcai@gmail.com)
if ~exist('bSqrt','var')
    bSqrt = 1; 
	% bSqrt=1: Euclidean distance; bSqrt=0: Euclidean distance^2
end
if (~exist('fea_b','var')) || isempty(fea_b)
    aa = sum(fea_a.*fea_a,2);
    ab = fea_a*fea_a';    
    if issparse(aa)
        aa = full(aa);
    end    
    D = bsxfun(@plus,aa,aa') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
    D = max(D,D');
else
    aa = sum(fea_a.*fea_a,2);
    bb = sum(fea_b.*fea_b,2);
    ab = fea_a*fea_b';
    if issparse(aa)
        aa = full(aa);
        bb = full(bb);
    end
    D = bsxfun(@plus,aa,bb') - 2*ab;
    D(D<0) = 0;
    if bSqrt
        D = sqrt(D);
    end
end
