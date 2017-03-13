function X = hard_thr(Y,T)
% 'hard_thr.m' performs thresholding on each column of Y, solving
%   min_X  ||X - Y||_F^2
%    s.t.  ||x_i||_0 <= T for all i (all columns of X)
% Input:
%   Y   -a matrix
%   T	-sparsity level
% Output:
%	X   -a matrix, where each column has only T non-zero positions

X = Y;
Yabs = abs(Y);
[matYabs,~] = sort(Yabs,1,'descend');
med = matYabs(T+1,:);
Yabs = Yabs - repmat(med,size(Y,1),1);
X(Yabs<=0) = 0;