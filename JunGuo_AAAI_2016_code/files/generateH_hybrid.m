function [H,T] = generateH_hybrid(Hinit, nFea)
% concatenating the Kron-form spectral codes
% to the sequency Walsh-ordered Hadamard codes
% Input:
%    Hinit -spectral matrix, each column has only one non-zero position
%    nFea  -feature dimension of original data Y
% Output:
%    H     -Kron-form spectral codes + sequency Walsh-ordered Hadamard codes
%    T     -sparsity level
%%%%%%%%% Notice: %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% in Analysis Dicitonary Learning (ADL) models:
% the length of sparse codes should be larger than 'nFea', 
% which is a natural request for the intrinsic ADL:
%   min_{Omega,X} ||X - Omega * Y||_F^2
%      s.t.       Omega \in a set denoted as W
%                 ||X_i||_0 <= T
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

nClass = size(Hinit,1); 

N = floor(log2((nFea+1)/3)); %the number '3' can be tuned by users
Code = walsh(2^N);
SearchRow = Code(2:nClass+1,2:end);
H1 = SearchRow'*Hinit;
H1(H1==-1) = 0; % +1/-1 --> +1/0

leng = ceil((nFea+2-2^N)/nClass);
H2 = kron(Hinit,ones(leng,1));

H = [H2; H1]; %concatenating 
T = sum(H(:,1)~=0);
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function H = walsh(N)
% H = walsh(N)
%  generate a sequency (Walsh) ordered Hadamard matrix of size N,
%  where N must be an integer power of 2.
k = log2(N);
if k-floor(k)>eps % Check that N==2^k.
  error('N must be an integer power of 2.');
end
H = hadamard(N); % Generate the Hadamard matrix
graycode = [0;1]; % generate Gray code of size N.
while size(graycode,1) < N
  graycode = [kron([0;1], ones(size(graycode,1),1)), ...
              [graycode; flipud(graycode)]];
end
% Generate indices from bit-reversed Gray code.
seqord = bin2dec(fliplr(char(graycode+'0')))+1;
% This line does the same thing, but requires the communication toolbox
% seqord = bi2de(graycode)'+1;
H = H(seqord,:); % Reorder H.
end
