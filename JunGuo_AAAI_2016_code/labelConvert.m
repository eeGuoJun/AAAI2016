function Lout = labelConvert(Lin)
% one-hot binary matrix ==> label-column vector
% or  label-column vector ==> one-hot binary matrix 
% Input:
%       Lin   -a one-hot binary matrix (size: nClass * nSmp)
%             or a label-column vector (size: nSmp * 1)
% Output:
%       Lout  -a label-column vector (size: nSmp * 1)
%             or a one-hot binary matrix (size: nClass * nSmp)


if min(size(Lin))~=1  % one-hot binary matrix ==> label-column vector
    [~, Lout] = max(Lin);
    Lout = Lout(:);
else  % label-column vector ==> one-hot binary matrix 
    Lin = Lin(:);
    nSmp = length(Lin);
    nClass = length(unique(Lin));
    Lout = full(sparse(Lin,[1:nSmp]',ones(nSmp,1),nClass,nSmp));
end  