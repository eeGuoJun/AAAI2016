function W = calculateW_corr(data,k,H,sigma) % may be memory-consuming
%% Constructing a supervised graph based on triplets
% Input:
%      data     -each column is a data point
%      k        -number of nearest neighbors
%      H    	-a one-hot binary matrix (size: nLabel * nSmp)
%      sigma    -Gaussian kernel: exp[-( ||xi-xj||_2^2 / sigma^2 )]
% Output:
%      W        -weighting matrix


%% Initialization
[nLabel, nSmp] = size(H);
if k > 0  && k < nSmp
    Ww = -ones(nSmp,nSmp,'int8'); 
    for idx = 1:1:nLabel
        classIdx = find(H(idx,:)==1);
        Ww(classIdx,classIdx) = 1;  
        % Ww(i,j) =1 if i and j are from the same class;  =-1, otherwise
    end
else
    error('Parameter k is error!');
end    


%% distance and weighting
Dist = EuDist2(data',[],0); % Euclidean distance ^2
Dist = 1-exp(-Dist/(sigma.^2)); % CIM ^2
[~, idx] = sort(Dist,2); % sort each row ascend
idx = idx(:,2:k+1); % default: not self-connected
G = sparse(repmat([1:nSmp]',[k,1]),idx(:),ones(numel(idx),1),nSmp,nSmp);
G = int8(full(G)); % G is a square matrix
% i^th row of G: Among the other (nSmp-1) samples, which belongs to 
% the i^th sample's k nearest neighbors  =1: belong; =0: not belong
W = single(Dist);
clear Dist idx % clear useless variable
Ww = sign(bsxfun(@plus,-repmat(Ww,nSmp,1),Ww(:))); 
% catergory. u=1:nSmp, for the u^th block: the (i,v)^th value 
% =0 means u&v&i are from the same class or u&v&i are from three classes
% =+1 means u&i are from the same class or v&i are from different classes
% =-1 means v&i are from the same class or u&i are from different classes
Ws = sign(bsxfun(@plus,-repmat(W,nSmp,1),W(:))); 
% The difference between distances. u=1:nSmp, for the u^th block:
% the (i,v)^th value =+1 means Siu>Siv while =-1 means Siu<Siv.
Ws = -single(Ww).*Ws; % which value {-1\0\+1} for each position to multiply 
Ws(Ww==0) = 1;
Ws = (Ws.*bsxfun(@minus,repmat(W,nSmp,1),W(:))).*single(repmat(G,nSmp,1));
clear Ww  % clear useless variable


%% obtain weighting matrix
W = reshape(sum(Ws,2),nSmp,nSmp).*single(G);
Ws = W;
Ws(W==0) = 1e15;
%%%%% the following two lines aim to scale non-zero weights to [0,1] %%%%
W = bsxfun(@rdivide,bsxfun(@minus,W,min(Ws,[],2)),max(W,[],2)-min(Ws,[],2));
W = W.*single(G);  % the scaling is row-wise
%%%%% For the above two lines, users can comment out by real demand %%%%%
clear G Ws % clear useless variable
W = double(full(0.5*(W+W'))); % calculate the symmetric part