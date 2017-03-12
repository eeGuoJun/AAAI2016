function W = computeW_corr(data,k,H,sigma) % may be time-consuming
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
W = [];


%% distance and weighting
Dist = EuDist2(data',[],0); % Euclidean distance ^2
Dist = 1-exp(-Dist/(sigma.^2)); % CIM ^2
[~, idx] = sort(Dist,2); % sort each row ascend
idx = idx(:,2:k+1); % default: not self-connected
G = sparse(repmat([1:nSmp]',[k,1]),idx(:),ones(numel(idx),1),nSmp,nSmp);
G = int8(full(G)); % G is a square matrix
% i^th row of G: Among the other (nSmp-1) samples, which belongs to 
% the i^th sample's k nearest neighbors  =1: belong; =0: not belong
Dist = single(Dist);
clear idx % clear useless variable
for id = 1:1:nSmp
    A_id = bsxfun(@minus,Dist(id,:)',Dist(id,:));
    Ww_id = bsxfun(@minus,Ww(id,:)',Ww(id,:));
    C_id = -sign(A_id).*single(sign(Ww_id)); 
    % which value {-1\0\+1} for each position to multiply 
    C_id(Ww_id==0) = 1;
    C_id(A_id==0) = 0;
    A_id = A_id.*C_id.*repmat(single(G(id,:)'),1,nSmp);
    W = [W; sum(A_id,1)];
end
clear Dist A_id C_id % clear useless variable


%% obtain weighting matrix
W = W.*single(G);
Ww_id = W;
Ww_id(W==0) = 1e15;
%%%%% the following two lines aim to scale non-zero weights to [0,1] %%%%
W = bsxfun(@rdivide,bsxfun(@minus,W,min(Ww_id,[],2)),max(W,[],2)-min(Ww_id,[],2));
W = W.*single(G);  % the scaling is row-wise
%%%%% For the above two lines, users can comment out by real demand %%%%%
clear G Ww_id % clear useless variable
W = double(full(0.5*(W+W'))); % calculate the symmetric part