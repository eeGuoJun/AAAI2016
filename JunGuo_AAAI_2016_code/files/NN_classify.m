function [predict,acc] = NN_classify(Omega,TrData,TtData,T,H_tr,H_tt)
% 'NN_classify.m' implements classification by 1-NN classifier
% Input:
%       Omega	-analysis dictionary
%       TrData  -each column is a training sample
%       TtData  -each column is a testing sample
%       T       -sparsity of x_i, ||X||_0 <= T
%       H_tr    -one-hot binary matrix (size: nClass * nTrainingSample)
%       H_tt    -one-hot binary matrix (size: nClass * nTestingSample)
% Output:
%       predict -a predicted label vector of testing samples
%       acc     -classification accuracy


%% coding
% Sometimes, ignoring the sparsity can achieve better performance 
% TestData = Omega*TtData; 
% TrnData = Omega*TrData;
TestData = hard_thr(Omega*TtData,T); 
TrnData = hard_thr(Omega*TrData,T); 
clear TtData TrData Omega % clear useless variable


%% classify
TrnData = TrnData'; 
TrnLabel = labelConvert(H_tr);
clear H_tr % clear useless variable
TestData = TestData'; 
TestLabel = labelConvert(H_tt);
clear H_tt % clear useless variable

% here we use 1-NN classifer (users can try other classifiers)
predict = knnclassify(TestData,TrnData,TrnLabel,1,'euclidean','nearest');
matchVec = find(predict==TestLabel);
acc = length(matchVec)/length(TestLabel);