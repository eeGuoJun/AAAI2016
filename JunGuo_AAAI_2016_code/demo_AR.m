close all;
clear all;
clc;


%% settings
addpath(genpath('.\files'));
load('AR.mat');
% we recommend to pre-process data via 'normcols.m' for other datasets.
training_feats = normcols(training_feats);	
testing_feats = normcols(testing_feats);
lamda1 = 10;
lamda2 = 1e-3;
lamda3 = 1e-1;
kNN = 7;
sigma = 10;


%% initialization
% W = calculateW_corr(training_feats,kNN,H_train,sigma); % memory-consuming
% W = computeW_corr(training_feats,kNN,H_train,sigma); % time-consuming
load('AR_W.mat'); % we have already computed W via 'computeW_corr.m'
[H,T] = generateH_hybrid(H_train,size(training_feats,1));
H = normcols(H);


%% training
fprintf('\nTraining...');
[Omega] = DADL(training_feats,W,H,lamda1,lamda2,lamda3,sigma,T);
fprintf('Done!');


%% testing
fprintf('\nTesting...');
[~,acc] = NN_classify(Omega,training_feats,testing_feats,T,H_train,H_test);
fprintf('Done!\n');


%% show ACC
fprintf('Classification accuracy is %.01f%%. \n',acc*100);