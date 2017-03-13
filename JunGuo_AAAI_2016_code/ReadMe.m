%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Descripion: 
% This source code is for our paper
% Jun Guo, Yanqing Guo, Xiangwei Kong, Man Zhang, and Ran He, 
% "Discriminative Analysis Dictionary Learning," In AAAI 2016.
%   -Source code version 1.0  2016/03/01 by Jun Guo
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Note 1: 
% The example feature dataset (AR) used in this code is from 
% Dr. Jiang: http://www.umiacs.umd.edu/~zhuolin/projectlcksvd.html.
% We have already split and normalized original data for AR dataset. 
% For other datasets, we recommend to pre-process data via 'normcols.m'.
% Note 2:
% For experiments on Extended Yale B and Caltech 101 datasets, 
% we also used the features provided by Dr. Jiang.
% For experiment on UCF 50, we used the Action bank features: 
% http://www.cse.buffalo.edu/~jcorso/r/actionbank/. 
% Please refer to our paper for detailed experimental settings.
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
% If you have problems, please contact us at eeguojun@outlook.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%