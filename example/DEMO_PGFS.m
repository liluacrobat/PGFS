function DEMO_PGFS
%% ========================================================================
% A demo for PGFS that performs feature selection and structure learning at 
% the same time over a prostate cancer data set.
%
%--------------------------------------------------------------------------
% Author: Lu Li
% update history: 08/10/2020
%% ========================================================================

close all;clc;clear;
addpath(genpath('../../PGFS'));

%% Load data
%--------------------------------------------------------------------------
% training: DxN data matrix of N samples in D dimensions
% Label, Label_legend: sample labels and the corresponding legends
%       1 - NP, normal prostate
%       2 - NA, normal tissues adjacent to tumors
%       3 - PT, primary tumors
%       4 - MT, metastatic tumors
%--------------------------------------------------------------------------
load('./demo_data.mat');

%% Preprocessing
% Select the top 1,000 featuers with the highest standard deviation to
% facilitate the compuation.
% Weight: feature weight
% PPoint: points on the principal graph (principal points)
% B: edge indicator
%--------------------------------------------------------------------------

training_sd = std(training,[],2);
[~,index]=sort(training_sd,'descend');
X = training(index(1:1000),:);
Y = Label;

%% Perform feature selecyion using PGFS
Para.it = 50;               % Maximum iteration
Para.M = size(X,2);        % Number of points on the principal graph
Para.sigma = 8;          % Kernel width
Para.beta = 21;             % Tree length regularization parameter
Para.lambda = 16;             % Sparness regularization parameter  

[Weight, PPoint, B, ~, ~, ~] = src_PGFS(X, Para);

%% Select features with 0.01 cutoff and visualize results

X_selected = X(Weight>0.01,:);
PPoint_selected = PPoint(Weight>0.01,:);

[PPoint_extend,B_extend,~,~] = src_postprocessing(PPoint_selected,...
    X_selected, Label_legend(Y),'NP');
src_plotPGraph(X_selected,Y,PPoint_extend,B_extend);
view(151,62)
legend({'NP','NA','PT','MT'})
set(gcf,'Position',[84   285   854   520]);
set(gca,'fontweight','normal');
box on
set(gca,'FontSize',18);
end


