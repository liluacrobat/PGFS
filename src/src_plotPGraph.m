function src_plotPGraph(X,Y,T,B)
%% ========================================================================
% plot the principal graph with PCA
%--------------------------------------------------------------------------
% X: data matrix whose columns are samples
% Y: class label
% T: principal points
% B: edge indicator
%--------------------------------------------------------------------------
% Author: Lu Li
% update history: 08/10/2020
%% ========================================================================

U=sort(unique(Y));
str=cell(1,length(U));
FaceColor = [
0.6510    0.8078    0.8902
    0.1216    0.4706    0.7059
    0.5961    0.3059    0.6392
    0.8902    0.1020    0.1098
    0.9290, 0.6940, 0.1250
    0.4940, 0.1840, 0.5560
    0.4660, 0.6740, 0.1880
    0.3010, 0.7450, 0.9330
    0.6350, 0.0780, 0.1840];

dim = size(X,1);
if dim<3
    X=[X;zeros(3-dim,size(X,2)) ];
end

[mapped_data,~,power]=compute_mapping([X T]','PCA',3);
mapped_data=mapped_data';
nx = size(X,2);
T = mapped_data(:,nx+1:end);
mapped_data = mapped_data(:,1:nx);

figure;
hold on
for i=1:length(U)
    plot3(mapped_data(1,Y==U(i)),mapped_data(2,Y==U(i)),mapped_data(3,Y==U(i)),...
        'o','MarkerFaceColor',FaceColor(mod(i-1,7)+1,:),'MarkerEdgeColor',FaceColor(mod(i-1,7)+1,:),'MarkerSize',8);
end
for i=1:size(B,1)
    for j=1:size(B,2)
        if B(i,j)>0
            plot3(T(1,[i j]),T(2,[i j]),T(3,[i j]),'-k','linewidth',2);
        end
    end
end
set(gca,'FontSize',14)
xlabel(['PC1 (' num2str(round(power(1)*1000)/10) '%)'],'FontSize',16);
ylabel(['PC2 (' num2str(round(power(2)*1000)/10) '%)'],'FontSize',16);
zlabel(['PC3 (' num2str(round(power(3)*1000)/10) '%)'],'FontSize',16);
grid
end
function [mappedA, mapping,power] = compute_mapping(A, type, no_dims, varargin)
%COMPUTE_MAPPING Performs dimensionality reduction on a dataset A
%
%   mappedA = compute_mapping(A, type)
%   mappedA = compute_mapping(A, type, no_dims)
%   mappedA = compute_mapping(A, type, no_dims, ...)
%
% Performs a technique for dimensionality reduction on the data specified 
% in A, reducing data with a lower dimensionality in mappedA.
% The data on which dimensionality reduction is performed is given in A
% (rows correspond to observations, columns to dimensions). A may also be a
% (labeled or unlabeled) PRTools dataset.
% The type of dimensionality reduction used is specified by type. Possible
% values are 'PCA', 'LDA', 'MDS', 'ProbPCA', 'FactorAnalysis', 'GPLVM', 
% 'Sammon', 'Isomap', 'LandmarkIsomap', 'LLE', 'Laplacian', 'HessianLLE', 
% 'LTSA', 'MVU', 'CCA', 'LandmarkMVU', 'FastMVU', 'DiffusionMaps', 
% 'KernelPCA', 'GDA', 'SNE', 'SymSNE', 'tSNE', 'LPP', 'NPE', 'LLTSA', 
% 'SPE', 'Autoencoder', 'LLC', 'ManifoldChart', 'CFA', 'NCA', 'MCML', and 'LMNN'. 
% The function returns the low-dimensional representation of the data in the 
% matrix mappedA. If A was a PRTools dataset, then mappedA is a PRTools 
% dataset as well. For some techniques, information on the mapping is 
% returned in the struct mapping.
% The variable no_dims specifies the number of dimensions in the embedded
% space (default = 2). For the supervised techniques ('LDA', 'GDA', 'NCA', 
% 'MCML', and 'LMNN'), the labels of the instances should be specified in 
% the first column of A (using numeric labels). 
%
%   mappedA = compute_mapping(A, type, no_dims, parameters)
%   mappedA = compute_mapping(A, type, no_dims, parameters, eig_impl)
%
% Free parameters of the techniques can be defined as well (on the place of
% the dots). These parameters differ per technique, and are listed below.
% For techniques that perform spectral analysis of a sparse matrix, one can 
% also specify in eig_impl the eigenanalysis implementation that is used. 
% Possible values are 'Matlab' and 'JDQR' (default = 'Matlab'). We advice
% to use the 'Matlab' for datasets of with 10,000 or less datapoints; 
% for larger problems the 'JDQR' might prove to be more fruitful. 
% The free parameters for the techniques are listed below (the parameters 
% should be provided in this order):
%
%   PCA:            - none
%   LDA:            - none
%   MDS:            - none
%   ProbPCA:        - <int> max_iterations -> default = 200
%   FactorAnalysis: - none
%   GPLVM:          - <double> sigma -> default = 1.0
%   Sammon:         - none
%   Isomap:         - <int> k -> default = 12
%   LandmarkIsomap: - <int> k -> default = 12
%                   - <double> percentage -> default = 0.2
%   LLE:            - <int> k -> default = 12
%                   - <char[]> eig_impl -> {['Matlab'], 'JDQR'}
%   Laplacian:      - <int> k -> default = 12
%                   - <double> sigma -> default = 1.0
%                   - <char[]> eig_impl -> {['Matlab'], 'JDQR'}
%   HessianLLE:     - <int> k -> default = 12
%                   - <char[]> eig_impl -> {['Matlab'], 'JDQR'}
%   LTSA:           - <int> k -> default = 12
%                   - <char[]> eig_impl -> {['Matlab'], 'JDQR'}
%   MVU:            - <int> k -> default = 12
%                   - <char[]> eig_impl -> {['Matlab'], 'JDQR'}
%   CCA:            - <int> k -> default = 12
%                   - <char[]> eig_impl -> {['Matlab'], 'JDQR'}
%   LandmarkMVU:    - <int> k -> default = 5
%   FastMVU:        - <int> k -> default = 5
%                   - <logical> finetune -> default = true
%                   - <char[]> eig_impl -> {['Matlab'], 'JDQR'}
%   DiffusionMaps:  - <double> t -> default = 1.0
%                   - <double> sigma -> default = 1.0
%   KernelPCA:      - <char[]> kernel -> {'linear', 'poly', ['gauss']} 
%                   - kernel parameters: type HELP GRAM for info
%   GDA:            - <char[]> kernel -> {'linear', 'poly', ['gauss']} 
%                   - kernel parameters: type HELP GRAM for info
%   SNE:            - <double> perplexity -> default = 30
%   SymSNE:         - <double> perplexity -> default = 30
%   tSNE:           - <int> initial_dims -> default = 30
%                   - <double> perplexity -> default = 30
%   LPP:            - <int> k -> default = 12
%                   - <double> sigma -> default = 1.0
%                   - <char[]> eig_impl -> {['Matlab'], 'JDQR'}
%   NPE:            - <int> k -> default = 12
%                   - <char[]> eig_impl -> {['Matlab'], 'JDQR'}
%   LLTSA:          - <int> k -> default = 12
%                   - <char[]> eig_impl -> {['Matlab'], 'JDQR'}
%   SPE:            - <char[]> type -> {['Global'], 'Local'}
%                   - if 'Local': <int> k -> default = 12
%   Autoencoder:    - <double> lambda -> default = 0
%   LLC:            - <int> k -> default = 12
%                   - <int> no_analyzers -> default = 20
%                   - <int> max_iterations -> default = 200
%                   - <char[]> eig_impl -> {['Matlab'], 'JDQR'}
%   ManifoldChart:  - <int> no_analyzers -> default = 40
%                   - <int> max_iterations -> default = 200
%                   - <char[]> eig_impl -> {['Matlab'], 'JDQR'}
%   CFA:            - <int> no_analyzers -> default = 2
%                   - <int> max_iterations -> default = 200
%   NCA:            - <double> lambda -> default = 0.0
%   MCML:           - none
%   LMNN:           - <int> k -> default = 3
%
%
% In the parameter list above, {.., ..} indicates a list of options, and []
% indicates the default setting. The variable k indicates the number of 
% nearest neighbors in a neighborhood graph. Alternatively, k may also have 
% the value 'adaptive', indicating the use of adaptive neighborhood selection
% in the construction of the neighborhood graph. Note that in LTSA and
% HessianLLE, the setting 'adaptive' might cause singularities. Using the
% JDQR-solver or a fixed setting of k might resolve this problem. SPE does
% not yet support adaptive neighborhood selection.
% 
% The variable sigma indicates the variance of a Gaussian kernel. The 
% parameters no_analyzers and max_iterations indicate repectively the number
% of factor analyzers that is used in an MFA model and the number of 
% iterations that is used in an EM algorithm. 
%
% The variable lambda represents an L2-regularization parameter.


% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology


    welcome;
    
    % Check inputs
    if nargin < 2
        error('Function requires at least two inputs.');
    end
    if ~exist('no_dims', 'var')
        no_dims = 2;
    end
    if ~isempty(varargin) && strcmp(varargin{length(varargin)}, 'JDQR')
        eig_impl = 'JDQR';
        varargin(length(varargin)) = [];
    elseif ~isempty(varargin) && strcmp(varargin{length(varargin)}, 'Matlab')
        eig_impl = 'Matlab';
        varargin(length(varargin)) = [];
    else
        eig_impl = 'Matlab';
    end        
    mapping = struct;
    
    % Handle PRTools dataset
    if strcmp(class(A), 'dataset')
        prtools = 1;
        AA = A;
        if ~strcmp(type, {'LDA', 'FDA', 'GDA', 'KernelLDA', 'KernelFDA', 'MCML', 'NCA', 'LMNN'})
            A = A.data;
        else
            A = [double(A.labels) A.data];
        end
    else 
        prtools = 0;
    end
    
    % Make sure there are no duplicates in the dataset
    A = double(A);
%     if size(A, 1) ~= size(unique(A, 'rows'), 1)
%         error('Please remove duplicates from the dataset first.');
%     end
    
    % Check whether value of no_dims is correct
    if ~isnumeric(no_dims) || no_dims > size(A, 2) || ((no_dims < 1 || round(no_dims) ~= no_dims) && ~any(strcmpi(type, {'PCA', 'KLM'})))
        error('Value of no_dims should be a positive integer smaller than the original data dimensionality.');
    end
    
    % Switch case
    switch type
        case {'PCA', 'KLM'}
            % Compute PCA mapping
			[mappedA, mapping,power] = pcadr(A, no_dims);
            mapping.name = 'PCA';
        otherwise
            error('Unknown dimensionality reduction technique.');
    end
    
    % JDQR makes empty figure; close it
    if strcmp(eig_impl, 'JDQR')
        close(gcf);
    end
    
    % Handle PRTools dataset
    if prtools == 1
        if sum(strcmp(type, {'Isomap', 'LandmarkIsomap', 'FastMVU'}))
            AA = AA(mapping.conn_comp,:);
        end
        AA.data = mappedA;
        mappedA = AA;
    end
end
function [mappedX, mapping,power] = pcadr(X, no_dims)
%PCA Perform the PCA algorithm
%
%   [mappedX, mapping] = pca(X, no_dims)
%
% The function runs PCA on a set of datapoints X. The variable
% no_dims sets the number of dimensions of the feature points in the 
% embedded feature space (no_dims >= 1, default = 2). 
% For no_dims, you can also specify a number between 0 and 1, determining 
% the amount of variance you want to retain in the PCA step.
% The function returns the locations of the embedded trainingdata in 
% mappedX. Furthermore, it returns information on the mapping in mapping.
%
%

% This file is part of the Matlab Toolbox for Dimensionality Reduction.
% The toolbox can be obtained from http://homepage.tudelft.nl/19j49
% You are free to use, change, or redistribute this code in any way you
% want for non-commercial purposes. However, it is appreciated if you 
% maintain the name of the original author.
%
% (C) Laurens van der Maaten, Delft University of Technology


    if ~exist('no_dims', 'var')
        no_dims = 2;
    end
	
	% Make sure data is zero mean
    mapping.mean = mean(X, 1);
	X = bsxfun(@minus, X, mapping.mean);

	% Compute covariance matrix
    if size(X, 2) < size(X, 1)
        C = cov(X);
    else
        C = (1 / size(X, 1)) * (X * X');        % if N>D, we better use this matrix for the eigendecomposition
    end
	
	% Perform eigendecomposition of C
	C(isnan(C)) = 0;
	C(isinf(C)) = 0;
    [M, lambda] = eig(C);
    
    % Sort eigenvectors in descending order
    [lambda, ind] = sort(diag(lambda), 'descend');
    if no_dims < 1
        no_dims = find(cumsum(lambda ./ sum(lambda)) >= no_dims, 1, 'first');
        disp(['Embedding into ' num2str(no_dims) ' dimensions.']);
    end
    if no_dims > size(M, 2)
        no_dims = size(M, 2);
        warning(['Target dimensionality reduced to ' num2str(no_dims) '.']);
    end
	M = M(:,ind(1:no_dims));
    lam_var=lambda/sum(lambda);
    lambda = lambda(1:no_dims);
    power=lam_var;
% 	display(lam_var(1:no_dims)');
	% Apply mapping on the data
    if ~(size(X, 2) < size(X, 1))
        M = bsxfun(@times, X' * M, (1 ./ sqrt(size(X, 1) .* lambda))');     % normalize in order to get eigenvectors of covariance matrix
    end
    mappedX = X * M;
    
    % Store information for out-of-sample extension
    mapping.M = M;
	mapping.lambda = lambda;
end