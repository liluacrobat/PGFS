function [Pcurve_ext,MST_ext,ProjectionDistance,all_branch] = src_postprocessing(...
    Pcurve,Data,subtype,normal_label)
%% ========================================================================
% Post-processing to extend the end of principal graph using curve fitting
%--------------------------------------------------------------------------
% Input
%   Pcurve: principal points
%   Data: data matrix whose columns are samples
%   subtype: class label of samples
%   normal_label: normal/health sample label 
%--------------------------------------------------------------------------
% Output
%   Pcurve_ext: principal points after postprocessing
%   MST_ext: edge indicator after postprocessing
%   ProjectionDistance: projection distance from samples to the principal graph
%   all_branch: list of branches
%--------------------------------------------------------------------------
% Author: Lu Li
% update history: 08/10/2020
%% ========================================================================
Pcurve = Pcurve';
Data = Data';
if size(Pcurve,1)>1
    %% ====================================================
    % Extend the curves
    % =====================================================
    % search for the root within HC
    root_index = search4Root(Pcurve, Data, subtype, normal_label);
    
    % extend the endpoints
    Pcurve= extendPcurve(Pcurve, Data, root_index(1), subtype,normal_label);
    
    % build a MST from pcurve
    MST = buildMST(Pcurve);
    degree = sum(full(MST)>0, 1);
    display(['Number of end points:' num2str(length(find(degree==1)))]);
    [all_branch, ~, ~] = search4Branch(MST, root_index);
    reassign = true;
    if(reassign)
        % reassign samples to its nearest neighbors in the curve
        distanceMatrix = pdist2(Data, Pcurve);
        [ProjectionDistance,~] = min(distanceMatrix,[],2);
    end
    Pcurve = Pcurve';
else
    disp('Error');
end
Pcurve_ext = Pcurve;
MST_ext = MST;
end
function exPcurve= extendPcurve(Pcurve, Data, root_index, subtype, normal_label)
% Extend pcurve to resovle the problem of many sample pts
% mapped to the same curve end point
% Input:
%      Pcurve: N-by-Feature matrix where N is # of curve pts
%      Data: M-by-Feature matrix where M is # of data samples where
%            pcurve is learned
%      root_index: Root node index on Pcurve as a path staring pt
%      subtype: subtype label of each sample
%      normal_label: the label of normal
% Output:
%      exPcurve: Extended curve pts
%DataProjection: projected data pt on the extended curve
%projectionDist: Distance b/w sample pt with its projection pt
%eXprojectionConnect: projectionConnect matrix after curve extension

small_branch = 0.1; % branch_path_ratio smaller than less value is a small branch
fit_sample   = 0.95;   % Percentage of sample on a branch used to extend curve


DataProjection = zeros(size(Data));
exPcurve = Pcurve;
distanceMatrix = pdist2(Data, Pcurve);
projectionDistance = zeros(1, size(Data, 1));
projectionConnect = zeros(size(Pcurve, 1), ...% row: pcurve pt, col: data pt
    size(Data, 1));     % 1: data pt projected on pcurve pt

% Find the nearest curve pt as sample's projection pt
for n = 1:size(Data, 1)
    [dist, index] = min(distanceMatrix(n,:));
    projectionDistance(n) = dist;
    DataProjection(n,:) = Pcurve(index,:);
    projectionConnect(index,n) = 1;
end
eXprojectionConnect = projectionConnect; % projectionConnect after exten

% Build a MST from pcurve
MST = buildMST(Pcurve);

% Find points to extend

degree = sum(full(MST)>0, 1);
singleton = find(degree==1);

singleton=setdiff(singleton,root_index);
plot_curve = 0;
if plot_curve == 1
    % Plot curve and end points to be extended from
    figure; hold on; grid on;
    plot3(Pcurve(:,1), Pcurve(:,2), Pcurve(:,3), 'o', ...
        'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'w');
    plot3(Pcurve(singleton,1), Pcurve(singleton,2), Pcurve(singleton,3), 'pr', 'MarkerSize', 20);
    set(gca, 'ZColor',[1 1 0],'YColor',[1 1 0],'XColor',[1 1 0],'Color',[0 0 0]);
    for n = 1:length(singleton)
        sampleIndex = find(projectionConnect(singleton(n),:)==1);
        plot3(Data(sampleIndex,1), Data(sampleIndex,2), Data(sampleIndex,3), 'oc');
    end
    title('Extend the principal curve','FontSize',18)
end
% Find all branches in MST

[branch, branch_ratio] = search4Branch(MST, root_index, singleton);
if length(branch)==1
    fit_sample=0.3;
end
[branchMain, branch_ratioMain] = search4MainPath(MST, root_index, singleton);
singleton=[singleton root_index];
branch=[branch branchMain];
branch_ratio=[branch_ratio branch_ratioMain];
% Extend curve

for n = 1:length(singleton)
    
    sampleIndex = find(projectionConnect(singleton(n),:)==1);
    
    sub_branch = Pcurve(branch{n},:);
    if plot_curve == 1
        plot3(sub_branch(:,1), sub_branch(:,2), sub_branch(:,3), '-ob', 'LineWidth',2, 'MarkerSize', 4);
    end
    poly_degree = 3;
    if(branch_ratio(n) > small_branch && length(branch{n})>10)
        if(fit_sample < 1)
            sub_branch = sub_branch(round(length(branch{n})*(1-fit_sample)):end,:);
        end
    else
        % Small branch fit a line
        poly_degree = 1;
    end
    if plot_curve == 1
        plot3(sub_branch(:,1), sub_branch(:,2), sub_branch(:,3), '-oy', 'LineWidth',2, 'MarkerSize',6);
    end
    if ~isempty(sampleIndex)
        [projection, extended, extendConnect] = project2poly(sub_branch, Data(sampleIndex,:), poly_degree);
        
        % Extended curve
        exPcurve = [exPcurve; extended];
        if plot_curve == 1
            plot3(extended(:,1), extended(:,2), extended(:,3), 'yo', 'MarkerSize', 6);
        end
        % Assign projection
        DataProjection(sampleIndex,:) = projection;
        if plot_curve == 1
            plot3(projection(:,1), projection(:,2), projection(:,3), ...
                'o', 'MarkerFaceColor', 'g', 'MarkerEdgeColor', 'g', 'MarkerSize', 6);
        end
        %recalculate  projection distance
        projectionDistance(sampleIndex) = sqrt(sum((Data(sampleIndex,:)-projection).^2, 2));
        
        % Re-assign projectionConnection
        eXprojectionConnect(singleton(n),sampleIndex) = 0;
        tempConnect = zeros(size(extended, 1), size(Data, 1));
        tempConnect(:, sampleIndex) = extendConnect;
        eXprojectionConnect = [eXprojectionConnect; tempConnect]; % Append to end
    end
end
% Remove the small branches generated by extending the curve
exPcurve = removeSmallBranch(exPcurve, Data, subtype, normal_label);
end
function newPcurve = removeSmallBranch(Pcurve, Data, subtype, normal_label)
%% ====================================================
%  Remove extremely small branches
%% ====================================================
distanceMatrix = pdist2(Data, Pcurve);
projectionConnect = zeros(size(Pcurve, 1), ...% row: pcurve pt, col: data pt
    size(Data, 1));     % 1: data pt projected on pcurve pt

% Find the nearest curve pt as sample's projection pt
for n = 1:size(Data, 1)
    [~, index] = min(distanceMatrix(n,:));
    projectionConnect(index,n) = 1;
end
branch2remove = [0];
while ~isempty(branch2remove)
    % Search for root node
    root_index = search4Root(Pcurve, Data, subtype, normal_label);
    % Build a MST from pcurve
    MST = buildMST(Pcurve);
    branch2remove = [];
    % Find all branches in MST
    
    [all_branch, branch_ratio, branch_length] = search4Branch(MST, root_index(1));
    L = zeros(size(Pcurve,1),1);
    for i=1:length(all_branch)
        temp = all_branch{i};
        L(temp(2:end)) = i;
    end
    if length(all_branch)>1
        % Rmove branches with small branches
        for n = 1:length(all_branch)
            branch = all_branch{n};
            if branch_ratio(n)< 0.1
                branch2remove = [branch2remove, branch(2:end)];
            end
        end
        newPcurve = Pcurve;
        %         newL = subtype;
        %         newL(branch2remove) = [];
        newPcurve(branch2remove,:) = [];
    else
        newPcurve = Pcurve;
    end
    Pcurve = newPcurve;
end
%
% plotPCA3D_color(Pcurve',L);
% plotPCA3D_color(newPcurve',newL);
end

function [branch, branch_ratio,branch_length] = search4Branch(MST, root_index, singleton)
% Find all branches from a minimal spanning tree (MST)
% Input:
%       MST: a sparse connection matrix to represent a MST
% Output:
%    branch: pt index of all branches found in MST

degree = sum(full(MST)>0, 1);
% No specified target points, use end points
if(nargin < 3)
    singleton = find(degree==1);
    singleton = setdiff(singleton,root_index);
end

branch = cell(1, length(singleton));
branch_length = zeros(1, length(singleton));
path_length = zeros(1, length(singleton));
% Find branch each end point resides
for n = 1:length(singleton) % Traverse every path from root node
    [path_length(n),  one_path, ~] = graphshortestpath(MST, root_index, singleton(n));
    
    
    % calculate branch
    branch_degree = degree(one_path);
    cross_index = find(branch_degree>2); % cross plot
    if(isempty(cross_index))
        % No cross point on this path, set branch length as one path length
        branch{n} = one_path;
        branch_length(n) = path_length(n);
    else
        % No. of samples from end point to the nearest cross point
        branch{n} = one_path(cross_index(end):end);
        [branch_length(n),  ~, ~] = graphshortestpath(MST, one_path(cross_index(end)), one_path(end));
    end
end
branch_ratio = branch_length./path_length;

end
function [branch, branch_ratio] = search4MainPath(MST, root_index, singleton)
% Find all branches from a minimal spanning tree (MST)
% Input:
%       MST: a sparse connection matrix to represent a MST
% Output:
%    branch: pt index of all branches found in MST

degree = sum(full(MST)>0, 1);
% No specified target points, use end points
if(nargin < 3)
    singleton = find(degree==1);
end

branch = cell(1, length(singleton));
path_length = zeros(1, length(singleton));
% Find branch each end point resides
flag=0;
for n = 1:length(singleton) % Traverse every path from root node
    [path_length(n),  one_path, ~] = graphshortestpath(MST, root_index, singleton(n));
    
    
    % calculate branch
    branch_degree = degree(one_path);
    cross_index = find(branch_degree>2); % cross plot
    if(~isempty(cross_index))
        new_root=one_path(cross_index(1));
        [branch, branch_ratio] = search4Branch(MST, new_root, root_index);
        flag=1;
        break;
    end
end
if flag==0
    new_root=one_path(end);
    [branch, branch_ratio] = search4Branch(MST, new_root, root_index);
end
end
function [projection, branch_extend, projectionConnect, branch_fitted] = ...
    project2poly(branch, Data, poly_degree, line_ratio)
% Project Data onto a polynomial line fitted by pcurve in every dimension
% Input:
%       branch: N-by-D curve branch pt
%         Data: M-by-D data sample pt
%  poly_degree: degree of polynomial
%   line_ratio: % qunatile of lambda to estimate extension ratio
% Output:
%   projection: M-by-D data projection on polynomial
%branch_extend: M-by-D extended branch pt
% branch_fitte: M-by-D fitted branch pt
%projectionConnect: M-by-M projection connection matrix

if(nargin < 4)
    line_ratio = 1;
end

% compute branch length
L = transpose(pdist2(branch(1,:), branch));

% Estimate extension ratio
[~, lambda] = project2line(branch(1,:)', branch(end,:)', Data');
extend_ratio = quantile(lambda, line_ratio); % robust to outliers


% Curve fitting of each dimension against length
XX = linspace(L(end)+0.001, max(L)*extend_ratio, size(Data,1));% start from next point
ZZ = linspace(0, max(L)*extend_ratio, size(Data,1)*2);% whole length
branch_extend = zeros(size(Data,1), size(Data,2));
branch_fitted = zeros(size(Data,1)*2, size(Data,2));
% XX = linspace(0, max(L)*extend_ratio, size(Data,1)+size(branch,1));% Extend curve length
% branch_extend = zeros(size(Data,1)+size(branch,1), size(Data,2));

for n = 1:size(branch,2)
    P = polyfit(L, branch(:,n), poly_degree);%3th polynomial
    YY = polyval(P, XX);
    branch_extend(:,n) = YY;
    branch_fitted(:,n) = polyval(P, ZZ);
end


% Calculate projection pt from extended branch
projection = zeros(size(Data));
distanceMatrix = pdist2(branch_extend, Data);
projectionConnect = zeros(size(branch_extend, 1), ...% row:extend pcurve pt col: data
    size(Data, 1));     % 1: data pt projected on pcurve pt
for n = 1:size(Data, 1)
    [~, index] = min(distanceMatrix(:,n));
    projection(n,:) = branch_extend(index,:);
    projectionConnect(index,n) = 1;
end

% Shrink extended curve to the further pt with data projection
with_proj = find(sum(projectionConnect, 2)>0);
last_index = with_proj(end);
branch_extend(last_index+1:end,:) = [];
projectionConnect(last_index+1:end,:) = [];

end
function root_index = search4Root(Pcurve, Data, subtype, normal_label)
%% ====================================================
% Search for root node on Pcurve
% =====================================================

MST = buildMST(Pcurve);
degree = sum(full(MST)>0, 1);
singleton = find(degree==1);
% find the 10 nearest samples of the leaf points
IDX = knnsearch(Data, Pcurve(singleton,:), 'K', 10);
% determine leaf points within HC sample
vote=zeros(size(singleton));
for i=1:length(vote)
    id = IDX(i,:);
    vote(i)=sum(strcmp(subtype(id), normal_label))/length(id);
end
root_index=singleton(vote>=0.1);
% determine the root point (winthin HC and have the largest distance to other leaf ponts)
Not_HC = setdiff(singleton,root_index);
root_dis=zeros(length(root_index),length(Not_HC));
for i=1:length(root_index)
    for j=1:length(Not_HC)
        [root_dis(i,j),  ~, ~] = graphshortestpath(MST, root_index(i), Not_HC(j));
    end
end
[~,idx]=sort(sum(root_dis,2),'descend');
root_index=root_index(idx);
root_index=root_index(1);
end