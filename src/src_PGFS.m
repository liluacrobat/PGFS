function [Weight, Y, B, R, Objective, History, MSE, TreeLength] = src_PGFS(DATA,Para)
%% ========================================================================
% Feature selection via principal graph learning
%--------------------------------------------------------------------------
% Input
%   DATA: data matrix whose columns are samples
%   Para: parameters used by PGFS
%       sigma - kernel width (default:0.01) 
%       beta - tree length regularization (default:10)
%       lambda - sparness regularization (default:16)
%       it - maximum iteration (defayult:50)
%       M - number of points on the principal graph (default: number of samples)
%--------------------------------------------------------------------------
% Output
%    Weight: feature weight
%    Y: latent points on the principal graph 
%    B: edge indicators
%    R: soft assignment coefficients
%    Objective: objective value
%    History: records of the leasrning process
%    MSE: mean squared error
%    TreeLength: tree length
% Author: Lu Li
% update history: 08/10/2020
%% ========================================================================
%% initilization
[dim,N_DATA] = size(DATA);
if nargin<2
    Para.sigma = 0.01;
    Para.beta = 10;
    Para.lambda = 16;
    Para.it = 50;
    Para.M = N_DATA;
end
beta=Para.beta;
sigma=Para.sigma;
lambda=Para.lambda;
T=Para.it;
M=Para.M;
Para.distance = 'euclidean';

Theta=zeros(T,1);
Weight = ones(dim,1)/sqrt(dim); %initial guess
if Para.M<size(DATA,2)
    [~,F]=kmeans(DATA',M);
    F=F';
else
    F=DATA;
end
History = zeros(dim,T+1);
History(:,1) = Weight;
Objective = [0];
Difference = 1;
t = 0;
Original_dim = dim;
Original_index = 1:dim;
OriginalDATA = DATA;
NZ_index=1:dim;
%% main body of the code
while Difference>0.001&&t<=T
    t=t+1;
    display(['iter ' num2str(t)]);

    % calculate the MST
    % X is N*D
    Xw=(sqrt(Weight)*ones(1,N_DATA).*DATA)';
    Fw=(sqrt(Weight)*ones(1,M).*F)';
    % Distance matrix
    Fdis=pdist2(Fw,Fw,Para.distance);
    Fdis=Fdis.^2;
    Xdis=pdist2(Xw,Fw,Para.distance);
    Xdis=Xdis.^2;
    %     G(G==0)=eps;
    G=Fdis-diag(diag(Fdis));
    % Construct the MST
    [Tree, ~] = graphminspantree(sparse(G),'METHOD','Kruskal');
    MST=full(Tree);
    B=MST>0;
    B=(B+B')/2;
    % Calculate the p_ij
    PTT=sort(Xdis');
    PTT=PTT';
    disP=Xdis-PTT(:,1)*ones(1,size(Xdis,2));
    P_ij= exp(-disP/sigma);
    SUM_P_ij=sum(P_ij,2);
    SUM_P_ij(SUM_P_ij==0)=eps;
    R=P_ij./(SUM_P_ij*ones(1,size(F,2)));
    % Calculate the F
    Rdiag=diag(R'*ones(N_DATA,1));
    BL=(diag(B*ones(M,1))-B);
    XR=DATA*R;
    RN=ones(N_DATA)/(N_DATA);
    XN=DATA*RN;
    FINV=beta*BL+Rdiag;
    F=XR/FINV;
    Xerror=diag(DATA*DATA'-2*XR*F'+F*Rdiag*F');
    Xerror=sum(Xerror'*Weight);
    Xlength=diag(F*BL*F');
    Xlength=sum(Xlength'*Weight);
    C=-diag(DATA*DATA'-2*XR*F'+F*FINV*F'-1*DATA*(diag(ones(1,N_DATA))-RN)*DATA');
    C_original = C;
    C(C<0) = 0;
    Weight = C/norm(C);
    
    if sum(Weight)>lambda
        min_delta = 0;
        max_delta = max(C_original);
        min_delta_dif = 10^(-3); % resolution, stop if the difference is less than this number.
        min_conv_dif = 10^(-3);  % convergence stopping, if abs(weightsum-old_weightsum) less than this number
        dif_conv = 1;%sum(Weight)-beta; %to verify convergence stopping
        dif_delta = 1;%max_delta-min_delta;
        while dif_delta > min_delta_dif && dif_conv > min_conv_dif
            mid_delta = min_delta + (max_delta - min_delta)/2;
            C = C_original - mid_delta;
            C(C<0) = 0;
            Weight = C/norm(C);
            if sum(Weight) > lambda
                min_delta = mid_delta;
            end
            if sum(Weight) <= lambda
                max_delta = mid_delta;
            end
            dif_conv = abs(sum(Weight)-lambda);
            dif_delta = max_delta - min_delta;
        end
    end
    LR=R;
    LR(R==0)=eps;
    Entropy=sigma*sum(diag(R'*log(LR)));
    
    
    Objective = [Objective, (-(Weight(:)')*C_original+Entropy)];
    temp = zeros(Original_dim,1);
    temp(Original_index) = Weight;
    Weight = temp;
    Original_index = 1:Original_dim;
    Theta(t) = Difference;
    Difference = norm(Weight-History(:,t));
    History(:,t+1) = Weight;
    Y=F;
    
    if t>4
        SUM = sum(History(:,t-4:end),2);
        Zero_index = find(SUM==0);
        
        DATA = OriginalDATA;
        DATA(Zero_index,:) = []; %if the sum of a weight in the past three iteration is zero, the feature is removed.
        ttF=F;
        F=zeros(Original_dim,M);
        F(NZ_index,:)=ttF;
        Y=F;
        NZ_index=find(SUM~=0);
        F(Zero_index,:) = [];
        Original_index(Zero_index) = [];
        dim = size(DATA,1);
        Weight(Zero_index) = [];
    end
end
Weight = History(:,t+1);
History = History(:, 1:t+1);
MSE = Xerror;
TreeLength = Xlength;
end