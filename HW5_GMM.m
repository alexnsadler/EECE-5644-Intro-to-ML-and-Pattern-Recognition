%% Dataset 1
% number of clusters
k = 2;
restarts = 3;
max_k = 4;

dataset1 = load('/Users/alexsadler/Downloads/gauss2.mat');
dataset1 = dataset1.gauss2;

sample1D1 = dataset1(dataset1(:,3)==0,:);
sample2D1 = dataset1(dataset1(:,3)==1,:);
sample3D1 = dataset1(dataset1(:,3)==2,:);
%% Dataset 1 GMM

[llhoods,bic,initInd,mus,covs,newInd,newmus,newcovs] = bestEM_GMM(dataset1,k,restarts);


% plot data and initial means and covariance elipses
figure()
plot_gauss(sample1D1,mus{1},covs{1},1,2)
plot_gauss(sample2D1,mus{2},covs{2},1,2)
xlabel('x')
ylabel('y')
title('Initial Means and Covariances - Dataset 1')

% plot final means and covariance elipses after running algorithm
figure()
plot_gauss(sample1D1,newmus{1},newcovs{1},1,2)
plot_gauss(sample2D1,newmus{2},newcovs{2},1,2)
xlabel('x')
ylabel('y')
title('GMM Clustering - Dataset 1')


% plot log-likelihoods from each iteration
figure()
plot(llhoods)
xlabel('Number of Iterations')
ylabel('Log-likelihoods')
title('Log-likelihood by Number of Iterations')

%%
% BIC and log-likelihood scores from GMM with 1 to max_k clusters
[bics1,llhoods1] = bic_clust(dataset1,restarts,max_k);
[bics1 llhoods1];



%% Dataset 2
k = 3;  % number of clusters
restarts = 50;
max_k = 8;

dataset2 = load('/Users/alexsadler/Downloads/gauss3.mat');
dataset2 = dataset2.gauss3;

sample1D2 = dataset2(dataset2(:,3)==0,:);
sample2D2 = dataset2(dataset2(:,3)==1,:);
sample3D2 = dataset2(dataset2(:,3)==2,:);
%% Dataset 2 GMM
[llhoods,bic,mus,covs,newmus,newcovs] = bestEM_GMM(dataset2,k,restarts);

% plot data and initial means and covariance elipses
figure()
plot_gauss(sample1D2,mus{1},covs{1},1,2)
plot_gauss(sample2D2,mus{2},covs{2},1,2)
plot_gauss(sample3D2,mus{3},covs{3},1,2)
xlabel('x')
ylabel('y')
title('Initial Means and Covariances - Dataset 2')

% plot final means and covariance elipses after running algorithm
figure()
plot_gauss(sample1D2,newmus{1},newcovs{1},1,2)
plot_gauss(sample2D2,newmus{2},newcovs{2},1,2)
plot_gauss(sample3D2,newmus{3},newcovs{3},1,2)
xlabel('x')
ylabel('y')
title('GMM Clustering - Dataset 2')

% plot log-likelihoods from each iteration
figure()
plot(llhoods)
xlabel('Number of Iterations')
ylabel('Log-likelihoods')
title('Log-likelihood by Number of Iterations')

% BIC and log-likelihood scores from GMM with 1 to max_k clusters
[bics2,llhoods2] = bic_clust(dataset2,restarts,max_k);
bicsd2 = [bics2 llhoods2]



%% Dataset 3
max_k = 8;             % number of iterations for k-means
restarts = 100;                     % number of restarts
k = 3;                           % number of clusters

dataset3 = load('/Users/alexsadler/Downloads/iris.mat');
dataset3 = dataset3.iris;

% split dataset based on class for scatter plot
sample1D3 = dataset3(dataset3(:,5)==1,1:2);
sample2D3 = dataset3(dataset3(:,5)==2,1:2);
sample3D3 = dataset3(dataset3(:,5)==3,1:2);
%% Dataset 3 GMM
[llhoods,bic,mus,covs,newmus,newcovs] = bestEM_GMM(dataset3,k,restarts);

% plot data and initial means and covariance elipses
figure()
plot_gauss(sample1D3,mus{1},covs{1},1,2)
plot_gauss(sample2D3,mus{2},covs{2},1,2)
plot_gauss(sample3D3,mus{3},covs{3},1,2)
xlabel('x')
ylabel('y')
title('Initial Means and Covariances - Dataset 3')

% plot final means and covariance elipses after running algorithm
figure()
plot_gauss(sample1D3,newmus{1},newcovs{1},1,2)
plot_gauss(sample2D3,newmus{2},newcovs{2},1,2)
plot_gauss(sample3D3,newmus{3},newcovs{3},1,2)
xlabel('x')
ylabel('y')
title('GMM Clustering - Dataset 3')

% plot log-likelihoods from each iteration
figure()
plot(llhoods)
xlabel('Number of Iterations')
ylabel('Log-likelihoods')
title('Log-likelihood by Number of Iterations')

% BIC and log-likelihood scores from GMM with 1 to max_k clusters
[bics3,llhoods3] = bic_clust(dataset3,restarts,max_k);
bicsd3 = [bics3 llhoods3]



%% Dataset 4
max_iterations = 10;               % number of iterations for k-means
r_reps = 1000;                     % number of restarts

dataset4 = load('/Users/alexsadler/Downloads/a3geyser.mat');
dataset4 = dataset4.a3geyser';
%% Dataset 4 GMM 1 Cluster

[llhoods,bic,mus,covs,newmus,newcovs] = bestEM_GMM(dataset4,1,restarts);

% plot data and initial means and covariance elipses
figure()
plot_gauss(dataset4,mus{1},covs{1},1,2)
xlabel('x')
ylabel('y')
title('Initial Means and Covariances - Dataset 4')

% plot final means and covariance elipses after running algorithm
figure()
plot_gauss(dataset4,newmus{1},newcovs{1},1,2)
xlabel('x')
ylabel('y')
title('GMM Clustering - Dataset 4')

% plot log-likelihoods from each iteration
figure()
plot(llhoods)
xlabel('Number of Iterations')
ylabel('Log-likelihoods')
title('Log-likelihood by Number of Iterations')
%% Dataset 4 GMM 2 Clusters
clc
[llhoods,bic,initInd,mus,covs,newInd,newmus,newcovs] = bestEM_GMM(dataset4,2,restarts);

% plot data and initial means and covariance elipses
figure()
plot_gauss(dataset4(initInd==1,1:2),mus{1},covs{1},1,2)
plot_gauss(dataset4(initInd==2,1:2),mus{2},covs{2},1,2)
xlabel('x')
ylabel('y')
title('Initial Means and Covariances - Dataset 4')

% plot final means and covariance elipses after running algorithm
figure()
plot_gauss(dataset4(newInd==1,1:2),newmus{1},newcovs{1},1,2)
plot_gauss(dataset4(newInd==2,1:2),newmus{2},newcovs{2},1,2)
xlabel('x')
ylabel('y')
title('GMM Clustering - Dataset 4')

% plot log-likelihoods from each iteration
figure()
plot(llhoods)
xlabel('Number of Iterations')
ylabel('Log-likelihoods')
title('Log-likelihood by Number of Iterations')
%% Dataset 4 GMM 3 Clusters

[llhoods,bic,initInd,mus,covs,newInd,newmus,newcovs] = bestEM_GMM(dataset4,3,restarts);

% plot data and initial means and covariance elipses
figure()
plot_gauss(dataset4(initInd==1,1:2),mus{1},covs{1},1,2)
plot_gauss(dataset4(initInd==2,1:2),mus{2},covs{2},1,2)
plot_gauss(dataset4(initInd==3,1:2),mus{3},covs{3},1,2)
xlabel('x')
ylabel('y')
title('Initial Means and Covariances - Dataset 4')

% plot final means and covariance elipses after running algorithm
figure()
plot_gauss(dataset4(newInd==1,1:2),newmus{1},newcovs{1},1,2)
plot_gauss(dataset4(newInd==2,1:2),newmus{2},newcovs{2},1,2)
plot_gauss(dataset4(newInd==3,1:2),newmus{3},newcovs{3},1,2)
xlabel('x')
ylabel('y')
title('GMM Clustering - Dataset 4')

% plot log-likelihoods from each iteration
figure()
plot(llhoods)
xlabel('Number of Iterations')
ylabel('Log-likelihoods')
title('Log-likelihood by Number of Iterations')
%% Dataset 4 GMM 4 Clusters

[llhoods,bic,initInd,mus,covs,newInd,newmus,newcovs] = bestEM_GMM(dataset4,4,restarts);

% plot data and initial means and covariance elipses
figure()
plot_gauss(dataset4(initInd==1,1:2),mus{1},covs{1},1,2)
plot_gauss(dataset4(initInd==2,1:2),mus{2},covs{2},1,2)
plot_gauss(dataset4(initInd==3,1:2),mus{3},covs{3},1,2)
plot_gauss(dataset4(initInd==4,1:2),mus{4},covs{4},1,2)
xlabel('x')
ylabel('y')
title('Initial Means and Covariances - Dataset 4')

% plot final means and covariance elipses after running algorithm
figure()
plot_gauss(dataset4(newInd==1,1:2),newmus{1},newcovs{1},1,2)
plot_gauss(dataset4(newInd==2,1:2),newmus{2},newcovs{2},1,2)
plot_gauss(dataset4(newInd==3,1:2),newmus{3},newcovs{3},1,2)
plot_gauss(dataset4(newInd==4,1:2),newmus{4},newcovs{4},1,2)
xlabel('x')
ylabel('y')
title('GMM Clustering - Dataset 4')

% plot log-likelihoods from each iteration
figure()
plot(llhoods)
xlabel('Number of Iterations')
ylabel('Log-likelihoods')
title('Log-likelihood by Number of Iterations')
%% Dataset 4 GMM 5 Clusters

[llhoods,bic,initInd,mus,covs,newInd,newmus,newcovs] = bestEM_GMM(dataset4,5,restarts);

% plot data and initial means and covariance elipses
figure()
plot_gauss(dataset4(initInd==1,1:2),mus{1},covs{1},1,2)
plot_gauss(dataset4(initInd==2,1:2),mus{2},covs{2},1,2)
plot_gauss(dataset4(initInd==3,1:2),mus{3},covs{3},1,2)
plot_gauss(dataset4(initInd==4,1:2),mus{4},covs{4},1,2)
plot_gauss(dataset4(initInd==5,1:2),mus{5},covs{5},1,2)
xlabel('x')
ylabel('y')
title('Initial Means and Covariances - Dataset 4')

% plot final means and covariance elipses after running algorithm
figure()
plot_gauss(dataset4(newInd==1,1:2),newmus{1},newcovs{1},1,2)
plot_gauss(dataset4(newInd==2,1:2),newmus{2},newcovs{2},1,2)
plot_gauss(dataset4(newInd==3,1:2),newmus{3},newcovs{3},1,2)
plot_gauss(dataset4(newInd==4,1:2),newmus{4},newcovs{4},1,2)
plot_gauss(dataset4(newInd==5,1:2),newmus{5},newcovs{5},1,2)
xlabel('x')
ylabel('y')
title('GMM Clustering - Dataset 4')

% plot log-likelihoods from each iteration
figure()
plot(llhoods)
xlabel('Number of Iterations')
ylabel('Log-likelihoods')
title('Log-likelihood by Number of Iterations')
%% Dataset 4 GMM 6 Clusters

[llhoods,bic,initInd,mus,covs,newInd,newmus,newcovs] = bestEM_GMM(dataset4,6,10);

% plot data and initial means and covariance elipses
figure()
plot_gauss(dataset4(initInd==1,1:2),mus{1},covs{1},1,2)
plot_gauss(dataset4(initInd==2,1:2),mus{2},covs{2},1,2)
plot_gauss(dataset4(initInd==3,1:2),mus{3},covs{3},1,2)
plot_gauss(dataset4(initInd==4,1:2),mus{4},covs{4},1,2)
plot_gauss(dataset4(initInd==5,1:2),mus{5},covs{5},1,2)
plot_gauss(dataset4(initInd==6,1:2),mus{6},covs{6},1,2)
xlabel('x')
ylabel('y')
title('Initial Means and Covariances - Dataset 4')

% plot final means and covariance elipses after running algorithm
figure()
plot_gauss(dataset4(newInd==1,1:2),newmus{1},newcovs{1},1,2)
plot_gauss(dataset4(newInd==2,1:2),newmus{2},newcovs{2},1,2)
plot_gauss(dataset4(newInd==3,1:2),newmus{3},newcovs{3},1,2)
plot_gauss(dataset4(newInd==4,1:2),newmus{4},newcovs{4},1,2)
plot_gauss(dataset4(newInd==5,1:2),newmus{5},newcovs{5},1,2)
plot_gauss(dataset4(newInd==6,1:2),newmus{6},newcovs{6},1,2)
xlabel('x')
ylabel('y')
title('GMM Clustering - Dataset 4')

% plot log-likelihoods from each iteration
figure()
plot(llhoods)
xlabel('Number of Iterations')
ylabel('Log-likelihoods')
title('Log-likelihood by Number of Iterations')

%% BIC and log-likelihood scores from GMM with 1 to max_k clusters
[bics4,llhoods4] = bic_clust(dataset4,5,max_k);
bicsd4 = [bics4 llhoods4]

%% GMM Functions

function [bics, llhoods] = bic_clust(data,r,max_c)
    % function to implement GMM on a dataset with 1 to max_c clusters and
    % return the BIC scores and log-likehood values for each iteration
    
    bics = zeros(max_c,1);
    llhoods = zeros(max_c,1);
    for c=1:max_c
        [llhoodsidx,bicidx,~,~,~,~] = bestEM_GMM(data,c,r);
        bics(c) = bicidx;
        llhoods(c) = max(llhoodsidx);
    end
    
end
 

function [initMu, initCov] = initData(data,c)
    % initialize means to be randomly selected points
    initMu = initCentroids(data,c);

    % assign each datapoint to nearest mean
    [~, index] = nearestCentroids(data,initMu);

    % initialize cov matrices as sample covariances of initial assignments
    initCov = cell(c,1);
    for idx=1:c
        initCov{idx} = cov(data(index==idx,1:2));
    end
    
    
end


function [llhoods,bic,initInd,initMus,initCovs,newInd,newMus,newCovs] = bestEM_GMM(data,c,r)
    % function to implement GMM with given r random restarts and then
    % return the best GMM implementation based on which iteration had the
    % highest overal log-likelihood
    
    init_mus = cell(r,1);
    init_covs = cell(r,1); 
    mus_list = cell(r,1);
    covs_list = cell(r,1);
    llhoods_list = zeros(50,r);
    
    max_llhoods = zeros(r,1);
    for idx=1:r
        
        % select initial means and covariance matrices
        [mus, covs] = initData(data,c);
        init_mus{idx} = mus;
        init_covs{idx} = covs;
        
        % implement GMM
        [llhood, musidx, covsidx] = EM_GMM(data,mus,covs,c);
        
        % store updated means and covs in mus_list and covs_list
        mus_list{idx} = musidx;
        covs_list{idx} = covsidx;
        
        % get number of iterations EM_GMM ran until convergence and store
        % in llhoods_list
        num_iter = length(llhood);
        llhoods_list(1:num_iter,idx) = llhood;
        
        % store the final/max value of log-likelihood from GMM
        max_llhoods(idx) = max(llhood);
    end
    
    % get the value and index of the highest log-likelihood from all runs
    [val, index] = max(max_llhoods);
    
    % get list of log-likelihoods from each iteration of GMM from the
    % highest overall log-likelihood
    llhoods = llhoods_list(:,index);
    llhoods = llhoods(llhoods ~= 0);
    
    % get initial means and covariances from best GMM run
    initMus = init_mus{index};
    initCovs = init_covs{index};
    
    % get updated means and covariances from best GMM run
    newMus = mus_list{index};
    newCovs = covs_list{index};
    
    % bic = llhood - (number of paramaters * log(number of data points))
    % where number of paramaters is (number of means) * (number of unique
    % elements in cov matrix) * (number of clusters) / 2
    bic = val - ((c * 3 * c) / 2) * log(size(data,1));
    
    
    [~, initInd] = max(evalGaus(data(:,1:2), initMus, initCovs),[],2);
    [~, newInd] = max(evalGaus(data(:,1:2), newMus, newCovs),[],2);
    
end


function [likelihoods, newMus, newCovs] = EM_GMM(data,initmus,initcovs,c)

    % initialize likelihoods to column of 50 (max number of iterations)
    % zeros
    likelihoods = zeros(50,1);
    
    % calculate variance of overall dataset for minimum cov threshold
    dat = data(:,1:2);
    var_dat = var(dat(:));
    
    % E-Step
    probs = evalGaus(data,initmus,initcovs);
    
    converged = 0;
    t=1;
    while ~converged
        % M-Step
        % mu_list is list of means weighted by the probability of each data
        % point belonging to each cluster (similar to k-means hard clustering
        % but now use probabilistic assignment)
        mu_list = cell(c,1);
        cov_list = cell(c,1);    
        for idx=1:c
            % multiply each datapoint by probability of it belonging to each
            % class and divide by sum of probabilities (weights)        
            normed_probs = probs ./ sum(probs,2);
            weighted_probs = normed_probs ./ sum(normed_probs);
            new_mu = sum(weighted_probs(:,idx) .* data(:,1:2));

            cov_idx = data(:,1:2) - new_mu;    % remove mean
            cov_idx = cov_idx' * (cov_idx .* repmat(weighted_probs(:,idx),1,2)); % weighted cov matrix
            
            % set minimum threshold for cov matrix, so can work around
            % singular matrices
            diags = diag(cov_idx);

            min_threshold = .01 * var_dat;
            for d=1:length(diags)
                if diags(d) < min_threshold
                    diags(d) = min_threshold;
                    
                end
            end
            
            mu_list{idx} = new_mu;
            cov_list{idx} = cov_idx;

        end
        
        % E-Step
        probs = evalGaus(data,mu_list,cov_list);
        
        % log-likelihoods
        logLikelihood = sum(log(sum(probs,2)));
        likelihoods(t) = logLikelihood;
        if t > 1
            llhoodChange = abs(abs((likelihoods(t) - likelihoods(t-1))) / likelihoods(1));
            % convergence is true if change in llhood is less than .001%
            converged = (llhoodChange < .00001);
        end
        
        % set maximimum number of iterations to halt algorithm
        if t > 50
            converged = 1;
        end
            
        % increment t
        t = t + 1;
        
    
    end
    % filter likelihood array to non-zero entries
    likelihoods = likelihoods(likelihoods ~= 0);
    
    newMus = mu_list;
    newCovs = cov_list;
    
end


function probs = evalGaus(data,mus,covs)
    m = length(data);
    c = length(mus);
    
    % use mvnpdf to determine prob that each datapoint belongs to each of
    % the clusters
    probs = zeros(m,c);
    for idx=1:c
        pdfs = mvnpdf(data(:,1:2),mus{idx},covs{idx});
        probs(:,idx) = pdfs;
    end
   
end


function centroids = initCentroids(data, k)
    
    % sample n points based on number of clusters from 0 to length(data), 
    % and then use sample to index into data
    centroids = data(randsample(length(data),k),1:2);
    
    mus = cell(k,1);
    for idx=1:k
        mus{idx} = centroids(idx,:);
    end
    
    centroids = mus;
    
    
end


function [errors, clust_assignments] = nearestCentroids(data, centroids)
    
    k = size(centroids,1);      % get num_clusters
    m = size(data, 1);          % get number of datapoints
    dists = zeros(m,k);         % initialize distances values to 0
    
    
    for row=1:m                 % loop through each datapoint
        for clust=1:k           % loop through number of clusters
            % calculate euclidean distance between each point and each
            % centroid, then add to dists
            xdist = (centroids{clust}(1) - data(row,1)).^2;
            ydist = (centroids{clust}(2) - data(row,2)).^2;
            euc_dist = sqrt(xdist + ydist);
            dists(row, clust) = euc_dist;
        end
    end
    
    % get the minimum error and index for each row (to classify each point)
    [errors, clust_assignments] = min(dists,[],2);
    % sum up all errors to get total sum-squared-errors of datapoints from
    % centroid
    errors = sum(errors);
  
    
end


function  plot_gauss(data, mean, covar,xaxis,yaxis)
% PLOT_GAUSS:  plot_gauss(data, mean, covar,xaxis,yaxis)
%
%  MATLAB function to plot a 2 dimensional scatter plot of
%  sample data (using xaxis and yaxis as the column indices into
%  an N x d data matrix) and superpose the mean of a Gaussian
%  model and its "covariance ellipse"  on this data.
%                                      ICS 274 Demo Function
%
%  INPUTS:
%    data: N x d matrix of d-dimensional feature vectors
%   means: 1 x d matrix: the d-dimensional mean of the Gaussian model
%   covar: d x d matrix: the dxd covariance matrix of the Gaussian model
%   xaxis: an integer between 1 and d indicating which of the features is 
%         to be used as the x axis
%   yaxis: another integer between 1 and d for the y axis

 plot(data(:,xaxis), data(:,yaxis), '.');
 %gscatter(data(:,xaxis),data(:,yaxis),data(:,3))
 hold on
 markers = get(gca, 'Children');
 markerColor = get(markers(1), 'Color');
 plot(mean(xaxis), mean(yaxis), '*', 'MarkerSize', 14, ...
     'MarkerEdgeColor', 'k', 'LineWidth', 2);
 hold on

% Calculate contours for the 2d normals at Mahalanobis dist = constant
 mhdist = 3;

% Extract the relevant dimensions from the ith component matrix
 covar2d = [covar(xaxis,xaxis) covar(xaxis,yaxis); covar(yaxis,xaxis) covar(yaxis,yaxis)];


% Use some results from standard geometry to figure out the ellipse
% equations from the covariance matrix. Probably other ways to
% do this, e.g., finding the principal component directions, etc.
% See Fraleigh, p.431 for details on rotating the ellipse, etc
 icov = inv(covar2d);
 a = icov(1,1);
 c = icov(2,2);
% we don't check if this is zero: which occasionally causes
% problems when we divide by it later! needs to be fixed.
 b = icov(1,2)*2;

 theta = 0.5*acot( (a-c)/b);

 sc = sin(theta)*cos(theta);
 c2 = cos(theta)*cos(theta);
 s2 = sin(theta)*sin(theta);

 a1 = a*c2 + b*sc + c*s2;
 c1 = a*s2 - b*sc + c*c2;

 th= 0:2*pi/100:2*pi;

 x1 = sqrt(mhdist/a1)*cos(th);
 y1 = sqrt(mhdist/c1)*sin(th);
 
 x = x1*cos(theta) - y1*sin(theta) + mean(xaxis);
 y = x1*sin(theta) + y1*cos(theta) + mean(yaxis);
% plot the ellipse 
 plot(x,y,'b')
end

