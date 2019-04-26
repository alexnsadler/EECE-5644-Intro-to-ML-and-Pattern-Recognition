clc
close all

%% Dataset 2

%scatter(sample1D1(:,1),sample1D1(:,2),'o')
% hold on
% scatter(sample1D2(:,1),sample1D2(:,2),'o')

dataset2 = load('/Users/alexsadler/Downloads/gauss3.mat')
mu1 = [0; 0]; mu2 = [0; 0]; mu3 = [2; 0];
cov1 = [.5 .5; .5 1]; cov2 = [.5 -.5; -.5 1]; cov3 = [.2 0; 0 1];

dataset2 = dataset2.gauss3;

s1 = dataset2(dataset2(:,3)==0,:);
s2 = dataset2(dataset2(:,3)==1,:);
s3 = dataset2(dataset2(:,3)==2,:);

figure()
scatter(s1(:,1),s1(:,2),'o')
hold on
%scatter(s2(:,1),s2(:,2),'o')
%hold on
scatter(s3(:,1),s3(:,2),'o')
hold off

%figure()
%plot_gauss(s1,mu1,cov1,1,2)


%%

%dataset2 = load('/Users/alexsadler/Downloads/gauss2.mat')

%dataset2 = dataset2.gauss2

%s1 = dataset2(dataset2(:,3)==0,:);
%s2 = dataset2(dataset2(:,3)==1,:);

%% Initialize parameters using randomly selected samples from dataset
clc

d2sample = dataset2;
%d2sample(:,3) = d2sample(:,3)+1;
c = 3;


% initialize means to be randomly selected points
mus = cell(c,1);
initMus = initCentroids(d2sample,c);
for idx=1:c
    mus{idx} = initMus(idx,:);
end

% assign each datapoint to nearest mean
[~, index] = nearestCentroids(d2sample,initMus);

% initialize cov matrices as sample covariances of initial assignments
covs = cell(c,1);
for idx=1:c
    covs{idx} = cov(d2sample(index==idx,1:2));
end


%% Run GMM

figure()
plot_gauss(s1,mus{1},covs{1},1,2)
plot_gauss(s2,mus{2},covs{2},1,2)
plot_gauss(s3,mus{3},covs{3},1,2)


num_iter = 30;

llhood = zeros(num_iter,1);
for r=1:num_iter
    
    if r==1     % if first iteration, calculate probs with init parameters
        probs = evalGaus(d2sample,mus,covs);
    else        % else calculate probs with improved means/cov matrices
        probs = evalGaus(d2sample,mu_list,cov_list);
    end
    
    % mu_list is list of means weighted by the probability of each data
    % point belonging to each cluster (similar to k-means hard clustering
    % but now use probabilistic assignment)
    mu_list = cell(c,1);
    cov_list = cell(c,1);    
    for idx=1:c
        % multiply each datapoint by probability of it belonging to each
        % class and divide by sum of probabilities (weights)
        % weighted_mu = sum((probs(:,idx) .* d2sample(:,1:2)) ./ sum(probs(:,idx)))
        
        normed_probs = probs ./ sum(probs,2);
        weighted_probs = normed_probs ./ sum(normed_probs);
        new_mu = sum(weighted_probs(:,idx) .* d2sample(:,1:2));
        
        cov_idx = d2sample(:,1:2) - new_mu;    % remove mean
        cov_idx = cov_idx' * (cov_idx .* repmat(weighted_probs(:,idx),1,2)); % weighted cov matrix
        
        mu_list{idx} = new_mu;
        cov_list{idx} = cov_idx;
                
    end
    
    
    logLikelihood = sum(log(sum(probs,2)));
    llhood(r) = logLikelihood;

    %probs(:,1)' * d2sample(:,1:2)
    %new_probs = probs(:,1) ./ sum(probs(:,1),1);
    %size(d2sample(:,1:2));
    %size(probs(:,1)');
    
    

end

figure()
plot_gauss(s1,mu_list{1},cov_list{1},1,2)
plot_gauss(s2,mu_list{2},cov_list{2},1,2)
plot_gauss(s3,mu_list{3},cov_list{3},1,2)


figure()
plot(llhood)



%scatter(mu_list{1}(1),mu_list{1}(2),'o')
%scatter(mu_list{2}(1),mu_list{2}(2),'o')
    

%%

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
    
    % weight and normalize probabilities (using 1 / number of clusters)
    % weighted_probs = (probs * 1/c);
    % normed_probs = probs ./ sum(probs,2);
    
    % probs = normed_probs;
   
end


function centroids = initCentroids(data, k)
    
    % sample n points based on number of clusters from 0 to length(data), 
    % and then use sample to index into data
    centroids = data(randsample(length(data),k),1:2);

end


function [errors, clust_assignments] = nearestCentroids(data, centroids)
    
    k = size(centroids,1);      % get num_clusters
    m = size(data, 1);          % get number of datapoints
    dists = zeros(m,k);         % initialize distances values to 0
    
    
    for row=1:m                 % loop through each datapoint
        for clust=1:k           % loop through number of clusters
            % calculate euclidean distance between each point and each
            % centroid, then add to dists
            xdist = (centroids(clust,1) - data(row,1)).^2;
            ydist = (centroids(clust,2) - data(row,2)).^2;
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

