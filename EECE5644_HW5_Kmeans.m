
clc
close all

%% Dataset 1
max_iterations = 10;            % number of iterations for k-means
r_reps = 100;                     % number of restarts
k = 2;                          % number of clusters

dataset1 = load('/Users/alexsadler/Downloads/gauss2.mat');
dataset1 = dataset1.gauss2;

% split dataset based on class for scatter plot
sample1D1 = dataset1(dataset1(:,3)==0,:);
sample2D1 = dataset1(dataset1(:,3)==1,:);
%% Dataset 1 K-Means
figure()
scatter(sample1D1(:,1), sample1D1(:,2), 'o')
hold on
scatter(sample2D1(:,1), sample2D1(:,2), 'o')
hold on

% plot centroids with lowest SSE over specified number of runs
centroids = bestKmeans(dataset1,k,max_iterations,r_reps);
scatter(centroids(:,1), centroids(:,2),300,'d','k','filled')
title('K-means Clustering - Dataset 1')
xlabel('x')
ylabel('y')
hold off

% plot SSE vs number of iterations
[~, errors] = kmeans(dataset1,k,max_iterations);
figure()
plot(errors)
title({'K-Means Sum-Squared-Error by Iteration','Dataset 1 (2 Clusters)'})
xlabel('Number of Iterations')
ylabel('Sum-Squared-Error (Divided by n)')



%% Dataset 2
max_iterations = 10;             % number of iterations for k-means
r_reps = 100;                     % number of restarts
k = 3;                           % number of clusters

dataset2 = load('/Users/alexsadler/Downloads/gauss3.mat');
dataset2 = dataset2.gauss3;

% split dataset based on class for scatter plot
sample1D2 = dataset2(dataset2(:,3)==0,:);
sample2D2 = dataset2(dataset2(:,3)==1,:);
sample3D2 = dataset2(dataset2(:,3)==2,:);
%% Dataset 2 K-Means
figure()
scatter(sample1D2(:,1), sample1D2(:,2), 'o')
hold on
scatter(sample2D2(:,1), sample2D2(:,2), 'o')
hold on
scatter(sample3D2(:,1), sample3D2(:,2), 'o')

% plot centroids with lowest SSE over specified number of runs
centroids = bestKmeans(dataset2,k,max_iterations,r_reps);
scatter(centroids(:,1), centroids(:,2),300,'d','k','filled')
title('K-means Clustering - Dataset 2')
xlabel('x')
ylabel('y')
hold off

% plot SSE vs number of iterations
[~, errors] = kmeans(dataset2,k,max_iterations);
figure()
plot(errors)
title({'K-Means Sum-Squared-Error by Iteration','Dataset 2 (3 Clusters)'})
xlabel('Number of Iterations')
ylabel('Sum-Squared-Error (Divided by n)')



%% Dataset 3
max_iterations = 10;             % number of iterations for k-means
r_reps = 100;                     % number of restarts
k = 3;                           % number of clusters

dataset3 = load('/Users/alexsadler/Downloads/iris.mat');
dataset3 = dataset3.iris;

% split dataset based on class for scatter plot
sample1D3 = dataset3(dataset3(:,5)==1,1:2);
sample2D3 = dataset3(dataset3(:,5)==2,1:2);
sample3D3 = dataset3(dataset3(:,5)==3,1:2);
%% Dataset 3 K-Means
figure()
scatter(sample1D3(:,1), sample1D3(:,2), 'o')
hold on
scatter(sample2D3(:,1), sample2D3(:,2), 'o')
hold on
scatter(sample3D3(:,1), sample3D3(:,2), 'o')

% plot centroids with lowest SSE over specified number of runs
centroids = bestKmeans(dataset3,k,max_iterations,r_reps);
scatter(centroids(:,1), centroids(:,2),300,'d','k','filled')
title('K-means Clustering - Dataset 3')
xlabel('x')
ylabel('y')
hold off

% plot SSE vs number of iterations
[~, errors] = kmeans(dataset3,k,max_iterations);
figure()
plot(errors)
title({'K-Means Sum-Squared-Error by Iteration','Dataset 3 (3 Clusters)'})
xlabel('Number of Iterations')
ylabel('Sum-Squared-Error (Divided by n)')



%% Dataset 4
max_iterations = 10;               % number of iterations for k-means
r_reps = 1000;                     % number of restarts

dataset4 = load('/Users/alexsadler/Downloads/a3geyser.mat');
dataset4 = dataset4.a3geyser';

%% Dataset 4 K-Means
for k=1:6   % implement k-means on geyser data for 1-6 clusters
    %figure()
    %scatter(dataset4(:,1), dataset4(:,2), 'o')
    %hold on

    % plot centroids with lowest SSE over specified number of runs
    [centroids, assignments] = bestKmeans(dataset4,k,max_iterations,r_reps);
    assignments
    figure()
    gscatter(dataset4(:,1), dataset4(:,2), assignments)
    legend('off')
    hold on
    scatter(centroids(:,1), centroids(:,2),300,'d','k','filled')
    title('K-means Clustering - Dataset 4')
    xlabel('x')
    ylabel('y')
    hold off

    % plot SSE vs number of iterations
    %[~, errors] = kmeans(dataset4,k,max_iterations);
    %figure()
    %plot(errors)
    %title({'K-Means Sum-Squared-Error by Iteration',strcat('Dataset 4 (', num2str(k), ' Clusters)')})
    %xlabel('Number of Iterations')
    %ylabel('Sum-Squared-Error (Divided by n)')
    
end


%% K-Means Clustering Functions

function centroids = initCentroids(data, k)
    
    % sample n points based on number of clusters from 0 to length(data), 
    % and then use sample to index into data
    centroids = data(randsample(length(data),k),1:2);

end


function [errors, clust_assignments] = nearestCentroids(data, centroids)
    
    k = size(centroids,1);   % get num_clusters
    m = size(data, 1);       % get number of datapoints
    dists = zeros(m,k);      % initialize distances values to 0
    
    
    for row=1:m              % loop through each datapoint
        for clust=1:k        % loop through number of clusters
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


function new_centroids = recomputeCentroids(data, k, indices)
    % calculate the average of all of the points belonging to each centroid
    % and use the k averages to set new centroid locations
    
    m = size(data, 1);
    new_centroids = zeros(k,2);
    
    for clust=1:k
        mean_clust = mean(data(indices==clust,1:2));
        new_centroids(clust,:) = mean_clust;        
        
    end

end


function [centroids, error_list] = kmeans(data, k, num_iterations)
    % implementation of k-means algorithm with previously created functions

    m = size(data, 1);
    error_list = zeros(num_iterations,1);   % errors calculated for HW SSE
    
    centroids = initCentroids(data,k);      % get initial, random centroids
    for i=1:num_iterations
        [error_sum, indices] = nearestCentroids(data, centroids);
        centroids = recomputeCentroids(data,k,indices);
        error_list(i) = error_sum/m;
    end
    
end


function [centroids, assignments] = bestKmeans(data, k, num_iterations, runs)
    % similar implentation to above but with additional input runs to run
    % k-means a given number of times with different initial, random points
    % selected. Returns the centroid locations with the lowest SSE for all
    % runs.

    error_list = zeros(runs,1);
    centroids_list = cell(runs,1);
    indices_list = cell(runs,1);
    
    for r=1:runs
        centroids = initCentroids(data,k);
        for i=1:num_iterations
            [error_sum, indices] = nearestCentroids(data, centroids);
            centroids = recomputeCentroids(data,k,indices);
            
        end
        error_list(r) = error_sum;
        centroids_list{r} = centroids;
        indices_list{r} = indices;

    end
    
    [~, idx] = min(error_list);
    centroids = centroids_list{idx};
    assignments = indices_list{idx};

end  