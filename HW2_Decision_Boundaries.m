%% EECE 5644 Homework 2


%% Set default figure size/position
set(groot,'defaultfigureposition',[400 250 900 600])


%% Define Variables
nSamples = 400;
% Create 2x1 cell array of identity covariance matrices
sigmasId = cell(2,1); sigmasId{1} = eye(2); sigmasId{2} = eye(2);
% Create two cases of prior probabilities
priors5050 = cell(2,1); priors5050{1} = 0.5; priors5050{2} = 0.5;
priors0595 = cell(2,1); priors0595{1} = 0.05; priors0595{2} = 0.95;
% Set x1 and x2 to symbolic variables (for decision boundary evaluation)
syms x1 x2


%% Graph 1.6a
mus1 = cell(2,1); mus1{1} = [0; 0]; mus1{2} = [3; 3];
sigmas1 = sigmasId;
priors1 = priors5050;

[data1, classIndex1] = generateGaussianSamples(mus1,sigmas1,nSamples,priors1);

% Plotting
figure()
gscatter(data1(:,1),data1(:,2),classIndex1,'rb','oo',5,'doleg','x1','x2')
legend('x1 given P(w1)=0.5','x2 given P(w2)=0.5')
title('(1.6a) Two Class-Conditional Multivariate Gaussian Densities')


%% Graph 1.6b
mus2 = cell(2,1); mus2{1} = [0; 0]; mus2{2} = [3; 3];
sigmas2 = cell(2,1); sigmas2{1} = [3 1; 1 0.8]; sigmas2{2} = [3 1; 1 0.8];
priors2 = priors5050;

[data2, classIndex2] = generateGaussianSamples(mus2,sigmas2,nSamples,priors2);

% Plotting
figure()
gscatter(data2(:,1),data2(:,2),classIndex2,'rb','oo',5,'doleg','x1','x2')
legend('x1 given P(w1)=0.5','x2 given P(w2)=0.5')
title('(1.6b) Two Class-Conditional Multivariate Gaussian Densities')


%% Graph 1.6c
mus3 = cell(2,1); mus3{1} = [0; 0]; mus3{2} = [2; 2];
sigmas3 = cell(2,1); sigmas3{1} = [2 0.5; 0.5 1]; sigmas3{2} = [2 -1.9; -1.9 5];
priors3 = priors5050;

[data3, classIndex3] = generateGaussianSamples(mus3,sigmas3,nSamples,priors3);

% Plotting
figure()
gscatter(data3(:,1),data3(:,2),classIndex3,'rb','oo',5,'doleg','x1','x2')
legend('x1 given P(w1)=0.5','x2 given P(w2)=0.5')
title('(1.6c) Two Class-Conditional Multivariate Gaussian Densities')


%% Graph 1.6d
mus4 = cell(2,1); mus4{1} = [0; 0]; mus4{2} = [3; 3];
sigmas4 = cell(2,1); sigmas4{1} = eye(2); sigmas4{2} = eye(2);
priors4 = priors0595;

[data4, classIndex4] = generateGaussianSamples(mus4,sigmas4,nSamples,priors4);

% Plotting
figure()
gscatter(data4(:,1),data4(:,2),classIndex4,'rb','oo',5,'doleg','x1','x2')
legend('x1 given P(w1)=0.05','x2 given P(w2)=0.95')
title('(1.6d) Two Class-Conditional Multivariate Gaussian Densities')


%% Graph 1.6e
mus5 = cell(2,1); mus5{1} = [0; 0]; mus5{2} = [3; 3];
sigmas5 = cell(2,1); sigmas5{1} = [3 1; 1 0.8]; sigmas5{2} = [3 1; 1 0.8];
priors5 = priors0595;

[data5, classIndex5] = generateGaussianSamples(mus5,sigmas5,nSamples,priors5);

% Plotting
figure()
gscatter(data5(:,1),data5(:,2),classIndex5,'rb','oo',5,'doleg','x1','x2')
legend('x1 given P(w1)=0.05','x2 given P(w2)=0.95')
title('(1.6e) Two Class-Conditional Multivariate Gaussian Densities')


%% Graph 1.6f
mus6 = cell(2,1); mus6{1} = [0; 0]; mus6{2} = [2; 2];
sigmas6 = cell(2,1); sigmas6{1} = [2 0.5; 0.5 1]; sigmas6{2} = [2 -1.9; -1.9 5];
priors6 = priors0595;

[data6, classIndex6] = generateGaussianSamples(mus6,sigmas6,nSamples,priors6);

% Plotting
figure()
gscatter(data6(:,1),data6(:,2),classIndex6,'rb','oo',5,'doleg','x1','x2')
legend('x1 given P(w1)=0.05','x2 given P(w2)=0.95')
title('(1.6f) Two Class-Conditional Multivariate Gaussian Densities')




%% Graph 1.7a 
% Data from 1.6a
[disc1, correct1] = discriminantFunc(mus1,sigmas1,priors1,data1,classIndex1);

% Plotting
figure()
gscatter(data1(:,1),data1(:,2),classIndex1,'rb','oo',5,'doleg','x1','x2')
hold on
title({'(1.7a) Two Class-Conditional Multivariate Gaussian Densities',...
       strcat(num2str(correct1),"/",num2str(nSamples),...
       " Correctly Classified Points")})
xrng = -3:7;
plot(xrng,3-xrng,'k-')
legend('x1 given P(w1)=0.5','x2 given P(w2)=0.5','Decision Boundary')
hold off

% Get decision boundary
gx11 = discrimVar(mus1,sigmas1,priors1,1);
gx12 = discrimVar(mus1,sigmas1,priors1,2);
dec_bound1 = solve(gx11 == gx12,x2);


%% Graph 1.7b (case 1) 
% Data from 1.6b
[disc2I, correct2I] = discriminantFunc(mus2,sigmasId,priors2,data2,classIndex2);

figure()
gscatter(data2(:,1),data2(:,2),classIndex2,'rb','oo',5,'doleg','x1','x2')
hold on
title({'(1.7b) Two Class-Conditional Multivariate Gaussian Densities',...
       'Case 1',...
       strcat(num2str(correct2I),"/",num2str(nSamples),...
       " Correctly Classified Points")})
xrng = -5:7;
plot(xrng,3-xrng,'k-')
legend('x1 given P(w1)=0.5','x2 given P(w2)=0.5','Decision Boundary')
hold off

% Get decision boundary
gx21I = discrimVar(mus2,sigmasId,priors2,1);
gx22I = discrimVar(mus2,sigmasId,priors2,2);
dec_bound2I = solve(gx21I == gx22I,x2);
%% Graph 1.7b (case 2)
% Data from 1.6b
[disc2II, correct2II] = discriminantFunc(mus2,sigmas2,priors2,data2,classIndex2);

figure()
gscatter(data2(:,1),data2(:,2),classIndex2,'rb','oo',5,'doleg','x1','x2')
hold on
title({'(1.7b) Two Class-Conditional Multivariate Gaussian Densities',...
       'Case 2',...
       strcat(num2str(correct2II),"/",num2str(nSamples),...
       " Correctly Classified Points")})
xrng = -6:10;
plot(xrng,xrng*.1+1.35,'k-')
legend('x1 given P(w1)=0.5','x2 given P(w2)=0.5','Decision Boundary')
hold off

% Get decision boundary
gx21II = discrimVar(mus2,sigmas2,priors2,1);
gx22II = discrimVar(mus2,sigmas2,priors2,2);
dec_bound2II = solve(gx21II == gx22II,x2);


%% Graph 1.7c (case 1)
[disc3I, correct3I] = discriminantFunc(mus3,sigmasId,priors3,data3,classIndex3);

figure()
gscatter(data3(:,1),data3(:,2),classIndex3,'rb','oo',5,'doleg','x1','x2')
hold on
title({'(1.7b) Two Class-Conditional Multivariate Gaussian Densities',...
       'Case 1',...
       strcat(num2str(correct3I),"/",num2str(nSamples),...
       " Correctly Classified Points")})
xrng = -6:7;
plot(xrng,2-xrng,'k-')
legend('x1 given P(w1)=0.5','x2 given P(w2)=0.5','Decision Boundary')
hold off

% Get decision boundary
gx31I = discrimVar(mus3,sigmasId,priors3,1);
gx32I = discrimVar(mus3,sigmasId,priors3,2);
dec_bound = solve(gx31I == gx32I,x2);
%% Graph 1.7c (case 2)
% Average two covariance matrices
covAvg3 = [sigmas3{1} + sigmas3{2}]/2;
sigmas3II = cell(2,1);
sigmas3II{1} = covAvg3; sigmas3II{2} = covAvg3;

[disc3II, correct3II] = discriminantFunc(mus3,sigmas3II,priors3,data3,classIndex3);

figure()
gscatter(data3(:,1),data3(:,2),classIndex3,'rb','oo',5,'doleg','x1','x2')
hold on
title({'(1.7b) Two Class-Conditional Multivariate Gaussian Densities',...
       'Case 2',...
       strcat(num2str(correct3II),"/",num2str(nSamples),...
       " Correctly Classified Points")})
xrng = -6:7;
plot(xrng,2.37-1.37*xrng,'k-');
legend('x1 given P(w1)=0.5','x2 given P(w2)=0.5','Decision Boundary');
hold off

% Get decision boundary
gx31II = discrimVar(mus3,sigmas3II,priors3,1);
gx32II = discrimVar(mus3,sigmas3II,priors3,2);
dec_bound3II = solve(gx31II == gx32II,x2);
%% Graph 1.7c (case 3) 
[disc3III, correct3III] = discriminantFunc(mus3,sigmas3,priors3,data3,classIndex3);

figure()
gscatter(data3(:,1),data3(:,2),classIndex3,'rb','oo',5,'doleg','x1','x2')
hold on
title({'(1.7c) Two Class-Conditional Multivariate Gaussian Densities',...
       'Case 3',...
       strcat(num2str(correct3III),"/",num2str(nSamples),...
       " Correctly Classified Points")})

xrng = -6:.01:2.08;
y1 =  0.7026.*xrng - 5.388e-8*(2.576e+14*xrng.^2 - 2.505e+15.*xrng + 4.089e+15).^(1/2) - 1.471;
y2 =  0.7026.*xrng + 5.388e-8*(2.576e+14*xrng.^2 - 2.505e+15.*xrng + 4.089e+15).^(1/2) - 1.471;
plot(xrng,y1,'k-')
plot(xrng,y2,'k-')

ylim([-5,5])

legend('x1 given P(w1)=0.5','x2 given P(w2)=0.5','Decision Boundary')
hold off

% Get decision boundary
gx31III = discrimVar(mus3,sigmas3,priors3,1);
gx32III = discrimVar(mus3,sigmas3,priors3,2);
dec_bound3III = solve(gx31III == gx32III,x2);


%% Question 1.7d
% Graph 1.7d
[disc4I, correct4I] = discriminantFunc(mus4,sigmas4,priors4,data4,classIndex4);

figure()
gscatter(data4(:,1),data4(:,2),classIndex4,'rb','oo',5,'doleg','x1','x2')
hold on
title({'(1.7d) Two Class-Conditional Multivariate Gaussian Densities',...
       strcat("Discriminant Function: ", num2str(correct4I),"/",num2str(nSamples)," Correctly Classified Points"),...
       strcat("Classification Based Solely on Priors: ", num2str(sum(classIndex4 == 2)),"/",num2str(nSamples)," Correctly Classified Points")})
xrng = -6:7;
plot(xrng,2.018-xrng,'k-')
legend('x1 given P(w1)=0.05','x2 given P(w2)=0.95','Decision Boundary')
hold off

% Get decision boundary
gx41 = discrimVar(mus4,sigmas4,priors4,1);
gx42 = discrimVar(mus4,sigmas4,priors4,2);
dec_bound4I = solve(gx41 == gx42,x2);


%% Graph 1.7e (case 1)
[disc5I, correct5I] = discriminantFunc(mus5,sigmasId,priors5,data5,classIndex5);

figure()
gscatter(data5(:,1),data5(:,2),classIndex5,'rb','oo',5,'doleg','x1','x2')
hold on
title({'(1.7e) Two Class-Conditional Multivariate Gaussian Densities',...
       'Case 1',...
       strcat("Discriminant Function: ", num2str(correct5I),"/",num2str(nSamples)," Correctly Classified Points"),...
       strcat("Classification Based Solely on Priors: ", num2str(sum(classIndex5 == 2)),"/",num2str(nSamples)," Correctly Classified Points")})
xrng = -4:8;
plot(xrng,2.018-xrng,'k-')
legend('x1 given P(w1)=0.05','x2 given P(w2)=0.95','Decision Boundary')
hold off

% Get decision boundary
gx51I = discrimVar(mus5,sigmasId,priors5,1);
gx52I = discrimVar(mus5,sigmasId,priors5,2);
dec_bound5I = solve(gx51I == gx52I,x2);
%% Graph 1.7e (case 2)
[disc5II, correct5II] = discriminantFunc(mus5,sigmas5,priors5,data5,classIndex5);

figure()
gscatter(data5(:,1),data5(:,2),classIndex5,'rb','oo',5,'doleg','x1','x2')
hold on
title({'(1.7e) Two Class-Conditional Multivariate Gaussian Densities',...
       'Case 2',...
       strcat("Discriminant Function: ", num2str(correct5II),"/",num2str(nSamples)," Correctly Classified Points"),...
       strcat("Classification Based Solely on Priors: ", num2str(sum(classIndex5 == 2)),"/",num2str(nSamples)," Correctly Classified Points")})
xrng = -6:10;
plot(xrng,xrng*.1+0.663,'k-')
legend('x1 given P(w1)=0.05','x2 given P(w2)=0.95','Decision Boundary')
hold off

% Get decision boundary
gx51II = discrimVar(mus5,sigmas5,priors5,1);
gx52II = discrimVar(mus5,sigmas5,priors5,2);
dec_bound5II = solve(gx51II == gx52II,x2);


%% Graph 1.7f (case 1)
[disc6I, correct6I] = discriminantFunc(mus6,sigmasId,priors6,data6,classIndex6);

figure()
gscatter(data6(:,1),data6(:,2),classIndex6,'rb','oo',5,'doleg','x1','x2')
hold on
title({'(1.7f) Two Class-Conditional Multivariate Gaussian Densities',...
       'Case 1',...
       strcat("Discriminant Function: ", num2str(correct6I),"/",num2str(nSamples)," Correctly Classified Points"),...
       strcat("Classification Based Solely on Priors: ", num2str(sum(classIndex6 == 2)),"/",num2str(nSamples)," Correctly Classified Points")})
xrng = -6:8;
plot(xrng,.5278-xrng,'k-')
legend('x1 given P(w1)=0.05','x2 given P(w2)=0.95','Decision Boundary')
hold off

% Get decision boundary
gx61I = discrimVar(mus6,sigmasId,priors6,1);
gx62I = discrimVar(mus6,sigmasId,priors6,2);
dec_bound6I = solve(gx61I == gx62I,x2);
%% Graph 1.7f (case 2)
% Average two covariance matrices
covAvg6 =[sigmas6{1} + sigmas6{2}]/2;
sigmas6II = cell(2,1);
sigmas6II{1} = covAvg6; sigmas6II{2} = covAvg6;

[disc6II, correct6II] = discriminantFunc(mus6,sigmas6II,priors6,data6,classIndex6);

figure()
gscatter(data6(:,1),data6(:,2),classIndex6,'rb','oo',5,'doleg','x1','x2')
hold on
title({'(1.7f) Two Class-Conditional Multivariate Gaussian Densities',...
       'Case 2',...
       strcat("Discriminant Function: ", num2str(correct6II),"/",num2str(nSamples)," Correctly Classified Points"),...
       strcat("Classification Based Solely on Priors: ", num2str(sum(classIndex6 == 2)),"/",num2str(nSamples)," Correctly Classified Points")})
xrng = -6:8;
plot(xrng,-0.6341-1.37*xrng,'k-');
legend('x1 given P(w1)=0.05','x2 given P(w2)=0.95','Decision Boundary');
hold off

% Get decision boundary
gx61II = discrimVar(mus6,sigmas6II,priors6,1);
gx62II = discrimVar(mus6,sigmas6II,priors6,2);
dec_bound6II = solve(gx61II == gx62II,x2);
%% Graph 1.7f (case 3) 
[disc6III, correct6III] = discriminantFunc(mus6,sigmas6,priors6,data6,classIndex6);

figure()
gscatter(data6(:,1),data6(:,2),classIndex6,'rb','oo',5,'doleg','x1','x2')
hold on
title({'(1.7f) Two Class-Conditional Multivariate Gaussian Densities',...
       'Case 3',...
       strcat("Discriminant Function: ", num2str(correct6III),"/",num2str(nSamples)," Correctly Classified Points"),...
       strcat("Classification Based Solely on Priors: ", num2str(sum(classIndex6 == 2)),"/",num2str(nSamples)," Correctly Classified Points")})

xrng = -6:.01:.71;
y1 =  0.7026*xrng - 5.388e-8*(2.576e+14*xrng.^2 - 2.505e+15*xrng + 1.645e+15).^(1/2) - 1.471;
y2 =  0.7026*xrng + 5.388e-8*(2.576e+14*xrng.^2 - 2.505e+15*xrng + 1.645e+15).^(1/2) - 1.471;
plot(xrng,y1,'k-')
plot(xrng,y2,'k-')

ylim([-5,5])

legend('x1 given P(w1)=0.05','x2 given P(w2)=0.95','Decision Boundary')
hold off

% Get decision boundary
gx61III = discrimVar(mus6,sigmas6,priors6,1);
gx62III = discrimVar(mus6,sigmas6,priors6,2);
dec_bound6III = solve(gx61III == gx62III,x2);


%% Code to save all open figures to desktop

% FolderPath = '/Users/alexsadler/Desktop/Figures/Figure_';
% FigList = findobj('Type', 'figure')
% 
% for iFig = 1:length(FigList)
%     FigHandle = FigList(iFig);
%     FilePath = strcat(FolderPath,num2str(iFig),'.jpg');
%     saveas(FigHandle,FilePath);
% end


%% Functions

function [data, classIndex] = generateGaussianSamples(mu,sigma,nSamples,prior)
    
    % Get number of classes based off of number of entries in mu
    muSize = size(mu);
    numClasses = muSize(1);
    
    % Create nSamples-by-1 array of uniformly distributed values
    uDist = rand(nSamples,1);
    
    for i=1:numClasses
        if i == 1
            % Calculate number of samples based off # of datapoints in
            % uDist less than first prior
            nSamplesi = sum(uDist < prior{1});
            % Take random multivariate normal sample
            randSamplei = mvnrnd(mu{i},sigma{i},nSamplesi);
            % Create class index
            classIndex1 = i*ones(nSamplesi,1);
            
            data = randSamplei;
            classIndex = classIndex1;
        else
            % Calculate number of samples based off # of datapoints in
            % uDist between i-1 cumulative sum of priors and ith prior
            nSamplesi = sum(uDist < sum([prior{1:i}]) & uDist > sum([prior{1:i-1}]));
             % Take random multivariate normal sample
            randSamplei = mvnrnd(mu{i},sigma{i},nSamplesi);
            % Create class index
            classIndexi = i*ones(nSamplesi,1);
            
            % Vertically concatenate data and classIndex
            data = [data; randSamplei];
            classIndex = [classIndex; classIndexi];
        end
    
    end
    
end


function [discFunc, numCorrect] = discriminantFunc(mu,sigma,prior,data,classIndex)
    
    % Get number of classes based off of number of entries in mu
    muSize = size(mu);
    numClasses = muSize(1);
    
    for i=1:numClasses
        
        % Define components of discriminant function for arb sigma
        Wi = -0.5 * inv(sigma{i});
        wi = inv(sigma{i}) * mu{i};
        wi0 = -0.5 * transpose(mu{i}) * inv(sigma{i}) * mu{i} ...
            - 0.5 * log(det(sigma{i})) + log(prior{i});
        
        % Define discriminant function handle based off of column 1 and 2 
        % from data
        discFunHandle = @(x1,x2) transpose([x1;x2]) * Wi * [x1;x2] ...
            + transpose(wi) * [x1;x2] + wi0;
        
        % Evaluate each data point (col 1 = x1, col 2 = x2) for the ith
        % class's descriminant function
        if i == 1
            discEval1 = arrayfun(discFunHandle,data(:,1),data(:,2));
        else
            discEvali = arrayfun(discFunHandle,data(:,1),data(:,2));
            discFunc = [discEval1 discEvali];
        end
        
    end
    
    % Calculate number of correct values by comparing column index of max
    % value for each row with classIndex
    [~,c] = max(discFunc');
    % Get sum of number of correctly identified values 
    numCorrect = sum(c' == classIndex);
    
end


function discFuncVar = discrimVar(mu,sigma,prior,i)
    
    % Define components of discriminant function for arb sigma
    Wi = -0.5 * inv(sigma{i});
    wi = inv(sigma{i}) * mu{i};
    wi0 = -0.5 * transpose(mu{i}) * inv(sigma{i}) * mu{i} ...
        - 0.5 * log(det(sigma{i})) + vpa(log(prior{i}));
    
    syms f(x1,x2)
    f(x1,x2) = transpose([x1;x2]) * Wi * [x1;x2] ...
            + transpose(wi) * [x1;x2] + wi0;
    
    % Return discriminant function in variable form
    discFuncVar = f(x1,x2);
    
end
