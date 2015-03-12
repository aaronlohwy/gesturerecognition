function soft_SVM_MultiClass()
% runs the perceptron model on a separable four 
% class dataset consisting of three dimensional data features. 

weights = [];
[D,b] = load_data();


for i = 1:4
%%% load data %%%
i
newlabels = b; % basically recodes the point with the current label to 1, the rest to -1
for j = 1:numel(newlabels)
    if newlabels(j) == i
        newlabels(j) = 1;
    else
        newlabels(j) = -1;
    end
end


%%% run perceptron from initial point %%%

% Calculate fixed steplength

% initial point, experiment
x0 = [0.5;0;0.9;130];

L = 2*norm(diag(newlabels)*D')^2;

%Run perceptron first time
lam = 10^2;        % regularization parameter 
alpha = 1/(L + 2*lam);        % step length
x = grad_descent_soft_SVM(D,newlabels,x0,alpha,lam);

% Run perceptron second time % found ideal weights by experimentation
lam = 54;
alpha = 1/(L + 2*lam);        % step length
y = grad_descent_soft_SVM(D,newlabels,x0,alpha,lam);

% Run perceptron third time
lam = 10^-2;
alpha = 1/(L + 2*lam);        % step length
z = grad_descent_soft_SVM(D,newlabels,x0,alpha,lam);

weights = [weights;y']; % change which set of weights you want to use, x,y or z

end 

% plot everything % Can't plot for 4 Features!
plot_classifiers(D',b,weights);


% label new points
[newSet] = load_testingdata();
[newlabelledSet] = createLabel(newSet,weights);
doSomethingWithNewLabelledDataPoints(newlabelledSet);
newlabelledSet
end


%% FUNCTIONS FOR BUILDING MODEL
function plot_classifiers(A,b,weights)
    x1 = weights(1,:);
    x2 = weights(2,:);
    x3 = weights(3,:);
    x4 = weights(4,:)
    plot_all(A,b,x1,x2,x3,x4);
end

%%% gradient descent function for perceptron %%%
function x = grad_descent_soft_SVM(D,b,x0,alpha,lam)
    % Initializations 
    x = x0;
    H = diag(b)*D';
    l = ones(size(D,2),1);
    iter = 1;
    max_its = 3000;
    grad = 1;

    while  norm(grad) > 10^-6 && iter < max_its
        
        % form gradient and take step
        grad = 2*lam*[0;x(2:end)] - 2*H'*max(l - H*x,0);
        x = x - alpha*grad;

        % update iteration count
        iter = iter + 1;
    end
end

%%% plots everything %%%
function plot_all(A,b,x1,x2,x3,x4)
    
    % plot points 
    ind = find(b == 1);
    scatter3(A(ind,2),A(ind,3), A(ind,4),'Linewidth',2,'Markeredgecolor','b','markerFacecolor','none');
    hold on
    ind = find(b == 2);
    scatter3(A(ind,2),A(ind,3),A(ind,4),'Linewidth',2,'Markeredgecolor','r','markerFacecolor','none');
    hold on
    ind = find(b == 3);
    scatter3(A(ind,2),A(ind,3),A(ind,4),'Linewidth',2,'Markeredgecolor','g','markerFacecolor','none');
    hold on
    ind = find(b == 4);
    scatter3(A(ind,2),A(ind,3),A(ind,4),'Linewidth',2,'Markeredgecolor','y','markerFacecolor','none');
    hold on


    % plot separators
    s =[min(A(:,2)):.01:max(A(:,2))];
    plot (s,(-x1(1)-x1(2)*s)/x1(3),'b','linewidth',2);
    hold on

    plot (s,(-x2(1)-x2(2)*s)/x2(3),'r','linewidth',2);
    hold on

    plot (s,(-x3(1)-x3(2)*s)/x3(3),'g','linewidth',2);
    hold on
    
    plot (s,(-x4(1)-x4(2)*s)/x4(3),'y','linewidth',2);
    hold on
    

    set(gcf,'color','w');
    axis([ (min(A(:,2)) - 1) (max(A(:,2)) + 1) (min(A(:,3)) - 1) (max(A(:,3)) + 1)])
    box off
    
    % graph info labels
    xlabel('a_1','Fontsize',14)
    ylabel('a_2  ','Fontsize',14)
    set(get(gca,'YLabel'),'Rotation',0)

end

%%% loads data %%%
function [A,b] = load_data()
    data = load('trainingdata4C.mat');
    data = data.data;

    A = data(:,1:end-1);
    A = A';
    b = data(:,end);
end

%% FUNCTIONS FOR TESTING NEW DATA POINTS

%% load the testing data set (no labels)
function [A] = load_testingdata()
    data = load('testingdata4C.mat');% create testing data mat with no labels [[1 a11 a12];[1 a21 a22]...]
    data = data.testingdata;
    A = data;
end

%% test a new data point
function [labelledDataSet] = createLabel(A,weights) %% A is an array of data points [1 a1 a2]
    labelledDataSet = [];
    for j = 1:size(A,1)
        d = A(j,:); % extract jth row of matrix ie. the jth data point
        max = -99999;
        label = 1;
        for i = 1:size(weights,1)
            x = weights(i,:);
            value = d*x'; %in the code it's d'*x but the rows are formatted differently so this is easier.
            if value>max
                label = i;
                max = value;
            end
        end
        labelledRow = [d,label];
        labelledDataSet = [labelledDataSet;labelledRow];
    end
end

%% do something with new data point
function doSomethingWithNewLabelledDataPoints(A) %% A is a matrix of newly labelled data points
    for j = 1:size(A,1)
        a = A(j,:);
        label = a(end);
        if label == 1
            disp('gesture 1')
        elseif label == 2
            disp('gesture 2')
        elseif label == 3
            disp('gesture 3')
        elseif label == 4
            disp('gesture 4')
        elseif label == 5
            disp('gesture 5')
        end
    end
end