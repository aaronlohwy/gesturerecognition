function soft_SVM_MultiClass()
% perceptron_demo_wrapper - runs the perceptron model on a separable two 
% class dataset consisting of two dimensional data features. 
% The perceptron is run 3 times with 3 initial points to show the 
% recovery of different separating boundaries.  All points and recovered
% as well as boundaries are then visualized.

weights = [];
[D,b] = load_data();

for i = 1:5
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


%%% run perceptron for 3 initial points %%%

% Calculate fixed steplength - via Lipschitz constant (see Chap 9 for 
% explanation) - for use in all three runs

x0 = [-5;-5;-5];    % initial point, experiment
L = 2*norm(diag(newlabels)*D')^2;

%Run perceptron first time
lam = 10^2;        % regularization parameter 
alpha = 1/(L + 2*lam);        % step length
x = grad_descent_soft_SVM(D,newlabels,x0,alpha,lam);

% Run perceptron second time
lam = 10;
alpha = 1/(L + 2*lam);        % step length
y = grad_descent_soft_SVM(D,newlabels,x0,alpha,lam);

% Run perceptron third time
lam = 10^-2;
alpha = 1/(L + 2*lam);        % step length
z = grad_descent_soft_SVM(D,newlabels,x0,alpha,lam);

weights = [weights;y']; % change which set of weights you want to use, x,y or z

end 

% plot everything
plot_classifiers(D',b,weights);

end

function plot_classifiers(A,b,weights)
    x1 = weights(1,:)
    x2 = weights(2,:)
    x3 = weights(3,:)
    x4 = weights(4,:)
    x5 = weights(5,:)
    plot_all(A,b,x1,x2,x3,x4,x5);
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
function plot_all(A,b,x1,x2,x3,x4,x5)
    
    % plot points 
    ind = find(b == 1);
    scatter(A(ind,2),A(ind,3),'Linewidth',2,'Markeredgecolor','b','markerFacecolor','none');
    hold on
    ind = find(b == 2);
    scatter(A(ind,2),A(ind,3),'Linewidth',2,'Markeredgecolor','r','markerFacecolor','none');
    hold on
    ind = find(b == 3);
    scatter(A(ind,2),A(ind,3),'Linewidth',2,'Markeredgecolor','g','markerFacecolor','none');
    hold on
    ind = find(b == 4);
    scatter(A(ind,2),A(ind,3),'Linewidth',2,'Markeredgecolor','y','markerFacecolor','none');
    hold on
    ind = find(b == 5);
    scatter(A(ind,2),A(ind,3),'Linewidth',2,'Markeredgecolor','m','markerFacecolor','none');
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
    
    plot (s,(-x5(1)-x5(2)*s)/x5(3),'m','linewidth',2);
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
    data = load('trainingdata5.mat');
    data = data.data;
    %if i == 1
    A = data(:,1:3);
    A = A';
    b = data(:,4);
    %else 
    %  A = data(:,(5 + (4 * (i-2))):(7 + (4 * (i-2))));
    %  A = A';
    %  b = data(:,(8 + (4 * (i-2))));
    %end
 
end




%% FUNCTIONS FOR TESTING NEW DATA POINTS, NOT TESTED YET.
function [A] = load_testingdata()
    data = load('testingdata.mat'); % create testing data mat with no labels [[1 a11 a12];[1 a21 a22]...]
    data = data.data;
    A = data
end

%%test a new data point
function [labelledDataSet] = createlabel(A) %% A is an array of data points [1 a1 a2]
    labelledDataSet = []
    for j = 1:size(A,1)
        d = A(j,:) % extract jth row of matrix ie. the jth data point
        max = -99999;
        label = 1;
        for i = 1:size(weights,1)
            x = weights(i,:)
            value = d'*x
            if value>max
                label = i
                max = value
            end
        end
        labelledRow = [d,label]
        labelledDataSet = [labelledDataSet;labelledRow]
    end
end

%% do something with new data point
function doSomethingWithNewLabelledDataPoint(a) %% a is a newly labelled data point
    label = a(end)
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