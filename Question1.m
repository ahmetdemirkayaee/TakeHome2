close all
clear

%Mean and covariance values
n = 2;
m_01 = [5;0];
C_01 = [4,0;0,2];
m_02 = [0;4];
C_02 = [1,0;0,3];
m_1 = [3;2];
C_1 = [2,0;0,2];
C(:,:,1) = C_01;
C(:,:,2) = C_02;
C(:,:,3) = C_1;
m(:,:,1) = m_01;
m(:,:,2) = m_02;
m(:,:,3) = m_1;

w1 = 0.5;
w2 = 0.5;

class_priors = [0.6 0.4];

D_sizes = [100,1000,10000,20000];

D_100_train = generate_dataset(D_sizes(1), class_priors, n, m, C); % 2 for data and third for the label
D_1000_train = generate_dataset(D_sizes(2), class_priors,n, m, C ); % 2 for data and third for the label
D_10000_train = generate_dataset(D_sizes(3), class_priors, n, m, C ); % 2 for data and third for the label
D_20000_validate = generate_dataset(D_sizes(4), class_priors, n, m, C); % 2 for data and third for the label

%=========Visualize Data============%
figure(1);
scatter(D_100_train(D_100_train(:,n+1)==0,1), D_100_train(D_100_train(:,n+1)==0,2), 'r^')
hold on
scatter(D_100_train(D_100_train(:,n+1)==1,1), D_100_train(D_100_train(:,n+1)==1,2), 'bo')
xlabel('X_{1}')
ylabel('X_{2}')
grid on
title('Visualization of the data D^{100}_{train}')
legend('Class 0', 'Class 1')
saveas(gcf,'d_100.png')

figure(2);
scatter(D_1000_train(D_1000_train(:,n+1)==0,1), D_1000_train(D_1000_train(:,n+1)==0,2), 'r^')
hold on
scatter(D_1000_train(D_1000_train(:,n+1)==1,1), D_1000_train(D_1000_train(:,n+1)==1,2), 'bo')
xlabel('X_{1}')
ylabel('X_{2}')
grid on
legend('Class 0', 'Class 1')
title('Visualization of the data D^{1000}_{train}')
saveas(gcf,'d_1000.png')


figure(3);
scatter(D_10000_train(D_10000_train(:,n+1)==0,1), D_10000_train(D_10000_train(:,n+1)==0,2), 'r^')
hold on
scatter(D_10000_train(D_10000_train(:,n+1)==1,1), D_10000_train(D_10000_train(:,n+1)==1,2), 'bo')
xlabel('X_{1}')
ylabel('X_{2}')
grid on
legend('Class 0', 'Class 1')
title('Visualization of the data D^{10000}_{train}')
saveas(gcf,'d_10000.png')

figure(4);
scatter(D_20000_validate(D_20000_validate(:,n+1)==0,1), D_20000_validate(D_20000_validate(:,n+1)==0,2), 'r^')
hold on
scatter(D_20000_validate(D_20000_validate(:,n+1)==1,1), D_20000_validate(D_20000_validate(:,n+1)==1,2), 'bo')
xlabel('X_{1}')
ylabel('X_{2}')
grid on
legend('Class 0', 'Class 1')
title('Visualization of the data D^{20K}_{validate}')
saveas(gcf,'d_20000.png')


%% Question 1 - Part 1
N = 20000;
data = load(strcat('question1_data_', string(N), '.mat'));
data = data.D;

class_labels = data(:,3);
class_counts = [N - sum(class_labels), sum(class_labels)];

discriminantScore = log(evalGaussian(data(:,1:2).',m(:,:,3),C(:,:,3)))-log(w1*evalGaussian(data(:,1:2).',m(:,:,1),C(:,:,1))+w1*evalGaussian(data(:,1:2).',m(:,:,2),C(:,:,2)));
lambda = [0 1;1 0];% 0-1 loss
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2))*class_priors(1)/class_priors(2);
decision = (discriminantScore >= log(gamma));
disp('Theoretical gamma: ');
gamma
sum(class_labels==decision)
ind00 = find(decision==0 & class_labels==0); p00 = length(ind00)/class_counts(1); % probability of true negative
ind10 = find(decision==1 & class_labels==0); p10 = length(ind10)/class_counts(1); % probability of false positive
ind01 = find(decision==0 & class_labels==1); p01 = length(ind01)/class_counts(2); % probability of false negative
ind11 = find(decision==1 & class_labels==1); p11 = length(ind11)/class_counts(2); % probability of true positive
if norm(lambda-[0,1;1,0])<eps % Using 0-1 loss indicates intent to minimize P(error)
    Perror_MAP = [p10,p01]*class_counts'/N, % probability of error, empirically estimated
end
theoretical_probability_of_false_positive = p10
theoretical_probability_of_true_positive  = p11


% In this part, code segments from Prof. Deniz Erdogmus' google drive
% account are used.
[sortedScores,ind] = sort(discriminantScore,'ascend');
thresholds = [min(sortedScores)-eps;(sortedScores(1:end-1)+sortedScores(2:end))/2; max(sortedScores)+eps];

probability_of_true_positive = zeros(1,length(thresholds));
probability_of_false_positive = zeros(1,length(thresholds));
probability_of_false_negative = zeros(1,length(thresholds));
probability_of_true_negative = zeros(1,length(thresholds));
probability_of_error = zeros(1,length(thresholds));

for i = 1:length(thresholds)
    gamma = thresholds(i);
    decision = (discriminantScore >= gamma);
    ind00 = find(decision==0 & class_labels==0); 
    p00 = length(ind00)/class_counts(1); % probability of true negative
    ind10 = find(decision==1 & class_labels==0); 
    p10 = length(ind10)/class_counts(1); % probability of false positive
    ind01 = find(decision==0 & class_labels==1); 
    p01 = length(ind01)/class_counts(2); % probability of false negative
    ind11 = find(decision==1 & class_labels==1); 
    p11 = length(ind11)/class_counts(2); % probability of true positive    
    probability_of_true_positive(1,i) = p11;
    probability_of_false_positive(1,i) = p10;
    probability_of_false_negative(1,i) = p01;
    probability_of_true_negative(1,i) = p00;
    probability_of_error(1,i) = [p10,p01]*class_counts'/N;
end

%Find the minimum probability of error
[min_value,index] = min(probability_of_error);

%Plot the error curve and 
figure(7);
plot(thresholds,probability_of_error);
hold on
scatter(thresholds(index),min_value,'ro');
title('Probability of Error vs Threshold')
xlabel('Threshold')
ylabel('Probability of Error')
disp('Experimental threshold:')
exp(thresholds(index))
disp('Experimental min error:')
min_value
disp('Experimental true positive rate:')
probability_of_true_positive(index)
disp('Experimental false positive rate:')
probability_of_false_positive(index)
saveas(gcf,'q1p1_errors.png')

figure(8);
grid on
plot(probability_of_false_positive,probability_of_true_positive,'bo','MarkerSize',2);
hold on
scatter(probability_of_false_positive(index),probability_of_true_positive(index),'r*');
hold on
scatter(theoretical_probability_of_false_positive,theoretical_probability_of_true_positive,'g+');
xlabel('False Positive Rate');
ylabel('True Positive Rate');
title('Receiver operating characteristic curve')
legend('ROC Curve','Experimental', 'Theoretical')
saveas(gcf,'q1p1_roc.png')

%% end of question 1 part 1

%%

N = 20000; % load validation data
data_validation = load(strcat('question1_data_', string(N), '.mat'));
data_validation = data_validation.D;
class_labels_validation = data_validation(:,3);

for N = [100, 1000, 10000]
    N
    disp('Part 2')
    data = load(strcat('question1_data_', string(N), '.mat'));
    data = data.D;
    class_labels = data(:,3);

    % Generate samples from a 3-component GMM
    alpha_true = [class_priors(1)*w1,class_priors(1)*w2,class_priors(2)];
    mu_true = m;

    options = statset('MaxIter',1000); % Specify max allowed number of iterations for EM
    gm_class_0 = fitgmdist(data(class_labels==0, 1:2), 2,'Replicates', 20,'Options',options, 'RegularizationValue',  1e-10); 
    gm_class_1 = fitgmdist(data(class_labels==1, 1:2), 1,'Replicates', 20,'Options',options, 'RegularizationValue',  1e-10); 

    discriminantScore = -log(pdf(gm_class_0,data_validation(:,1:2))) + (log(pdf(gm_class_1,data_validation(:,1:2))));
    lambda = [0 1;1 0];% 0-1 loss
    gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2))*class_priors(1)/class_priors(2);
    decision = (discriminantScore >= log(gamma));

    gamma
    sum(class_labels_validation==decision)
    ind00 = find(decision==0 & class_labels_validation==0); p00 = length(ind00)/class_counts(1); % probability of true negative
    ind10 = find(decision==1 & class_labels_validation==0); p10 = length(ind10)/class_counts(1); % probability of false positive
    ind01 = find(decision==0 & class_labels_validation==1); p01 = length(ind01)/class_counts(2); % probability of false negative
    ind11 = find(decision==1 & class_labels_validation==1); p11 = length(ind11)/class_counts(2); % probability of true positive
    if norm(lambda-[0,1;1,0])<eps   % Using 0-1 loss indicates intent to minimize P(error)
        Perror_MAP = [p10,p01]*class_counts'/20000, % probability of error, empirically estimated
    end
    theoretical_probability_of_false_positive = p10
    theoretical_probability_of_true_positive  = p11
    


    % Part 2
    % In this part, code segments from Prof. Deniz Erdogmus' google drive
    % account are used.
    [sortedScores,ind] = sort(discriminantScore,'ascend');
    thresholds = [min(sortedScores)-eps;(sortedScores(1:end-1)+sortedScores(2:end))/2; max(sortedScores)+eps];
    
    probability_of_true_positive = zeros(1,length(thresholds));
    probability_of_false_positive = zeros(1,length(thresholds));
    probability_of_false_negative = zeros(1,length(thresholds));
    probability_of_true_negative = zeros(1,length(thresholds));
    probability_of_error = zeros(1,length(thresholds));
    
    for i = 1:length(thresholds)
        gamma = thresholds(i);
        decision = (discriminantScore >= gamma);
        ind00 = find(decision==0 & class_labels_validation==0); 
        p00 = length(ind00)/class_counts(1); % probability of true negative
        ind10 = find(decision==1 & class_labels_validation==0); 
        p10 = length(ind10)/class_counts(1); % probability of false positive
        ind01 = find(decision==0 & class_labels_validation==1); 
        p01 = length(ind01)/class_counts(2); % probability of false negative
        ind11 = find(decision==1 & class_labels_validation==1); 
        p11 = length(ind11)/class_counts(2); % probability of true positive    
        probability_of_true_positive(1,i) = p11;
        probability_of_false_positive(1,i) = p10;
        probability_of_false_negative(1,i) = p01;
        probability_of_true_negative(1,i) = p00;
        probability_of_error(1,i) = [p10,p01]*class_counts'/20000;
    end

%     Find the minimum probability of error
    [min_value,index] = min(probability_of_error);
    
    %Plot the error curve and 
    figure();
    plot(thresholds,probability_of_error);
    hold on
    scatter(thresholds(index),min_value,'ro');
    title(strcat('Probability of Error vs Threshold, size: ',string(N)))
    xlabel('Threshold')
    ylabel('Probability of Error')
    disp('Experimental threshold:')
    exp(thresholds(index))
    disp('Experimental min error:')
    min_value
    disp('Experimental true positive rate:')
    probability_of_true_positive(index)
    disp('Experimental false positive rate:')
    probability_of_false_positive(index)
    saveas(gcf,strcat('errors_part1b_',string(N),'.png'))
    
    
    figure;
    grid on
    plot(probability_of_false_positive,probability_of_true_positive,'bo','MarkerSize',2);
    hold on
    scatter(probability_of_false_positive(index),probability_of_true_positive(index),'r*');
    hold on
    scatter(theoretical_probability_of_false_positive,theoretical_probability_of_true_positive,'g+');
    xlabel('False Positive Rate');
    ylabel('True Positive Rate');
    title(strcat('Receiver operating characteristic curve, size: ',string(N)))
    legend('ROC Curve','Experimental', 'Theoretical')
    saveas(gcf,strcat('roc_part1b_',string(N),'.png'));

end


% Question 1 Part 3
N = 20000; % load validation data
data_validation = load(strcat('question1_data_', string(N), '.mat'));
data_validation = data_validation.D;
x_validation = data_validation(:,1:2)';
class_labels_validation = double(data_validation(:,3));

%In this part, I used code segments from the code 
%provided by Prof. Deniz Erdogmus

for N = [100, 1000, 10000]
    N
    data = load(strcat('question1_data_', string(N), '.mat'));
    data = data.D;
    x = data(:,1:2)';
    class_labels = double(data(:,3));
    class_counts = [N - sum(class_labels), sum(class_labels)];%Found using training data

    % linear parameters initial theta 
    x_L = [ones(N,1) x']; 
    initial_theta_L=zeros(n+1,1); 
    
    % quadratic parameters initial theta 
    x_Q = [ones(N,1) x(1 ,:)' x(2,:)' (x(1,:).^2)' (x(1,:).* x(2,:))' (x (2 ,:) .^2)']; 
    initial_theta_Q=zeros(6,1); 
     
    options = optimset('MaxFunEvals',1000);
    [theta_L,cost_L] = fminsearch(@(t)(cost_func(t,x_L,class_labels,N)),initial_theta_L,options);
    [theta_Q,cost_Q] = fminsearch(@(t)(cost_func(t,x_Q,class_labels,N)),initial_theta_Q,options);
    decisions_linear = [ones(20000, 1) x_validation']*theta_L > 0;
    min_error_linear = find_min_error(decisions_linear, class_labels_validation, class_counts, 20000);
    fprintf('Min error for linear model trained using %i samples is %d \n', N, min_error_linear*100);
    
    decisions_quadratic = [ones(20000, 1) x_validation(1,:)' x_validation(2,:)' (x_validation(1,:).^2)' (x_validation(1,:).*x_validation(2,:))' (x_validation(2,:).^2)']*theta_Q > 0;
    min_error_quadratic = find_min_error(decisions_quadratic, class_labels_validation, class_counts, 20000);
    fprintf('Min error for quadratic model trained using %i samples is %d \n', N, min_error_quadratic *100);
end

function D =  generate_dataset(N, class_priors, n , m, C)
    if ~isfile(strcat('question1_data_', string(N), '.mat'))
        D = zeros(N, n+1);
        labels = rand(1,N)> class_priors(1);
        class_counts = [N-sum(labels) sum(labels)];
        D(:,n+1) = labels;
        class_01 = mvnrnd(m(:,:,1), C(:,:,1), floor(class_counts(1)/2));% since w is deterministic I multiplied with 1/2 to find the number of elements for that case
        class_02 = mvnrnd(m(:,:,2), C(:,:,2), ceil(class_counts(1)/2));
        class_1 = mvnrnd(m(:,:,3), C(:,:,3), class_counts(2));
        class_0 = [class_01;class_02];
        D(labels == 0, 1:n) = class_0(randperm(class_counts(1)),:);
        D(labels == 1, 1:n) = class_1;
        save(strcat('question1_data_', string(N), '.mat'), 'D')
    else
        D = load(strcat('question1_data_', string(N), '.mat'));
        D = D.D;
    end
end
%Provided by Prof. Deniz Erdogmus
function cost = cost_func(theta, x, label, N) 
    % Function is provided by Prof. Deniz Erdogmus
    % Cost function to be minimized to get best fitting parameters 
    h = 1 ./ (1 + exp(-x*theta ) ) ; % Sigmoid function 
    cost = (-1/N)*((sum( label' * log(h)))+(sum((1-label )' * log(1-h))));
end

function probability_of_error = find_min_error(decision, class_labels_validation, class_counts, N)
    ind10 = find(decision==1 & class_labels_validation==0); 
    p10 = length(ind10)/class_counts(1); % probability of false positive
    ind01 = find(decision==0 & class_labels_validation==1); 
    p01 = length(ind01)/class_counts(2); % probability of false negative
    probability_of_error = [p10,p01]*class_counts'/N;
end
function g = evalGaussian(x,mu,Sigma)
    %Function implemented by Prof. Deniz Erdogmus
    % Evaluates the Gaussian pdf N(mu,Sigma) at each column of X
    [n,N] = size(x);
    C = ((2*pi)^n * det(Sigma))^(-1/2);
    E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
    g = C*exp(E);
    g = g';
end