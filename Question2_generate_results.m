warning('off','all')
clear 
C = 10; % # of components
mu = [1:C; 1:C]';
for i= 1:C
    Sigma(:,:,i) = (1+rand(1)/10)*eye(2)/10; 
    alpha(i,1) = 1/C; % weights of each class
end;

dataset_sizes = 10.^[2:5];

E = 100; % number of times to run the experiment
maxM = 20;

%BIC
BIC_loglikelihood2 = zeros(E, length(dataset_sizes), maxM);

parfor experiment = 1:E
     tic
     fprintf('E: %i \n',experiment)
     experiment_results = run_experiment_bic(mu, Sigma,alpha, maxM, dataset_sizes);
     BIC_loglikelihood2(experiment, :,:) = experiment_results;
     toc
end
save('BIC_loglikelihood_100experiment_2to5_11_4.mat')

%% Cross validation
K = 5;% K folds
kfold_loglikelihood_val2 = zeros(E, length(dataset_sizes), maxM, K);
kfold_loglikelihood_train2 = zeros(E, length(dataset_sizes), maxM, K);
parfor experiment = 1:E
    tic
    fprintf('E: %i \n',experiment)
    [experiment_results_val, experiment_results_training] = run_experiment_kfold(mu, Sigma,alpha, K, maxM, dataset_sizes);
    kfold_loglikelihood_val2(experiment, :, :, :) =  experiment_results_val;
    kfold_loglikelihood_train2(experiment, :, :, :) = experiment_results_training;
    toc
end
save('kfold_lkikelihood_train_04_11_different)sigmas.mat')

%% Functions for the BIC
function experiment_results = run_experiment_bic(mu, Sigma,alpha, maxM, dataset_sizes)
    experiment_results = zeros(1, length(dataset_sizes), maxM);
    gmtrue = gmdistribution(mu, Sigma,alpha);
    [d,Mtrue] = size(gmtrue.mu'); % determine dimensionality of samples and number of GMM components
    
    parfor N = 1:length(dataset_sizes)
        fprintf('N: %i \n',dataset_sizes(N))
        x = random(gmtrue,dataset_sizes(N))'; 
        experiment_results_for_each_M = BIC_get_results_at_each_M(maxM, x);
        experiment_results(1, N, :) = experiment_results_for_each_M;
    end
end

function experiment_results_for_each_M = BIC_get_results_at_each_M(maxM, x)
    experiment_results_for_each_M = zeros(1, maxM);
    neg2logLikelihood = zeros(1, maxM);
    parfor M = 1:maxM
        fprintf('M: %i \n',M)
        nParams(1,M) = (M-1) + 2*M + M*(2+nchoosek(2,2));
        options = statset('MaxIter',1000); % Specify max allowed number of iterations for EM
        gm = fitgmdist(x',M,'Replicates',2,'Options',options, 'RegularizationValue',  1e-10); 
        neg2logLikelihood(1,M) = -2*sum(log(pdf(gm,x')));
        experiment_results_for_each_M(1,M) = neg2logLikelihood(1,M) + nParams(1,M)*log(size(x,1)*size(x,2));
    end
end

%% Functions for the k fold
function [experiment_results_val, experiment_results_training] = run_experiment_kfold(mu, Sigma,alpha, K, maxM, dataset_sizes)
    experiment_results_val = zeros(1, length(dataset_sizes), maxM, K);
    experiment_results_training = zeros(1, length(dataset_sizes), maxM, K);
    gmtrue = gmdistribution(mu, Sigma,alpha);
    [d,Mtrue] = size(gmtrue.mu'); % determine dimensionality of samples and number of GMM components
    
    parfor N = 1:length(dataset_sizes)
        x = random(gmtrue,dataset_sizes(N))'; % Drexperiment_results_for_each_M = BIC_get_results_at_each_M(maxM, x);
    
        [validation_M_results, training_M_results] = get_results_at_each_M(maxM, K, x, dataset_sizes(N));
        experiment_results_val(1, N, :,:) = validation_M_results;
        experiment_results_training(1, N, :,:) = training_M_results;
        fprintf('N: %i \n',N)
    end
end

function [validation_M_results, training_M_results] = get_results_at_each_M(maxM, K, x, N)
    validation_M_results = zeros(1,maxM, K);
    training_M_results = zeros(1,maxM, K);    
    parfor M = 1:maxM
        fprintf('M: %i \n',M)

        [validation_results , training_results] = kfold(K, M, x, N);
        training_M_results(1,M, :) = training_results;
        validation_M_results(1,M, :) = validation_results;
    end
end

function [validation_results , training_results] = kfold(K, M, x, N)
    validation_results = zeros(1,K);
    training_results = zeros(1,K);
    parfor k = 1:K
        D_train = [x(:,1:floor(((k-1)/K)*N)),x(:,floor(k/K*N)+1:N)];
        D_validate = x(:,floor(((k-1)/K)*N)+1:floor(k/K*N));
        fprintf('k: %i size %i',k, N)
        options = statset('MaxIter',1000); % Specify max allowed number of iterations for EM
        gm = fitgmdist(D_train',M,'Replicates',1,'Options',options, 'RegularizationValue',  1e-10); 
        validation_results(1,k) = -2*sum(log(pdf(gm,D_validate'))) ;% + nParams(1,M)*log(nSamples);
        training_results(1,k)   = -2*sum(log(pdf(gm,D_train'))) ;%+ nParams(1,M)*log(nSamples);
    end
    fprintf('End of folds \n ')
end