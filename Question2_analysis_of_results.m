close all
matrix = load('BIC_loglikelihood_100experiment_2to5_11_4.mat');
BIC_loglikelihood = matrix.BIC_loglikelihood2;

datasizes = 2:5;
predictions = zeros(size(BIC_loglikelihood, 1), length(datasizes));

disp('BIC')
for data = 1:length(datasizes)
    figure
    for experiment = 1:size(BIC_loglikelihood, 1)
        temp = BIC_loglikelihood(experiment, data,: );
        temp = reshape(temp,[],1);
        [M,I] = min(temp);
        temp = temp/(10^datasizes(data));
        plot(temp), 
        xlabel('Number of Gaussian Components in GMM'),
        ylabel('BIC'),
        title(strcat('Dataset size: 10^',string(datasizes(data))))
        predictions(experiment,data) = I;
        hold on
    end
    saveas(gcf,strcat('bic_', string(data), 'results.png'));
end

figure
for data = 1:length(datasizes)
    fprintf('BIC \n')
    fprintf('Minimum estimate at dataset sample count %i is %i \n', 10^datasizes(data), min(predictions(:,data)));
    fprintf('Median estimate at dataset sample count  %i is %i \n', 10^datasizes(data), median(predictions(:,data)));
    fprintf('Mean estimate at dataset sample count    %i is %i \n', 10^datasizes(data), mean(predictions(:,data)));
    fprintf('Maximum estimate at dataset sample count %i is %i \n', 10^datasizes(data), max(predictions(:,data)));

    scatter(data+1, min(predictions(:,data)), 'blue');
    hold on
    scatter(data+1, max(predictions(:,data)), 'green');
    hold on
    scatter(data+1, median(predictions(:,data)), 'black');
    hold on
    scatter(data+1, 10, 'red');
    hold on
    xlabel('log[dataset sample count]'),
    ylabel('Estimate of M using BIC'),
end
legend('Min', 'Max', 'Median', 'True M')
saveas(gcf,strcat('bic_minmaxmedian_', string(data), 'results.png'));


for data = 1:length(datasizes)
    figure
    histogram(predictions(:, data, :))
    title(strcat('Prediction of # of Gaussian Components in GMM using BIC , Set size 10^', string(data+1)))
    xlabel('Number of Gaussian Components in GMM')
    ylabel('frequency')
    saveas(gcf,strcat('Prediction of # of Gaussian Components in GMM using BIC , Set size 10^', strcat(string(data+1),'.png')));
end

matrix = load('kfold_lkikelihood_train_04_11_different)sigmas_100experiments.mat');
loglikelihood = matrix.kfold_loglikelihood_val2;

datasizes = 2:5;
predictions = zeros(size(loglikelihood, 1), length(datasizes));
for data = 1:length(datasizes)
    figure
    for experiment = 1:size(loglikelihood, 1)
        temp = loglikelihood(experiment, data,: ,:);
        temp = mean(temp,4);%take mean of k folds;
        size(temp);
        temp = reshape(temp,[],1);
        temp = temp/(10^datasizes(data));
        [M,I] = min(temp);
        plot(temp), 
        xlabel('Number of Gaussian Components in GMM'),
        ylabel('-Loglikelihood averaged over k-folds'),
        title(strcat('Dataset size: 10^',string(datasizes(data))))
        hold on
        predictions(experiment,data) = I;
    end
	fprintf('K-FOLD \n')
    fprintf('Minimum estimate at dataset sample count %i is %i \n', 10^datasizes(data), min(predictions(:,data)));
    fprintf('Median estimate at dataset sample count  %i is %i \n', 10^datasizes(data), median(predictions(:,data)));
    fprintf('Mean estimate at dataset sample count    %i is %i \n', 10^datasizes(data), mean(predictions(:,data)));
    fprintf('Maximum estimate at dataset sample count %i is %i \n', 10^datasizes(data), max(predictions(:,data)));

    saveas(gcf,strcat('kfold_', string(data), 'results.png'));
end


figure
for data = 1:length(datasizes)
    fprintf('kfold \n')
    fprintf('Minimum estimate at dataset sample count %i is %i \n', 10^datasizes(data), min(predictions(:,data)));
    fprintf('Median estimate at dataset sample count  %i is %i \n', 10^datasizes(data), median(predictions(:,data)));
    fprintf('Mean estimate at dataset sample count    %i is %i \n', 10^datasizes(data), mean(predictions(:,data)));
    fprintf('Maximum estimate at dataset sample count %i is %i \n', 10^datasizes(data), max(predictions(:,data)));

    scatter(data+1, min(predictions(:,data)), 'blue');
    hold on
    scatter(data+1, max(predictions(:,data)), 'green');
    hold on
    scatter(data+1, median(predictions(:,data)), 'black');
    hold on
    scatter(data+1, 10, 'red');
    hold on
    xlabel('log[dataset sample count]'),
    ylabel('Estimate of M using kfold'),
end
legend('Min', 'Max', 'Median', 'True M')
saveas(gcf,strcat('kfold_minmaxmedian_', string(data), 'results.png'));


for data = 1:length(datasizes)
    figure
    histogram(predictions(:, data, :))
    title(strcat('Prediction of # of Gaussian Components in GMM using kfold , Set size 10^', string(data+1)))
    xlabel('Number of Gaussian Components in GMM')
    ylabel('frequency')
    saveas(gcf,strcat('Prediction of # of Gaussian Components in GMM using kfold , Set size 10^', strcat(string(data+1),'.png')));
end


