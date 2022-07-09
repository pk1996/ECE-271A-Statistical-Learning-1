clc;
clear all;

cd('/home/parth/work/UCSD/Fall 2021/ECE 271 A - Statistical Learning 1/Assignement 3/hw3Data/errorResults');

if isfolder('plots')
    rmdir('plots', 's')
end
mkdir('plots');

dataset = {'D1';'D2';'D3';'D4'};
methods = {'BPE'; 'MAP'; 'MLE'};
load('/home/parth/work/UCSD/Fall 2021/ECE 271 A - Statistical Learning 1/Assignement 3/hw3Data/Alpha.mat');

for i = 1 : numel(dataset)
    for j = 1 : 2 
        y_min = 1;
        y_max = 0;
        for k = 1 : 3
           name = strcat(dataset{i}, '_', methods{k}, '_', int2str(j),'.mat');
           load(name);
           f = plot(log(alpha), pError, 'LineWidth', 2);
           y_min = min(y_min, min(pError));
           y_max = max(y_max, max(pError));
           hold on;
        end
        xlabel('log(\alpha)')
        ylabel('Probability of error')
        l = legend(methods, 'Location', 'best');
        l.FontSize = 13;
        title(strcat('Dataset ', int2str(i), ' | Strategy ', int2str(j)));
        name = strcat(dataset{i}, '_',int2str(j),'.jpg');
        saveas(gcf, fullfile('plots', name));
        close all;
    end
end