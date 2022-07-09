clc;
clear all;

% Driver to run the experiments across datasets, strategies and methods as
% in Assignment 3

% Handle folder to store results
if isfolder('errorResults')
    rmdir('errorResults', 's')
end
mkdir('errorResults');

% Load data
load('TrainingSamplesDCT_subsets_8.mat');

% Test image and gtruth
maskGT = imread('cheetah_mask.bmp');
I_cheetah = imread('cheetah.bmp');

% load alpha 
load("Alpha.mat");

% Loop over strategy
for strategy = 1 : 2  
    disp(strcat('Strategy -', int2str(strategy)));

    % Select strategy
    if (strategy == 1)
        load('Prior_1.mat');
    else
        load('Prior_2.mat');
    end
    
    % Loop over each dataset
    dataset = {'D1';'D2';'D3';'D4'};
    for d = 1 : length(dataset)
        disp(dataset{d});
        
        % Get data
        data_BG = eval(strcat(dataset{d},'_BG'));
        data_FG = eval(strcat(dataset{d},'_FG'));
        
        % To store probability of error
        pError = zeros(size(alpha));
        mask = {};
        % BPE for different alpha values
        for i = 1 : size(alpha,2)
            [mask{end+1}, pError(i)] = BPEEstimate(data_BG, data_FG, mu0_FG, mu0_BG, W0,...
                alpha(i), I_cheetah, maskGT);
        end

        name = getName(dataset(d), strategy, 'BPE');
        save(name,'pError');

        name = getName(dataset(d), strategy, 'BPE', true);
        save(name, 'mask');

        % MLE for different alpha values
        mask = {};
        for i = 1 : size(alpha,2)
            [mask{end+1}, pError(i)] = MAPEstimate(data_BG, data_FG, mu0_FG, mu0_BG, W0,...
                alpha(i), I_cheetah, maskGT);
        end

        name = getName(dataset(d), strategy, 'MAP');
        save(name,'pError');

        name = getName(dataset(d), strategy, 'MAP', true);
        save(name, 'mask');

        % MLE for different alpha values
        mask = {};
        [mask{end+1}, pErrorMLE] = MLEEstimate(data_BG, data_FG, I_cheetah, maskGT);
        pError = repmat(pErrorMLE, size(pError));

        name = getName(dataset(d), strategy, 'MLE');
        save(name,'pError');
        
        name = getName(dataset(d), strategy, 'MLE', true);
        save(name, 'mask');
    end
end

%% Helper Function
function name = getName(dataName, strategy, methods, isMask)
    % Returns an appropriate name for saving the error metric
    if(nargin == 3)
        isMask = false;
    end
    disp(methods);
    dataName = dataName{1};
    if (isMask)
        name = strcat(dataName, '_', methods, '_', int2str(strategy),'_mask.mat');
    else
        name = strcat(dataName, '_', methods, '_', int2str(strategy),'.mat');
    end
    name = fullfile(pwd, 'errorResults', name);
end