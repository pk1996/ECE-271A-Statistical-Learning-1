function [mask, pError] = MLEEstimate(data_BG, data_FG, I_cheetah, maskGT)

    % Function returns the probability of error on given test image and
    % gtruth mask.
    % The function expects the dataset of the 2 classes
    % Uses the Maximum likelihood approach
    
    % Calculate the MAP estimate of the model parameter(mean).
    [mu_ccd_FG, cov_ccd_FG] = calcMLEEstimate(data_FG);
    [mu_ccd_BG, cov_ccd_BG] = calcMLEEstimate(data_BG);
    
    % Calculate prior probabilities 
    p_BG = length(data_BG)/(length(data_BG) + length(data_FG));
    p_FG = 1 - p_BG;
    
    % Predict mask
    zigZagIdx = readmatrix('Zig-Zag Pattern.txt');
    mask = predictMask(cov_ccd_FG, cov_ccd_BG, mu_ccd_FG, mu_ccd_BG,...
        p_FG, p_BG, I_cheetah, zigZagIdx);
    
    % Calculate Error
    pError = calculateError(mask, maskGT, p_BG, p_FG);
end

%% Helper functions
function [mu_ccd, cc_cov] = calcMLEEstimate(data)
    % Given the data, this function returns the MLE estimate of
    % the model
    cc_cov = cov(data);
    mu_ccd = mean(data);
end