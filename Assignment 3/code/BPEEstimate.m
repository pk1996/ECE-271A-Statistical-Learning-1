function [mask, pError] = BPEEstimate(data_BG, data_FG, mu_0_FG, mu_0_BG, W0,...
    alpha, I_cheetah, maskGT)

    % Function returns the probability of error on given test image and
    % gtruth mask.
    % The function expects the dataset of the 2 classes, and parameters of
    % the gaussian prior (mean).
    % Uses the Bayesian Parameter Estimation appraoch

    % Construct cov matrix of gaussian prior.
    cov_0_BG = alpha*diag(W0);
    cov_0_FG = alpha*diag(W0);
    
    % Calculate the predictive distribution parameters based on prior and data
    [mu_ccd_FG, cov_ccd_FG] = calcParamsPredictiveDist(data_FG, mu_0_FG, cov_0_FG);
    [mu_ccd_BG, cov_ccd_BG] = calcParamsPredictiveDist(data_BG, mu_0_BG, cov_0_BG);
    
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
function [mu_ccd, cov_ccd] = calcParamsPredictiveDist(data, mu_0, cov_0)
    % Given the prior and data, this function returns the parameters of the
    % predictive distribution
    cc_cov = cov(data);

    % Number of training data
    n = length(data);

    % Compute the sample means of the data.
    sample_mean = mean(data);

    % Calulate the mean and covaraince of the posteriror densities of the model
    % parameter (Mean of the Gaussian)
    mu_post  =  calculatePosteriorMean(cov_0, cc_cov, sample_mean, mu_0, n);
    cov_post =  calculatePosteriorCov(cov_0, cc_cov, n);

    % Calculate parameters of the predictive distribution
    mu_ccd = mu_post';
    cov_ccd = cc_cov + cov_post;
end

function mu_n = calculatePosteriorMean(sigma_0, sigma, mu_hat, mu_0, n)
    % Calculates posterior mean of model parameter (mean)
    % sigma_0 - Cov of prior
    % sigma - cov of ccd
    % mu_hat - sample mean
    % mu_0 - mean of prior
    % n - number of trainig data
    weighted_cov = pinv(sigma_0 + sigma/n);
    mu_n = sigma_0*weighted_cov*mu_hat' + (sigma*weighted_cov*mu_0')/n;
end

function sigma_n = calculatePosteriorCov(sigma_0, sigma, n)
    % Calculates posterior variance of model parameter (mean)
    % sigma_0 - cov of proir
    % sigma - cov of ccd
    % n - number of trainig data
    sigma_n = sigma_0*pinv((sigma_0 + sigma/n))*sigma/n;
end