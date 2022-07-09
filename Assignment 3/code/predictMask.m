function mask = predictMask(cov_ccd_FG, cov_ccd_BG,...
    mu_ccd_FG, mu_ccd_BG, p_FG, p_BG, I_cheetah, zigZagIdx)
    
    % Function predicts the mask on test image based on parameters of the
    % posterior distribution and class priors
    alpha_FG = log(det(cov_ccd_FG))- 2*log(p_FG);
    alpha_BG = log(det(cov_ccd_BG))- 2*log(p_BG);
    
    % Predict mask for test image
    I_cheetah = im2double(I_cheetah);
    mask = zeros(size(I_cheetah));
    I_cheetah = padarray(I_cheetah,[7,7],'replicate','post');
    
    % Slide a 8X8 window over the image, calculate its DCT coeffecients. Select
    % the index of 2nd largest value as the feature to calculate posterior
    % probabilities.
    for i = 1 : 255
        for j = 1 : 270
            block = I_cheetah(i:i+7, j:j+7);
            dctF = dct2(block);
            fIdx(zigZagIdx(:)+1) = dctF(:);
            f = fIdx;
    
            dFG = (f - mu_ccd_FG)*inv(cov_ccd_FG)*(f - mu_ccd_FG)' + alpha_FG;
            dBG = (f - mu_ccd_BG)*inv(cov_ccd_BG)*(f - mu_ccd_BG)' + alpha_BG;
            if(dFG < dBG)
                mask(i,j) = 1;
            end
        end
    end
end