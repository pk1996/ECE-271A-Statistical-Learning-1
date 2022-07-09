classdef GaussianMixtureModel < handle

    % GaussianMixtureModel - class to train a Gaussian mixture model and
    % use it to predict the class conditional probability.
    % gmm = GaussianMixtureModel(C,D,threshold, maxIters, verbose) to
    % create a GMM object.
    % gmm.train(data) - To train GMM with data 
    % gmm.predict(x) - To calculate probability.

    properties
        % Parameters to define a GMM
        Components
        Dimension
        Threshold
        Verbose
        MaxIters
    end

    properties
        % Parameters to store the trained parameters values
        Params
    end

    methods
        %------------------------------------------------------------------
        % Constructor
        function this = GaussianMixtureModel(C, D, threshold, maxIters, verbose)
            this.Components = C;
            this.Dimension = D;
            this.Threshold = threshold;
            this.MaxIters = maxIters;
            this.Verbose = verbose;
        end
        
        %------------------------------------------------------------------
        function train(this, data)
            %% Parameters for training GMM
            % Mixture components
            C = this.Components;
            % Dimension of feature space
            D = this.Dimension;
            % Threshold for converging
            threshold = this.Threshold;

            %% Gaussian Mixture Model

            % Taking first D features.
            data = data(:,1:D);

            % Initialize parameters
            params = this.initaliseParameters(C,D);
            % Count on number of EM steps
            EM_step = 1;
            % change in likelihood
            lc = 1;
            if (this.Verbose)
                    disp('Training started');
            end
            while(lc > threshold && EM_step <= this.MaxIters)
                likelihood_before = this.computeLikelihood(data, params, D);
                % Generate matrix of posterior probabilties
                H = this.generatePosteriorProbability(data, params, C, D);
                % Update parameters
                params = this.updateParams(data, H, C);
                likelihood_after = this.computeLikelihood(data, params, D);
                % change in likelihood
                lc = abs(likelihood_after - likelihood_before);
                EM_step = EM_step + 1;
            end
            if (this.Verbose)
                    disp('Training ccompleted');
            end
            this.Params = params;
        end
        
        %------------------------------------------------------------------
        function p = predict(this, x)
            % Calculate the class conditional probability using the learnt
            % GMM and given data point x
            [mean, cov,  P] = this.unpackParams(this.Params,this.Dimension);
            p = 0;
            % Predict using first d feature dimension.
            dim = size(x);
            if(length(dim) == 2)
                d = 1;
            else
                d = dim(3);
            end
            for j = 1 :  size(this.Params,1)
                p = p + mvnpdf(reshape(x, [dim(1)*dim(2),d]), mean(j,1:d), cov(j,1:d))*P(j);%this.mvg(x, mean(j,1:d), cov(j,1:d))*P(j);
            end
            p = reshape(p, [dim(1), dim(2)]);
        end
    end

    %----------------------------------------------------------------------
    % Private helper methods
    %----------------------------------------------------------------------
    methods (Access = private)
        %------------------------------------------------------------------
        function params_new = updateParams(this, data, H, C)
            % Update the parameters given the posterior probabiities
            % Get current parameters
            N = length(data);

            % Update component probability
            p_new = max(sum(H,1)/N,0.001);
            mean_new = zeros(C,this.Dimension);
            cov_new = zeros(C,this.Dimension);

            for j = 1 : C
                % update component mean
                mean_new(j,:) = sum(H(:,j).*data)/(N*p_new(j));

                % update component covariance
                cov_new(j,:) = max(sum(H(:,j).*power(data- mean_new(j,:),2))/(N*p_new(j)),1e-3);
            end

            % Update parameters
            params_new = this.packParams(mean_new, cov_new, p_new');
        end

        %------------------------------------------------------------------
        function H = generatePosteriorProbability(this, data, params, C, D)
            % Function generates the posterior probability matrix
            H = zeros(size(data,1), C);
            for j = 1 : size(H,2)
                [mean, cov, p] = this.unpackParams(params(j,:), D);
                H(:,j) = mvnpdf(data,mean, cov)*p;
            end
            H = H./sum(H,2);
        end

        %------------------------------------------------------------------
        function params = initaliseParameters(this, C, D)
            % Random initialization of parameters of the C components. Each
            % component has a D dimensional mean & covaraince and probability value
            % assosciated with that class
            % M - CxD mean matrix
            % C - CxD covariance matrix
            % P - 1xD probability vector

            % probabilities should add up to 1.
            P = rand(C,1);
            P = P/sum(P);

            % Initialize mean and covaraince.
            M = rand(C,D);
            Co = rand(C,D);
            params = this.packParams(M, Co, P);
        end

        %------------------------------------------------------------------
        function likelihood = computeLikelihood(this, data, params, D)
            [mean, cov,  P] = this.unpackParams(params,D);
            
            p = zeros(size(data,1),1);

            for j = 1 : size(params,1)
                p = p + mvnpdf(data, mean(j,:), cov(j,:))*P(j);
            end
            likelihood = sum(log(p));
        end
    end

    methods (Static, Access = private)
        %------------------------------------------------------------------
        function [mean, cov,  p] = unpackParams(params,D)
            % Helper function to extract parameters
            mean = params(:,1:D);
            cov = params(:,D+1:2*D);
            p = params(:,2*D+1);
        end

        %------------------------------------------------------------------
        function params = packParams(mean, cov, p)
            % Pack parameters into 1 unified matrix
            params = [mean cov p];
        end
    end
end