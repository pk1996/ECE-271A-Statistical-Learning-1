% Script to run experiments for Q6(a) & Q6(b)

clc;
clear all;

load('TrainingSamplesDCT_8_new.mat')

%% Training params
maxIter = 100;
C = [1,2,4,8,16,32];
d = [1,2,4,8,16,32,40,48,56,64];

if ~(isfile('Q6_a.mat'))
%% Q6 (a)
    disp('Learning Models for Q6(a)')
    models_1 = cell(5,2);
    for i = 1 : 5
        disp(strcat('GMM_' , int2str(i)))
        % learn mixture models with 8 cc
I_maskGT = imread('cheetah_mask.bmp');

% Prior probabilities
pFG = size(TrainsampleDCT_FG,1)/(size(TrainsampleDCT_BG,1)+size(TrainsampleDCT_FG,1));
pBG = 1 - pFG;

%% Predict
% Q6 (a) 25 combinations

pair_id = 1;
for a = 1 : 5
    for b = 1 : 5
        % For each of the 25 pairs of models, predict using d dim features.
        pError = d*0;
        modelFG = models_1{a,1};
        modelBG = models_1{b,2};
        for dIdx = 1 : length(d)
            disp(strcat('Model #',int2str(pair_id) ,' and ,dim = ', int2str(d(dIdx))));
            pError(dIdx) = segmentAndCalulateError(...
                I_cheetah_DCT, I_maskGT, modelBG, modelFG, d(dIdx), pFG, pBG);
        end

        % Save error plot
        name = strcat('GMM_', int2str(pair_id), '_6a');
        save(name, "pError");
        pair_id = pair_id + 1;
    end
end

% Q6 (b) 11 combinations
% Predict on cheetah image
for k = 1 : length(models_2)
    disp(strcat('Predicting using GMM of c = ', int2str(C(k))));
    % For each model, segment the test image with differnet dimension of
    % feature space
    modelFG = models_2{k,1};    
    modelBG = models_2{k,2};

    d = [1,2,4,8,16,32,40,48,56,64];
    pError = d*0;
    for dIdx = 1 : length(d)
        disp(strcat('Using c = ', int2str(C(k)), 'and dim = ', int2str(d(dIdx))));
        pError(dIdx) = segmentAndCalulateError(...
            I_cheetah_DCT, I_maskGT, modelBG, modelFG, d(dIdx), pFG, pBG);
    end
    
    % Save error plot
    name = strcat('GMM_', int2str(C(k)),'_Q6b');
    save(name, "pError");
end

%% Helper function
function I_cheetah_DCT = getDCTMatrix()
   % Helper function to get a 64 channel matrix of size of cheetah image
   % with each pixel storing the corresponding DCT coeffecients
   img = imread('cheetah.bmp');
   img = im2double(img);
   % Initialzie DCT matrix of image
   I_cheetah_DCT = zeros(size(img,1),size(img,2),64);
   % Pad image
   img = padarray(img,[7,7],'replicate','post');
   zigZagIdx = readmatrix('Zig-Zag Pattern.txt');
   for i = 1 : 255
       for j = 1 : 270
           block = img(i:i+7, j:j+7);
           dctF = dct2(block);
           fIdx(zigZagIdx(:)+1) = dctF(:);
           I_cheetah_DCT(i,j,:) = fIdx(:);
       end
   end
end

%--------------------------------------------------------------------------
function pError = segmentAndCalulateError(I_cheetah_DCT, I_maskGT, ...
    modelBG, modelFG, d, pFG, pBG)
    % Function takes input the DCT matrix, GMM for background and
    % foreground. Segments the image outputs the probability of error.
    I_cheetah_DCT_parsed = I_cheetah_DCT(:,:,1:d);
    dFG = modelFG.predict(I_cheetah_DCT_parsed)*pFG;
    dBG = modelBG.predict(I_cheetah_DCT_parsed)*pBG;
    mask = dFG*0;
    mask(dFG>dBG) = 1;
    pError = calculateError(mask, I_maskGT, pBG, pFG);
end

%--------------------------------------------------------------------------
function pError = calculateError(mask, gTruth, pB, pF)
    % Calculate probability of error given grounf truth mask, original
    % mask and class probabilities
    gTruth = im2double(gTruth);
    nCheetah = nnz(gTruth);
    nGrass = nnz(1 - gTruth);
    nMislabeledCheetah = nnz((mask-gTruth)>0);
    nMislabeledGrass = nnz((mask-gTruth)<0);
    pError = nMislabeledGrass/nGrass*pB + nMislabeledCheetah/nCheetah*pF;
end