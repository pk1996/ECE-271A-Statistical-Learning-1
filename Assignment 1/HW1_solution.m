clc;
clear all;

% Load training data
load('TrainingSamplesDCT_8.mat');

%%  5a)
% Calculate a-priori probability
pF = size(TrainsampleDCT_FG,1)/(size(TrainsampleDCT_BG,1)...
    + size(TrainsampleDCT_FG,1));
pB = 1 - pF;

% Print the a-priori probabilities
disp("Prior probability of cheetah is ")
disp(pF)
disp("Prior probability of grass is ")
disp(pB)

%% 5b)
% Extract index of second largest element
[~,trainBG] = sort(abs(TrainsampleDCT_BG),2,'descend');
trainBG = trainBG(:,2);
[~,trainFG] = sort(abs(TrainsampleDCT_FG),2,'descend');
trainFG = trainFG(:,2);

% Calculate class conditional probability distribution for feature
figure();
hBG = histogram(trainBG, 64, 'BinEdges', (1:65), ...
    Normalization = 'probability');
hBG.Parent.Title.String = 'P_X_|_Y(x|cheetah)';
figure();
hFG = histogram(trainFG, 64, 'BinEdges', (1:65), ...
    Normalization = 'probability');
hFG.Parent.Title.String = 'P_X_|_Y(x|grass)';

%% 5c)
% Predict mask for test image
img = imread('cheetah.bmp');
mask = zeros(size(img));
error = zeros(size(img));
img = im2double(img);
img = padarray(img,[7,7],0,'post');

% Slide a 8X8 window over the image, calculate its DCT coeffecients. Select
% the index of 2nd largest value as the feature to calculate posterior
% probabilities.
zigZagIdx = readmatrix('Zig-Zag Pattern.txt');
for i = 1 : 255
    for j = 1 : 270
        block = img(i:i+7, j:j+7);
        dctF = abs(dct2(block));
        fIdx(zigZagIdx(:)+1) = dctF(:);
        [~,idx] = sort(fIdx, 2, 'descend');
        f = idx(2);
        
        % Make decision based on posteriror probability
        error(i,j) = hFG.Values(f);
        if(hBG.Values(f)*pB < hFG.Values(f)*pF)
            mask(i,j) = 1;
            error(i,j) = hBG.Values(f);
        end 
    end
end
figure();
imshow(mask);

%% 5d)

gTruth = im2double(imread('cheetah_mask.bmp'));  
nCheetah = nnz(gTruth);
nGrass = nnz(1 - gTruth);
nMislabeledCheetah = nnz((mask-gTruth)>0);
nMislabeledGrass = nnz((mask-gTruth)<0);
pError = nMislabeledGrass/nGrass*pB + nMislabeledCheetah/nCheetah*pF;
disp('Probability of error')
disp(pError)