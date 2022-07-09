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