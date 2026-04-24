function modelParameters = positionEstimatorTraining_pcaDim(training_data, pcaDim)
% Architecture: Pre-computed PCA+LDA/Regression & Soft Ensembling
% This version allows manual PCA dimension control.
%
% Usage:
%   modelParameters = positionEstimatorTraining_pcaDim(training_data, 10);
%   modelParameters = positionEstimatorTraining_pcaDim(training_data, []);  % original auto variance mode

    if nargin < 2
        pcaDim = [];
    end

    % ----------------------------
    % hyperparameters
    % ----------------------------
    binSize = 20;
    startTime = 320;
    smoothMode = 'ema';
    emaAlpha = 0.35;
    minMeanRate = 0.5;
    pcaVarKeep = 0.95;      % used only when pcaDim = []
    ldaDim = 7;
    ridgeLambda = 10;

    [nTrials, nDirs] = size(training_data);

    % ----------------------------
    % determine shortest valid trial length
    % ----------------------------
    allLens = zeros(nTrials, nDirs);
    for i = 1:nTrials
        for d = 1:nDirs
            allLens(i,d) = size(training_data(i,d).spikes, 2);
        end
    end
    minLen = min(allLens(:));
    maxBinCount = floor(minLen / binSize);
    startBin = floor(startTime / binSize);
    if startBin < 1; startBin = 1; end
    timeBins = startBin:maxBinCount;
    nTimeModels = numel(timeBins);

    % ----------------------------
    % preprocess all training trials
    % ----------------------------
    proc = cell(nTrials, nDirs);
    posAtBins = cell(nTrials, nDirs);
    for i = 1:nTrials
        for d = 1:nDirs
            spikes = training_data(i,d).spikes(:, 1:maxBinCount*binSize);
            spikes(isnan(spikes)) = 0;
            proc{i,d} = preprocessSpikes_pcaDim(spikes, binSize, emaAlpha);

            hp = training_data(i,d).handPos(1:2, :);
            hp = fillHandTrajectory_pcaDim(hp);
            hp = hp(:, 1:maxBinCount*binSize);
            posAtBins{i,d} = hp(:, binSize:binSize:maxBinCount*binSize);
        end
    end
    nNeurons = size(proc{1,1}, 1);

    % ----------------------------
    % remove low-rate neurons
    % ----------------------------
    globalRate = zeros(nNeurons, 1);
    for n = 1:nNeurons
        acc = 0; cnt = 0;
        for i = 1:nTrials
            for d = 1:nDirs
                acc = acc + sum(proc{i,d}(n, :));
                cnt = cnt + size(proc{i,d}, 2);
            end
        end
        globalRate(n) = acc / cnt;
    end
    dropIdx = find(globalRate < minMeanRate);
    keepIdx = setdiff(1:nNeurons, dropIdx);
    keptNeuronCount = numel(keepIdx);

    % ----------------------------
    % build classification models
    % ----------------------------
    classModel = struct([]);
    classPcaDims = zeros(1, nTimeModels);

    for tIdx = 1:nTimeModels
        bNow = timeBins(tIdx);
        Xcls = zeros(keptNeuronCount * bNow, nTrials * nDirs);
        ycls = zeros(1, nTrials * nDirs);
        col = 1;
        for d = 1:nDirs
            for i = 1:nTrials
                featMat = proc{i,d}(keepIdx, 1:bNow);
                Xcls(:, col) = featMat(:);
                ycls(col) = d;
                col = col + 1;
            end
        end

        mu = mean(Xcls, 2);
        X0 = Xcls - mu;

        [Wpca, scorePca, ~, npc] = runPCAfromCov_pcaDim(X0, pcaVarKeep, pcaDim);
        Wlda = runLDA_pcaDim(scorePca, ycls, ldaDim);

        classPcaDims(tIdx) = npc;

        classModel(tIdx).mu = mu;
        classModel(tIdx).W_cls = Wpca(:, 1:npc) * Wlda;
        classModel(tIdx).Ztrain = Wlda' * scorePca;
        classModel(tIdx).labels = ycls;
    end

    % ----------------------------
    % train regression model
    % ----------------------------
    regModel = struct([]);
    regPcaDims = zeros(nDirs, nTimeModels);

    for d = 1:nDirs
        for tIdx = 1:nTimeModels
            bNow = timeBins(tIdx);
            Xreg = zeros(nTrials, keptNeuronCount * bNow);
            tx = zeros(nTrials,1);
            ty = zeros(nTrials,1);

            for i = 1:nTrials
                featMat = proc{i,d}(keepIdx, 1:bNow);
                Xreg(i,:) = featMat(:)';
                tx(i) = posAtBins{i,d}(1, bNow);
                ty(i) = posAtBins{i,d}(2, bNow);
            end

            mx = mean(Xreg, 1);
            Xc = Xreg - mx;

            [WpcaReg, scoreReg, ~, npcReg] = runPCAfromCov_pcaDim(Xc', pcaVarKeep, pcaDim);
            P = scoreReg';

            bx = ridgeSolve_pcaDim(P, tx - mean(tx), ridgeLambda);
            by = ridgeSolve_pcaDim(P, ty - mean(ty), ridgeLambda);

            regPcaDims(d, tIdx) = npcReg;

            regModel(d, tIdx).mx = mx;
            regModel(d, tIdx).W_x = WpcaReg(:, 1:npcReg) * bx;
            regModel(d, tIdx).W_y = WpcaReg(:, 1:npcReg) * by;
            regModel(d, tIdx).x0 = mean(tx);
            regModel(d, tIdx).y0 = mean(ty);
        end
    end

    % ----------------------------
    % package
    % ----------------------------
    modelParameters = struct;
    modelParameters.binSize = binSize;
    modelParameters.startTime = startTime;
    modelParameters.startBin = startBin;
    modelParameters.maxBinCount = maxBinCount;
    modelParameters.timeBins = timeBins;
    modelParameters.nDirs = nDirs;
    modelParameters.keepIdx = keepIdx;
    modelParameters.classModel = classModel;
    modelParameters.regModel = regModel;
    modelParameters.smoothMode = smoothMode;
    modelParameters.emaAlpha = emaAlpha;
    modelParameters.cumScore = [];

    modelParameters.pcaDimRequested = pcaDim;
    modelParameters.pcaVarKeep = pcaVarKeep;
    modelParameters.classPcaDims = classPcaDims;
    modelParameters.regPcaDims = regPcaDims;
end

% =========================================================
% Estimator
% =========================================================
function [x, y, modelParameters] = positionEstimator_pcaDim(test_data, modelParameters)

    binSize = modelParameters.binSize;
    startBin = modelParameters.startBin;
    maxBinCount = modelParameters.maxBinCount;

    spikes = test_data.spikes;
    spikes(isnan(spikes)) = 0;
    currBinCount = floor(size(spikes,2) / binSize);
    currBinCount = min(currBinCount, maxBinCount);

    if currBinCount < 1
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
        return;
    end

    spikes = spikes(:, 1:currBinCount*binSize);
    rate = preprocessSpikes_pcaDim(spikes, binSize, modelParameters.emaAlpha);
    rate = rate(modelParameters.keepIdx, :);

    if currBinCount <= startBin
        tIdx = 1;
    else
        tIdx = currBinCount - startBin + 1;
    end
    tIdx = max(1, min(tIdx, numel(modelParameters.timeBins)));

    % Direction Scoring
    dirScore = getDirScore_pcaDim(rate, tIdx, modelParameters);

    if ~isfield(test_data, 'decodedHandPos') || isempty(test_data.decodedHandPos)
        modelParameters.cumScore = dirScore;
    else
        modelParameters.cumScore = 0.6 * modelParameters.cumScore + 0.4 * dirScore;
    end

    w = modelParameters.cumScore.^3;
    w = w / sum(w);

    % Ensembled Regression
    usedBins = min(currBinCount, modelParameters.timeBins(tIdx));
    feat = rate(:,1:usedBins);
    feat = feat(:)';

    x_pred = 0;
    y_pred = 0;

    for d = 1:modelParameters.nDirs
        R = modelParameters.regModel(d, tIdx);
        L = min(numel(feat), numel(R.mx));

        f_d = feat(1:L) - R.mx(1:L);

        x_d = f_d * R.W_x(1:L) + R.x0;
        y_d = f_d * R.W_y(1:L) + R.y0;

        x_pred = x_pred + w(d) * x_d;
        y_pred = y_pred + w(d) * y_d;
    end

    x = x_pred;
    y = y_pred;

    % Temporal smoothing
    if isfield(test_data, 'decodedHandPos') && ~isempty(test_data.decodedHandPos)
        prev = test_data.decodedHandPos(:, end);
        x = 0.80 * x + 0.20 * prev(1);
        y = 0.80 * y + 0.20 * prev(2);
    else
        x = 0.60 * x + 0.40 * test_data.startHandPos(1);
        y = 0.60 * y + 0.40 * test_data.startHandPos(2);
    end
end

% =========================================================
% helpers
% =========================================================
function score = getDirScore_pcaDim(rate, tIdx, modelParameters)
    C = modelParameters.classModel(tIdx);
    feat = rate(:,1:modelParameters.timeBins(tIdx));
    feat = feat(:) - C.mu;

    zlda = C.W_cls' * feat;

    d2 = sum((C.Ztrain - zlda).^2, 1);
    [d2s, idx] = sort(d2, 'ascend');

    k = 20;
    idx = idx(1:min(k, numel(idx)));
    d2s = d2s(1:min(k, numel(d2s)));

    w_k = 1 ./ (d2s + 1e-6);
    score = zeros(1, modelParameters.nDirs);

    for i = 1:numel(idx)
        c = C.labels(idx(i));
        score(c) = score(c) + w_k(i);
    end

    score = score / sum(score);
end

function rate = preprocessSpikes_pcaDim(spikes, binSize, emaAlpha)
    [n, T] = size(spikes);
    nb = floor(T / binSize);
    rate = zeros(n, nb);

    for b = 1:nb
        rate(:,b) = sum(spikes(:,(b-1)*binSize+1 : b*binSize), 2);
    end

    rate = sqrt(rate);

    out = zeros(size(rate));
    out(:,1) = rate(:,1);

    for b = 2:nb
        out(:,b) = emaAlpha * rate(:,b) + (1-emaAlpha) * out(:,b-1);
    end

    rate = out / (binSize/1000);
end

function hp = fillHandTrajectory_pcaDim(hp)
    for r = 1:size(hp,1)
        v = hp(r,:);

        if all(~isnan(v))
            hp(r,:) = v;
            continue;
        end

        for i = 2:numel(v)
            if isnan(v(i))
                v(i) = v(i-1);
            end
        end

        for i = numel(v)-1:-1:1
            if isnan(v(i))
                v(i) = v(i+1);
            end
        end

        v(isnan(v)) = 0;
        hp(r,:) = v;
    end
end

function [W, score, latent, npc] = runPCAfromCov_pcaDim(X, varKeep, fixedDim)
    if nargin < 3
        fixedDim = [];
    end

    C = X' * X;
    [V,D] = eig(C);

    eigvals = diag(D);
    [eigvals, order] = sort(eigvals, 'descend');
    V = V(:, order);

    valid = eigvals > 1e-10;
    eigvals = eigvals(valid);
    V = V(:, valid);

    if isempty(eigvals)
        npc = 1;
        W = zeros(size(X,1), 1);
        score = zeros(1, size(X,2));
        latent = 0;
        return;
    end

    if ~isempty(fixedDim)
        npc = min(fixedDim, numel(eigvals));
    else
        ratio = cumsum(eigvals) / sum(eigvals);
        npc = find(ratio >= varKeep, 1, 'first');
        if isempty(npc)
            npc = numel(eigvals);
        end
    end

    V = V(:,1:npc);
    latent = eigvals(1:npc);

    W = X * V .* (1 ./ sqrt(latent(:)'));
    score = W' * X;
end

function Wlda = runLDA_pcaDim(X, y, outDim)
    cls = unique(y);
    p = size(X,1);
    mu = mean(X,2);

    Sw = zeros(p,p);
    Sb = zeros(p,p);

    for k = 1:numel(cls)
        idx = (y == cls(k));
        Xk = X(:,idx);
        muk = mean(Xk,2);

        Dk = Xk - muk;
        Sw = Sw + Dk * Dk';

        diff = muk - mu;
        Sb = Sb + sum(idx) * (diff * diff');
    end

    [V,D] = eig(pinv(Sw + 1e-6*eye(p)) * Sb);
    evals = real(diag(D));

    [~, order] = sort(evals, 'descend');
    outDim = min(outDim, size(V,2));

    Wlda = real(V(:, order(1:outDim)));
end

function b = ridgeSolve_pcaDim(X, y, lambda)
    p = size(X,2);
    b = (X' * X + lambda * eye(p)) \ (X' * y);
end
