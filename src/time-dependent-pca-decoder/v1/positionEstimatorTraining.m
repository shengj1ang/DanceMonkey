function modelParameters = positionEstimatorTraining(training_data)
% Architecture:
% preprocessing -> neuron filtering -> PCA/LDA for classification
% -> per-direction per-time regression for x/y prediction

    % ----------------------------
    % hyperparameters
    % ----------------------------
    binSize = 20;
    startTime = 320;
    smoothMode = 'ema';
    emaAlpha = 0.35;
    gaussSigma = 50;

    minMeanRate = 0.5;
    pcaVarKeep = 0.90;      % keep enough variance for classification/regression
    ldaDim = 6;
    ridgeLambda = 5;

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
    if startBin < 1
        startBin = 1;
    end

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

            proc{i,d} = preprocessSpikes(spikes, binSize, smoothMode, emaAlpha, gaussSigma);

            hp = training_data(i,d).handPos(1:2, :);
            hp = fillHandTrajectory(hp);
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
        acc = 0;
        cnt = 0;
        for i = 1:nTrials
            for d = 1:nDirs
                r = proc{i,d}(n, :);
                acc = acc + sum(r);
                cnt = cnt + numel(r);
            end
        end
        globalRate(n) = acc / cnt;
    end

    dropIdx = find(globalRate < minMeanRate);
    keepIdx = setdiff(1:nNeurons, dropIdx);
    keptNeuronCount = numel(keepIdx);

    % ----------------------------
    % build classification models for each time window
    % ----------------------------
    classModel = struct([]);

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

        [Wpca, scorePca, latent, npc] = runPCAfromCov(X0, pcaVarKeep);
        Wlda = runLDA(scorePca, ycls, ldaDim);

        Ztrain = Wlda' * scorePca;

        classModel(tIdx).mu = mu;
        classModel(tIdx).Wpca = Wpca(:, 1:npc);
        classModel(tIdx).Wlda = Wlda;
        classModel(tIdx).Ztrain = Ztrain;
        classModel(tIdx).labels = ycls;
    end

    % ----------------------------
    % mean trajectory at each time/direction
    % ----------------------------
    meanX = zeros(nTimeModels, nDirs);
    meanY = zeros(nTimeModels, nDirs);

    for d = 1:nDirs
        for tIdx = 1:nTimeModels
            bNow = timeBins(tIdx);
            xs = zeros(nTrials,1);
            ys = zeros(nTrials,1);
            for i = 1:nTrials
                xs(i) = posAtBins{i,d}(1, bNow);
                ys(i) = posAtBins{i,d}(2, bNow);
            end
            meanX(tIdx,d) = mean(xs);
            meanY(tIdx,d) = mean(ys);
        end
    end

    % ----------------------------
    % train regression model per direction/time
    % ----------------------------
    regModel = struct([]);

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

            [WpcaReg, scoreReg, ~, npcReg] = runPCAfromCov(Xc', pcaVarKeep);
            P = scoreReg';

            bx = ridgeSolve(P, tx - mean(tx), ridgeLambda);
            by = ridgeSolve(P, ty - mean(ty), ridgeLambda);

            regModel(d, tIdx).mx = mx;
            regModel(d, tIdx).Wpca = WpcaReg(:, 1:npcReg);
            regModel(d, tIdx).bx = bx;
            regModel(d, tIdx).by = by;
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
    modelParameters.dropIdx = dropIdx;

    modelParameters.classModel = classModel;
    modelParameters.regModel = regModel;
    modelParameters.meanX = meanX;
    modelParameters.meanY = meanY;

    modelParameters.smoothMode = smoothMode;
    modelParameters.emaAlpha = emaAlpha;
    modelParameters.gaussSigma = gaussSigma;

    modelParameters.cachedTrialId = [];
    modelParameters.cachedDirection = 1;
end

% =========================================================
% helpers
% =========================================================

function rate = preprocessSpikes(spikes, binSize, smoothMode, emaAlpha, gaussSigma)
    [n, T] = size(spikes);
    nb = floor(T / binSize);
    rate = zeros(n, nb);

    for b = 1:nb
        left = (b-1)*binSize + 1;
        right = b*binSize;
        rate(:,b) = sum(spikes(:,left:right), 2);
    end

    rate = sqrt(rate);

    switch lower(smoothMode)
        case 'ema'
            out = zeros(size(rate));
            out(:,1) = rate(:,1);
            for b = 2:nb
                out(:,b) = emaAlpha * rate(:,b) + (1-emaAlpha) * out(:,b-1);
            end
            rate = out / (binSize/1000);

        case 'gaussian'
            k = buildGaussianKernel(binSize, gaussSigma);
            out = zeros(size(rate));
            for i = 1:n
                out(i,:) = conv(rate(i,:), k, 'same');
            end
            rate = out / (binSize/1000);

        otherwise
            rate = rate / (binSize/1000);
    end
end

function k = buildGaussianKernel(binSize, sigmaMs)
    sigmaBins = sigmaMs / binSize;
    halfWidth = max(2, ceil(3 * sigmaBins));
    x = -halfWidth:halfWidth;
    k = exp(-(x.^2) / (2 * sigmaBins^2));
    k = k / sum(k);
end

function hp = fillHandTrajectory(hp)
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

function [W, score, latent, npc] = runPCAfromCov(X, varKeep)
% X is features x samples
    C = X' * X;
    [V,D] = eig(C);
    eigvals = diag(D);
    [eigvals, order] = sort(eigvals, 'descend');
    V = V(:, order);

    valid = eigvals > 1e-10;
    eigvals = eigvals(valid);
    V = V(:, valid);

    ratio = cumsum(eigvals) / sum(eigvals);
    npc = find(ratio >= varKeep, 1, 'first');
    if isempty(npc)
        npc = numel(eigvals);
    end

    V = V(:,1:npc);
    latent = eigvals(1:npc);

    W = X * V .* (1 ./ sqrt(latent(:)')) ;
    score = W' * X;
end

function Wlda = runLDA(X, y, outDim)
% X: p x N
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

function b = ridgeSolve(X, y, lambda)
% X: samples x features
    p = size(X,2);
    b = (X' * X + lambda * eye(p)) \ (X' * y);
end