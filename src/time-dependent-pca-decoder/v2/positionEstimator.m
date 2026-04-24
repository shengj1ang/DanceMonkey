function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
% preprocessing -> PCA/LDA + kNN direction classification
% -> regression conditioned on direction and time window

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

    rate = preprocessSpikes( ...
        spikes, ...
        binSize, ...
        modelParameters.smoothMode, ...
        modelParameters.emaAlpha, ...
        modelParameters.gaussSigma);

    rate = rate(modelParameters.keepIdx, :);

    % map current bin count to stored model index
    if currBinCount <= startBin
        tIdx = 1;
    else
        tIdx = currBinCount - startBin + 1;
    end
    tIdx = max(1, min(tIdx, numel(modelParameters.timeBins)));

    % ---------------------------------------
    % direction classification
    % ---------------------------------------
    % New trajectory: decodedHandPos is empty, so force a fresh direction
    if ~isfield(test_data, 'decodedHandPos') || isempty(test_data.decodedHandPos)
        dirHat = classifyDirection(rate, tIdx, modelParameters);
        modelParameters.cachedDirection = dirHat;
    else
        % For ongoing trajectory, update direction estimate as more data arrives
        dirNow = classifyDirection(rate, tIdx, modelParameters);

        % small stabilization: do not switch too aggressively
        if isfield(modelParameters, 'cachedDirection') && ~isempty(modelParameters.cachedDirection)
            if currBinCount <= startBin + 2
                dirHat = dirNow;
            else
                % keep old one unless new classification is available;
                % this avoids random flips while still tracking current trial
                dirHat = dirNow;
            end
        else
            dirHat = dirNow;
        end

        modelParameters.cachedDirection = dirHat;
    end

    % ---------------------------------------
    % regression
    % ---------------------------------------
    usedBins = min(currBinCount, modelParameters.timeBins(tIdx));
    feat = rate(:,1:usedBins);
    feat = feat(:)';

    R = modelParameters.regModel(dirHat, tIdx);

    if numel(feat) ~= numel(R.mx)
        L = min(numel(feat), numel(R.mx));
        feat = feat(1:L);
        mx = R.mx(1:L);
        Wpca = R.Wpca(1:L,:);
    else
        mx = R.mx;
        Wpca = R.Wpca;
    end

    z = (feat - mx) * Wpca;
    x = z * R.bx + R.x0;
    y = z * R.by + R.y0;

    % mild temporal smoothing only within the same trajectory
    if isfield(test_data, 'decodedHandPos') && ~isempty(test_data.decodedHandPos)
        prev = test_data.decodedHandPos(:, end);
        x = 0.90 * x + 0.10 * prev(1);
        y = 0.90 * y + 0.10 * prev(2);
    end
end

% =========================================================
% helpers
% =========================================================

function label = classifyDirection(rate, tIdx, modelParameters)
    C = modelParameters.classModel(tIdx);

    feat = rate(:,1:modelParameters.timeBins(tIdx));
    feat = feat(:);

    feat = feat - C.mu;
    zpca = C.Wpca' * feat;
    zlda = C.Wlda' * zpca;

    trainZ = C.Ztrain;
    labels = C.labels;

    label = softKNN(zlda, trainZ, labels, modelParameters.nDirs, 18);
end

function label = softKNN(z, trainZ, labels, nDirs, k)
    d2 = sum((trainZ - z).^2, 1);
    [d2s, idx] = sort(d2, 'ascend');
    idx = idx(1:min(k, numel(idx)));
    d2s = d2s(1:min(k, numel(d2s)));

    w = 1 ./ (d2s + 1e-8);
    score = zeros(1, nDirs);

    for i = 1:numel(idx)
        c = labels(idx(i));
        score(c) = score(c) + w(i);
    end

    [~, label] = max(score);
end

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