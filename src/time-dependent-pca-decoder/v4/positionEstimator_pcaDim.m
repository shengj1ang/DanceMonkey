function [x, y, modelParameters] = positionEstimator_pcaDim(test_data, modelParameters)
% Position estimator paired with positionEstimatorTraining_pcaDim.m

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
    rate = preprocessSpikes_pcaDim_local(spikes, binSize, modelParameters.emaAlpha);
    rate = rate(modelParameters.keepIdx, :);

    if currBinCount <= startBin
        tIdx = 1;
    else
        tIdx = currBinCount - startBin + 1;
    end
    tIdx = max(1, min(tIdx, numel(modelParameters.timeBins)));

    dirScore = getDirScore_pcaDim_local(rate, tIdx, modelParameters);

    if ~isfield(test_data, 'decodedHandPos') || isempty(test_data.decodedHandPos)
        modelParameters.cumScore = dirScore;
    else
        modelParameters.cumScore = 0.6 * modelParameters.cumScore + 0.4 * dirScore;
    end

    w = modelParameters.cumScore.^3;
    w = w / sum(w);

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

    if isfield(test_data, 'decodedHandPos') && ~isempty(test_data.decodedHandPos)
        prev = test_data.decodedHandPos(:, end);
        x = 0.80 * x + 0.20 * prev(1);
        y = 0.80 * y + 0.20 * prev(2);
    else
        x = 0.60 * x + 0.40 * test_data.startHandPos(1);
        y = 0.60 * y + 0.40 * test_data.startHandPos(2);
    end
end

function score = getDirScore_pcaDim_local(rate, tIdx, modelParameters)
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

function rate = preprocessSpikes_pcaDim_local(spikes, binSize, emaAlpha)
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
