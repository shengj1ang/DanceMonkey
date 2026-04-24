function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
% Main line kept:
% preprocessing -> PCA -> LDA -> kNN -> direction-conditioned regression

    cfg = modelParameters.cfg;

    spikes = test_data.spikes;
    spikes(isnan(spikes)) = 0;

    currBins = floor(size(spikes,2) / cfg.bin);
    currBins = min(currBins, modelParameters.nBins);

    if currBins < 1
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
        return;
    end

    spikes = spikes(:, 1:currBins*cfg.bin);
    rate = local_preprocess_test(spikes, cfg);
    rate = rate(modelParameters.keepIdx, :);

    stepIdx = currBins - modelParameters.startBin + 1;
    stepIdx = max(1, min(stepIdx, numel(modelParameters.modelBins)));

    % ------------------------------------
    % classify direction
    % ------------------------------------
    dirNow = local_predict_direction(rate, stepIdx, modelParameters);

    if ~isfield(test_data, 'decodedHandPos') || isempty(test_data.decodedHandPos)
        votes = zeros(1, modelParameters.nDirs);
        votes(dirNow) = 1;
    else
        votes = 0.7 * modelParameters.runtimeVotes;
        votes(dirNow) = votes(dirNow) + 1;
    end

    modelParameters.runtimeVotes = votes;
    [~, dirHat] = max(votes);

    % ------------------------------------
    % regress residual
    % ------------------------------------
    feat = reshape(rate(:, 1:currBins), [], 1)';
    R = modelParameters.regModel{dirHat, stepIdx};

    L = min(numel(feat), numel(R.muX));
    feat = feat(1:L);
    muX = R.muX(1:L);
    stdX = R.stdX(1:L);
    Wpca = R.Wpca(1:L,:);

    zn = ((feat - muX) ./ stdX) * Wpca;
    residual = zn * R.B;

    base = modelParameters.meanTraj(:, currBins, dirHat)';
    pred = base + residual;

    % ------------------------------------
    % tiny within-trajectory smoothing
    % ------------------------------------
    if isfield(test_data, 'decodedHandPos') && ~isempty(test_data.decodedHandPos)
        prev = test_data.decodedHandPos(:, end)';
        pred = 0.92 * pred + 0.08 * prev;
    end

    x = pred(1);
    y = pred(2);
end

% =========================================================
% local helpers
% =========================================================

function rate = local_preprocess_test(spikes, cfg)
    [n, T] = size(spikes);
    nb = floor(T / cfg.bin);
    rate = zeros(n, nb);

    for b = 1:nb
        idx1 = (b-1)*cfg.bin + 1;
        idx2 = b*cfg.bin;
        rate(:,b) = sum(spikes(:, idx1:idx2), 2);
    end

    rate = sqrt(rate);

    switch lower(cfg.smooth)
        case 'ema'
            out = zeros(size(rate));
            out(:,1) = rate(:,1);
            for b = 2:nb
                out(:,b) = cfg.ema_alpha * rate(:,b) + (1 - cfg.ema_alpha) * out(:,b-1);
            end
            rate = out / (cfg.bin / 1000);

        case 'gaussian'
            sigmaBins = cfg.gauss_sigma / cfg.bin;
            rad = max(2, ceil(3*sigmaBins));
            x = -rad:rad;
            k = exp(-(x.^2) / (2*sigmaBins^2));
            k = k / sum(k);

            out = zeros(size(rate));
            for i = 1:n
                out(i,:) = conv(rate(i,:), k, 'same');
            end
            rate = out / (cfg.bin / 1000);

        otherwise
            rate = rate / (cfg.bin / 1000);
    end
end

function dirHat = local_predict_direction(rate, stepIdx, modelParameters)
    C = modelParameters.classModel{stepIdx};
    bUse = modelParameters.modelBins(stepIdx);

    feat = local_class_feature_test(rate, bUse, modelParameters.cfg.class_recent_bins);
    feat = feat - C.mu;

    zpca = C.Wpca' * feat;
    zlda = C.Wlda' * zpca;

    dirHat = local_soft_knn(zlda, C.Ztrain, C.labels, modelParameters.nDirs, modelParameters.cfg.knn_k);
end

function feat = local_class_feature_test(rate, bNow, recentBins)
    left = max(1, bNow - recentBins + 1);

    cumPart = sum(rate(:, 1:bNow), 2);
    recentPart = sum(rate(:, left:bNow), 2);

    feat = [cumPart; recentPart];
end

function label = local_soft_knn(z, trainZ, labels, nDirs, k)
    dist2 = sum((trainZ - z).^2, 1);
    [dist2, ord] = sort(dist2, 'ascend');

    ord = ord(1:min(k, numel(ord)));
    dist2 = dist2(1:min(k, numel(dist2)));

    w = 1 ./ (dist2 + 1e-8);
    score = zeros(1, nDirs);

    for i = 1:numel(ord)
        score(labels(ord(i))) = score(labels(ord(i))) + w(i);
    end

    [~, label] = max(score);
end