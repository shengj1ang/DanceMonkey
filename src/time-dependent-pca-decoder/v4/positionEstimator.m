function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)

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
    rate = preprocessSpikes(spikes, binSize, modelParameters.emaAlpha);
    rate = rate(modelParameters.keepIdx, :);

    if currBinCount <= startBin; tIdx = 1;
    else; tIdx = currBinCount - startBin + 1; end
    tIdx = max(1, min(tIdx, numel(modelParameters.timeBins)));

    % ---------------------------------------
    % Direction Scoring (Cumulative)
    % ---------------------------------------
    dirScore = getDirScore(rate, tIdx, modelParameters);
    
    % 如果 decodedHandPos 为空，代表进入了新的一轮试次，清空累积得分！
    if ~isfield(test_data, 'decodedHandPos') || isempty(test_data.decodedHandPos)
        modelParameters.cumScore = dirScore;
    else
        % 利用记忆平滑方向预测，抗干扰
        modelParameters.cumScore = 0.6 * modelParameters.cumScore + 0.4 * dirScore;
    end

    % 强化置信度高的方向 (Softmax 效果)
    w = modelParameters.cumScore.^3; 
    w = w / sum(w);

    % ---------------------------------------
    % Ensembled Regression (Extreme Fast O(1))
    % ---------------------------------------
    usedBins = min(currBinCount, modelParameters.timeBins(tIdx));
    feat = rate(:,1:usedBins);
    feat = feat(:)'; % 1 x L

    x_pred = 0; 
    y_pred = 0;
    
    % 将 8 个方向的回归结果按概率混合，避免选错方向导致的极端灾难
    for d = 1:modelParameters.nDirs
        R = modelParameters.regModel(d, tIdx);
        L = min(numel(feat), numel(R.mx));
        
        f_d = feat(1:L) - R.mx(1:L); % 1 x L
        
        % 得益于我们在训练中合并了权重，这里的投影只需要做一次普通的向量点积！
        x_d = f_d * R.W_x(1:L) + R.x0;
        y_d = f_d * R.W_y(1:L) + R.y0;
        
        x_pred = x_pred + w(d) * x_d;
        y_pred = y_pred + w(d) * y_d;
    end

    x = x_pred;
    y = y_pred;

    % ---------------------------------------
    % Temporal smoothing
    % ---------------------------------------
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
function score = getDirScore(rate, tIdx, modelParameters)
    C = modelParameters.classModel(tIdx);
    feat = rate(:,1:modelParameters.timeBins(tIdx));
    feat = feat(:) - C.mu; 
    
    % 极速计算：省去了两层计算，直接一步投影！
    zlda = C.W_cls' * feat; 

    d2 = sum((C.Ztrain - zlda).^2, 1);
    [d2s, idx] = sort(d2, 'ascend');
    
    k = 20; % 找最近的 20 个样本来给方向投票
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

function rate = preprocessSpikes(spikes, binSize, emaAlpha)
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