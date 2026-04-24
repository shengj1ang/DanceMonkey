function modelParameters = positionEstimatorTraining(training_data)
% Main line kept:
% preprocess -> neuron screening -> PCA -> LDA -> kNN
% -> per-direction per-time regression
%
% Differences from the previous version:
% 1) classification uses cumulative + recent-window features
% 2) regression predicts residual relative to mean trajectory
% 3) classification PCA and regression PCA are separated
% 4) code structure is reorganized

    % -----------------------------
    % configuration
    % -----------------------------
    cfg = struct;
    cfg.bin = 20;
    cfg.start_ms = 320;
    cfg.smooth = 'ema';
    cfg.ema_alpha = 0.30;
    cfg.gauss_sigma = 45;

    cfg.min_rate = 0.5;
    cfg.class_recent_bins = 5;   % recent local window for classification
    cfg.class_pca_keep = 0.85;
    cfg.reg_pca_keep = 0.92;
    cfg.lda_dim = 5;

    cfg.knn_k = 18;
    cfg.reg_lambda = 8;

    [nTrials, nDirs] = size(training_data);

    % -----------------------------
    % common usable duration
    % -----------------------------
    T = zeros(nTrials, nDirs);
    for tr = 1:nTrials
        for d = 1:nDirs
            T(tr,d) = size(training_data(tr,d).spikes, 2);
        end
    end

    minLen = min(T(:));
    nBins = floor(minLen / cfg.bin);
    startBin = max(1, floor(cfg.start_ms / cfg.bin));
    modelBins = startBin:nBins;
    nSteps = numel(modelBins);

    % -----------------------------
    % preprocess all data once
    % -----------------------------
    rateCell = cell(nTrials, nDirs);
    posCell  = cell(nTrials, nDirs);

    for tr = 1:nTrials
        for d = 1:nDirs
            spikes = training_data(tr,d).spikes(:, 1:nBins*cfg.bin);
            spikes(isnan(spikes)) = 0;
            rateCell{tr,d} = local_preprocess(spikes, cfg);

            hp = training_data(tr,d).handPos(1:2, :);
            hp = local_fill_hand(hp);
            hp = hp(:, 1:nBins*cfg.bin);
            posCell{tr,d} = hp(:, cfg.bin:cfg.bin:nBins*cfg.bin);
        end
    end

    nNeurons = size(rateCell{1,1}, 1);

    % -----------------------------
    % neuron screening
    % -----------------------------
    meanRate = zeros(nNeurons,1);
    for n = 1:nNeurons
        s = 0;
        c = 0;
        for tr = 1:nTrials
            for d = 1:nDirs
                rr = rateCell{tr,d}(n,:);
                s = s + sum(rr);
                c = c + numel(rr);
            end
        end
        meanRate(n) = s / c;
    end

    keepIdx = find(meanRate >= cfg.min_rate);
    if isempty(keepIdx)
        keepIdx = 1:nNeurons;
    end

    % -----------------------------
    % mean trajectory per direction/time
    % -----------------------------
    meanTraj = zeros(2, nBins, nDirs);
    for d = 1:nDirs
        stack = zeros(2, nBins, nTrials);
        for tr = 1:nTrials
            stack(:,:,tr) = posCell{tr,d};
        end
        meanTraj(:,:,d) = mean(stack, 3);
    end

    % -----------------------------
    % classification models
    % -----------------------------
    classModel = cell(1, nSteps);

    for s = 1:nSteps
        bNow = modelBins(s);
        X = [];
        y = [];

        for d = 1:nDirs
            for tr = 1:nTrials
                feat = local_class_feature(rateCell{tr,d}(keepIdx,:), bNow, cfg.class_recent_bins);
                X(:, end+1) = feat;
                y(end+1) = d;
            end
        end

        mu = mean(X, 2);
        Xc = X - mu;

        [Wpca, Zpca] = local_pca_fit(Xc, cfg.class_pca_keep);
        Wlda = local_lda_fit(Zpca, y, cfg.lda_dim);
        Zlda = Wlda' * Zpca;

        classModel{s} = struct( ...
            'mu', mu, ...
            'Wpca', Wpca, ...
            'Wlda', Wlda, ...
            'Ztrain', Zlda, ...
            'labels', y ...
        );
    end

    % -----------------------------
    % regression models
    % -----------------------------
    regModel = cell(nDirs, nSteps);

    for d = 1:nDirs
        for s = 1:nSteps
            bNow = modelBins(s);

            X = [];
            Y = zeros(nTrials, 2);

            for tr = 1:nTrials
                feat = local_reg_feature(rateCell{tr,d}(keepIdx,:), bNow);
                X(tr, :) = feat';

                target = posCell{tr,d}(:, bNow)';
                base = meanTraj(:, bNow, d)';
                Y(tr, :) = target - base;
            end

            muX = mean(X, 1);
            stdX = std(X, 0, 1);
            stdX(stdX < 1e-8) = 1;

            Xn = (X - muX) ./ stdX;
            [WpcaReg, Zreg] = local_pca_fit(Xn', cfg.reg_pca_keep);
            Z = Zreg';   % samples x dim

            B = (Z' * Z + cfg.reg_lambda * eye(size(Z,2))) \ (Z' * Y);

            regModel{d,s} = struct( ...
                'muX', muX, ...
                'stdX', stdX, ...
                'Wpca', WpcaReg, ...
                'B', B ...
            );
        end
    end

    % -----------------------------
    % pack
    % -----------------------------
    modelParameters = struct;
    modelParameters.cfg = cfg;
    modelParameters.keepIdx = keepIdx;
    modelParameters.nDirs = nDirs;
    modelParameters.nBins = nBins;
    modelParameters.startBin = startBin;
    modelParameters.modelBins = modelBins;
    modelParameters.meanTraj = meanTraj;
    modelParameters.classModel = classModel;
    modelParameters.regModel = regModel;

    % runtime state for direction voting inside one trajectory only
    modelParameters.runtimeVotes = zeros(1, nDirs);
end

% =========================================================
% local helpers
% =========================================================

function rate = local_preprocess(spikes, cfg)
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

function hp = local_fill_hand(hp)
    for r = 1:size(hp,1)
        v = hp(r,:);
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

function feat = local_class_feature(rate, bNow, recentBins)
% cumulative activity + recent window activity
    left = max(1, bNow - recentBins + 1);

    cumPart = sum(rate(:, 1:bNow), 2);
    recentPart = sum(rate(:, left:bNow), 2);

    feat = [cumPart; recentPart];
end

function feat = local_reg_feature(rate, bNow)
% keep regression close to original mainline: use full history to bNow
    feat = reshape(rate(:, 1:bNow), [], 1);
end

function [W, Z] = local_pca_fit(X, keepRatio)
% X: features x samples
    C = X' * X;
    [V,D] = eig(C);
    evals = real(diag(D));
    [evals, ord] = sort(evals, 'descend');
    V = real(V(:, ord));

    good = evals > 1e-10;
    evals = evals(good);
    V = V(:, good);

    if isempty(evals)
        W = ones(size(X,1), 1);
        Z = W' * X;
        return;
    end

    r = cumsum(evals) / sum(evals);
    k = find(r >= keepRatio, 1, 'first');
    if isempty(k)
        k = numel(evals);
    end

    V = V(:, 1:k);
    evals = evals(1:k);

    W = X * V * diag(1 ./ sqrt(evals));
    Z = W' * X;
end

function Wlda = local_lda_fit(X, y, outDim)
% X: p x N
    cls = unique(y);
    p = size(X,1);
    mu = mean(X,2);

    Sw = zeros(p,p);
    Sb = zeros(p,p);

    for i = 1:numel(cls)
        idx = (y == cls(i));
        Xi = X(:, idx);
        mui = mean(Xi, 2);

        Di = Xi - mui;
        Sw = Sw + Di * Di';

        dm = mui - mu;
        Sb = Sb + sum(idx) * (dm * dm');
    end

    [V,D] = eig(pinv(Sw + 1e-6*eye(p)) * Sb);
    evals = real(diag(D));
    [~, ord] = sort(evals, 'descend');

    outDim = min(outDim, size(V,2));
    Wlda = real(V(:, ord(1:outDim)));
end