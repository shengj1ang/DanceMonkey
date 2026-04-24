function modelParameters = positionEstimatorTraining(training_data)
%==========================================================================
% positionEstimatorTraining
%
% This function trains a fast, fully-causal continuous decoder for the
% Neural Decoding Competition.
%
% It also supports model caching:
%   - If modelParameters.mat exists in the current folder, it will be loaded
%     and training will be skipped (saves a lot of time).
%   - Otherwise it will train from training_data and save the model.
%==========================================================================

    %-------------------------- Cache: load if exists -----------------------
    cacheFile = 'modelParameters.mat';
    if exist(cacheFile, 'file') == 2
        S = load(cacheFile, 'modelParameters');
        if isfield(S, 'modelParameters')
            modelParameters = S.modelParameters;
            return;
        end
    end

    %----------------------------- Hyperparams -----------------------------
    binSizeMs = 20;
    lagBins   = 3;
    dt        = binSizeMs/1000;
    lambda    = 50;

    numNeurons = 98;
    numDirs    = 8;

    %--------------------- Containers for training data --------------------
    dirTemplatesSum = zeros(numNeurons, numDirs);
    dirTemplatesCnt = zeros(1, numDirs);

    Xall = cell(1, numDirs);
    Yall = cell(1, numDirs);

    x0y0Sum = zeros(2, numDirs);
    x0y0Cnt = zeros(1, numDirs);

    %------------------------------ Build data -----------------------------
    nTrials = size(training_data, 1);

    for tr = 1:nTrials
        for d = 1:numDirs
            spikes  = training_data(tr, d).spikes;          % 98 x T
            handPos = training_data(tr, d).handPos(1:2, :); % 2  x T
            T = size(spikes, 2);

            % Typical initial position
            x0y0Sum(:, d) = x0y0Sum(:, d) + handPos(:, 1);
            x0y0Cnt(d)    = x0y0Cnt(d) + 1;

            % Direction template from first 320 ms
            t320 = min(320, T);
            fr320 = mean(spikes(:, 1:t320), 2);
            dirTemplatesSum(:, d) = dirTemplatesSum(:, d) + fr320;
            dirTemplatesCnt(d)    = dirTemplatesCnt(d) + 1;

            % Bin spikes
            nBins = floor(T / binSizeMs);
            if nBins <= lagBins
                continue;
            end

            Sbin = zeros(numNeurons, nBins);
            for b = 1:nBins
                idx1 = (b-1)*binSizeMs + 1;
                idx2 = b*binSizeMs;
                Sbin(:, b) = sum(spikes(:, idx1:idx2), 2);
            end

            % Bin positions at bin ends
            P = zeros(2, nBins);
            for b = 1:nBins
                idx2 = b*binSizeMs;
                P(:, b) = handPos(:, idx2);
            end

            % Velocity target (mm/s)
            V = diff(P, 1, 2) / dt; % 2 x (nBins-1)

            kStart = lagBins;
            kEnd   = nBins - 1;

            nSamples = kEnd - kStart + 1;
            nFeat = numNeurons * lagBins;

            X = zeros(nSamples, nFeat);
            Y = zeros(nSamples, 2);

            row = 1;
            for k = kStart:kEnd
                feat = zeros(nFeat, 1);
                base = 1;
                for lag = 0:(lagBins-1)
                    feat(base:base+numNeurons-1) = Sbin(:, k-lag);
                    base = base + numNeurons;
                end
                X(row, :) = feat.';
                Y(row, :) = V(:, k).';
                row = row + 1;
            end

            Xall{d} = [Xall{d}; X];
            Yall{d} = [Yall{d}; Y];
        end
    end

    %--------------------------- Finalise templates -------------------------
    dirTemplates = zeros(numNeurons, numDirs);
    for d = 1:numDirs
        if dirTemplatesCnt(d) > 0
            dirTemplates(:, d) = dirTemplatesSum(:, d) / dirTemplatesCnt(d);
        end
    end

    templateNorms = sqrt(sum(dirTemplates.^2, 1)) + 1e-12;
    dirTemplatesUnit = dirTemplates ./ templateNorms;

    x0y0 = cell(1, numDirs);
    for d = 1:numDirs
        if x0y0Cnt(d) > 0
            x0y0{d} = x0y0Sum(:, d) / x0y0Cnt(d);
        else
            x0y0{d} = [0; 0];
        end
    end

    %-------------------------- Train ridge regressors ----------------------
    velW = cell(1, numDirs);
    velB = cell(1, numDirs);
    muXcell = cell(1, numDirs);

    for d = 1:numDirs
        X = Xall{d};
        Y = Yall{d};

        if isempty(X)
            velW{d} = zeros(numNeurons*lagBins, 2);
            velB{d} = zeros(1, 2);
            muXcell{d} = zeros(1, numNeurons*lagBins);
            continue;
        end

        muX = mean(X, 1);
        Xc  = X - muX;

        XtX = (Xc.' * Xc);
        nFeat = size(XtX, 1);

        W = (XtX + lambda * eye(nFeat)) \ (Xc.' * Y);
        b = mean(Y, 1);

        velW{d} = W;
        velB{d} = b;
        muXcell{d} = muX;
    end

    %------------------------------ Pack output -----------------------------
    modelParameters = struct();
    modelParameters.binSizeMs    = binSizeMs;
    modelParameters.lagBins      = lagBins;
    modelParameters.dt           = dt;
    modelParameters.lambda       = lambda;

    modelParameters.dirTemplates = dirTemplatesUnit; % 98 x 8
    modelParameters.velW         = velW;
    modelParameters.velB         = velB;
    modelParameters.muX          = muXcell;
    modelParameters.x0y0         = x0y0;

    %------------------------------- Save cache -----------------------------
    try
        save(cacheFile, 'modelParameters', '-v7.3');
    catch
        save(cacheFile, 'modelParameters');
    end
end