function modelParameters = positionEstimatorTraining(training_data)
%==========================================================================
% positionEstimatorTraining
%
% Method #3 (small change, often improves RMSE):
%   - Direction classification: KNN on first 320ms firing-rate vector
%     using cosine similarity (dot product on unit-normalised vectors).
%   - Trajectory decoding: per-direction ridge regression -> velocity
%     + light Kalman filter (handled in positionEstimator.m).
%
% Notes:
%   - KNN is evaluated once per trial (at >=320ms), so runtime impact is tiny.
%   - Keeps the exact I/O compatibility required by the test script.
%   - Includes cache with field-check to avoid stale modelParameters.mat.
%==========================================================================

    %-------------------------- Cache: load if exists -----------------------
    cacheFile = 'modelParameters.mat';
    if exist(cacheFile, 'file') == 2
        S = load(cacheFile, 'modelParameters');
        if isfield(S, 'modelParameters')
            mp = S.modelParameters;
            requiredFields = {'binSizeMs','lagBins','dt','lambda', ...
                              'velW','velB','muX','sigX','x0y0', ...
                              'knnFrUnit','knnLabels','knnK'};
            ok = true;
            for i = 1:numel(requiredFields)
                if ~isfield(mp, requiredFields{i})
                    ok = false; break;
                end
            end
            if ok
                modelParameters = mp;
                return;
            end
        end
    end

    %----------------------------- Hyperparams ------------------------------
    binSizeMs = 20;
    lagBins   = 3;
    dt        = binSizeMs/1000;
    lambda    = 50;

    numNeurons = 98;
    numDirs    = 8;

    % KNN for direction classification
    knnK = 11; % good starting point (try 7/11/15 later)

    %--------------------- Containers for training data --------------------
    Xall = cell(1, numDirs);
    Yall = cell(1, numDirs);

    x0y0Sum = zeros(2, numDirs);
    x0y0Cnt = zeros(1, numDirs);

    % Store 320ms firing-rate vectors for KNN (unit-normalised)
    knnFrUnit = [];   % [N_samples x 98]
    knnLabels = [];   % [N_samples x 1] labels 1..8

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

            % --------- KNN direction feature: first 320 ms -----------------
            t320 = min(320, T);
            fr320 = mean(spikes(:, 1:t320), 2); % 98 x 1
            frNorm = sqrt(sum(fr320.^2)) + 1e-12;
            frUnit = (fr320 / frNorm).';        % 1 x 98
            knnFrUnit = [knnFrUnit; frUnit];
            knnLabels = [knnLabels; d];

            % ---------------- Regression samples ---------------------------
            nBins = floor(T / binSizeMs);
            if nBins <= lagBins
                continue;
            end

            % Bin spikes
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

    %---------------------- Initial position fallback -----------------------
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
    muX  = cell(1, numDirs);
    sigX = cell(1, numDirs);

    for d = 1:numDirs
        X = Xall{d};
        Y = Yall{d};

        nFeat = numNeurons * lagBins;

        if isempty(X)
            velW{d} = zeros(nFeat, 2);
            velB{d} = zeros(1, 2);
            muX{d}  = zeros(1, nFeat);
            sigX{d} = ones(1, nFeat);
            continue;
        end

        mu = mean(X, 1);
        sg = std(X, 0, 1);
        sg(sg < 1e-6) = 1;

        Xz = (X - mu) ./ sg;

        XtX = (Xz.' * Xz);
        W = (XtX + lambda * eye(size(XtX, 1))) \ (Xz.' * Y);
        b = mean(Y, 1);

        velW{d} = W;
        velB{d} = b;
        muX{d}  = mu;
        sigX{d} = sg;
    end

    %------------------------------ Pack output -----------------------------
    modelParameters = struct();
    modelParameters.binSizeMs  = binSizeMs;
    modelParameters.lagBins    = lagBins;
    modelParameters.dt         = dt;
    modelParameters.lambda     = lambda;

    modelParameters.velW       = velW;
    modelParameters.velB       = velB;
    modelParameters.muX        = muX;
    modelParameters.sigX       = sigX;
    modelParameters.x0y0       = x0y0;

    % KNN direction classifier data
    modelParameters.knnFrUnit  = knnFrUnit;  % N x 98
    modelParameters.knnLabels  = knnLabels;  % N x 1
    modelParameters.knnK       = knnK;

    %------------------------------- Save cache -----------------------------
    try
        save(cacheFile, 'modelParameters', '-v7.3');
    catch
        save(cacheFile, 'modelParameters');
    end
end