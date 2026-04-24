function modelParameters = positionEstimatorTraining(training_data)
%==========================================================================
% positionEstimatorTraining
%
% Method #8:
%   - Direction classification: lightweight LSTM trained on the first 320ms
%     binned spike counts sequence (16 steps @ 20ms, 98 features).
%   - Trajectory decoding: per-direction ridge regression for velocity with
%     feature standardization.
%
% Runtime impact: small (LSTM is evaluated ONCE per trial at 320ms).
% Training time: higher (needs Deep Learning Toolbox).
% Compatibility: keeps the same modelParameters struct interface.
%==========================================================================

    %-------------------------- Cache: load if exists -----------------------
    cacheFile = 'modelParameters.mat';
    if exist(cacheFile, 'file') == 2
        S = load(cacheFile, 'modelParameters');
        if isfield(S, 'modelParameters')
            mp = S.modelParameters;
            requiredFields = {'binSizeMs','lagBins','dt','lambda', ...
                              'velW','velB','muX','sigX','x0y0', ...
                              'lstmNet','lstmBinSteps'};
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

    % LSTM direction settings
    lstmMs       = 320;
    lstmBinSteps = lstmMs / binSizeMs; % 16 steps

    %--------------------- Containers for training data ---------------------
    Xall = cell(1, numDirs);
    Yall = cell(1, numDirs);

    x0y0Sum = zeros(2, numDirs);
    x0y0Cnt = zeros(1, numDirs);

    % LSTM sequences: each sample is [98 x 16] (features x time)
    seqX = {};
    seqY = [];

    %------------------------------ Build data ------------------------------
    nTrials = size(training_data, 1);

    for tr = 1:nTrials
        for d = 1:numDirs
            spikes  = training_data(tr, d).spikes;          % 98 x T (ms)
            handPos = training_data(tr, d).handPos(1:2, :); % 2  x T (mm)
            T = size(spikes, 2);

            % Initial position stats
            x0y0Sum(:, d) = x0y0Sum(:, d) + handPos(:, 1);
            x0y0Cnt(d)    = x0y0Cnt(d) + 1;

            % ------------------- LSTM direction sequence -------------------
            % Use first 320ms spikes binned into 20ms counts => 16 timesteps.
            if T >= lstmMs
                Sseq = zeros(numNeurons, lstmBinSteps);
                for b = 1:lstmBinSteps
                    idx1 = (b-1)*binSizeMs + 1;
                    idx2 = b*binSizeMs;
                    Sseq(:, b) = sum(spikes(:, idx1:idx2), 2);
                end

                % Optional normalization per sample (helps training stability)
                % Scale counts to "rate-like" values
                Sseq = Sseq / binSizeMs; % spikes per ms (small numbers)

                seqX{end+1,1} = single(Sseq);
                seqY(end+1,1) = d;
            end

            % ---------------- Regression samples for velocity ---------------
            nBins = floor(T / binSizeMs);
            if nBins <= lagBins
                continue;
            end

            % Bin spikes across entire trial
            Sbin = zeros(numNeurons, nBins);
            for b = 1:nBins
                idx1 = (b-1)*binSizeMs + 1;
                idx2 = b*binSizeMs;
                Sbin(:, b) = sum(spikes(:, idx1:idx2), 2);
            end

            % Positions at bin ends
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
            nFeat    = numNeurons * lagBins;

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

    %------------------- Train per-direction ridge regressors ----------------
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

    %-------------------------- Train LSTM classifier ------------------------
    % This requires Deep Learning Toolbox. If not available, error will occur.
    yCat = categorical(seqY, 1:numDirs);

    layers = [
        sequenceInputLayer(numNeurons, "Name","in")
        lstmLayer(64, "OutputMode","last", "Name","lstm")
        fullyConnectedLayer(numDirs, "Name","fc")
        softmaxLayer("Name","sm")
        classificationLayer("Name","cls")
    ];

    options = trainingOptions("adam", ...
        "MaxEpochs", 6, ...
        "MiniBatchSize", 64, ...
        "InitialLearnRate", 1e-3, ...
        "Shuffle","every-epoch", ...
        "Verbose", false, ...
        "ExecutionEnvironment","cpu");

    lstmNet = trainNetwork(seqX, yCat, layers, options);

    %------------------------------ Pack output -----------------------------
    modelParameters = struct();
    modelParameters.binSizeMs      = binSizeMs;
    modelParameters.lagBins        = lagBins;
    modelParameters.dt             = dt;
    modelParameters.lambda         = lambda;

    modelParameters.velW           = velW;
    modelParameters.velB           = velB;
    modelParameters.muX            = muX;
    modelParameters.sigX           = sigX;
    modelParameters.x0y0           = x0y0;

    modelParameters.lstmNet        = lstmNet;
    modelParameters.lstmBinSteps   = lstmBinSteps;

    %------------------------------- Save cache -----------------------------
    try
        save(cacheFile, 'modelParameters', '-v7.3');
    catch
        save(cacheFile, 'modelParameters');
    end
end