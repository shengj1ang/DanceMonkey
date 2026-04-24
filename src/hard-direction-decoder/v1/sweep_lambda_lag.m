function results = sweep_lambda_lag()
%==========================================================================
% sweep_lambda_lag
%
% Grid search for (lagBins, lambda) without touching:
%   - testFunction_for_students_MTb.m
%   - positionEstimatorTraining.m
%
% This script builds a modelParameters struct internally (same fields as the
% fast linear decoder) and evaluates RMSE on a subset of test trials.
%==========================================================================

    load monkeydata_training.mat
    rng(2013);
    ix = randperm(length(trial));

    trainingData = trial(ix(1:50),:);
    testData     = trial(ix(51:end),:);

    % --- grid to search ---
    lagList = [2 3 4 5];
    lamList = [10 30 50 100 200];

    % --- evaluation subset (increase for more accurate selection) ---
    nEvalBlocks = 10;   % use 10 for fast sweep; set to size(testData,1) for full
    nEvalBlocks = min(nEvalBlocks, size(testData,1));

    results = []; % [lagBins, lambda, RMSE]

    for lagBins = lagList
        % Build training matrices once per lagBins (heavy part)
        trainCache = buildTrainingCache(trainingData, lagBins);

        for lambda = lamList
            % Train model quickly for this lambda (cheap once cache built)
            modelParameters = trainFromCache(trainCache, lambda);

            % Evaluate RMSE using the *existing* positionEstimator.m
            RMSE = evalRMSE(testData, modelParameters, nEvalBlocks);

            results = [results; lagBins lambda RMSE]; %#ok<AGROW>
            fprintf('lag=%d  lambda=%d  RMSE=%.4f\n', lagBins, lambda, RMSE);
        end
    end

    results = sortrows(results, 3);
    disp('Top 10 (lag, lambda, RMSE):');
    disp(results(1:min(10,size(results,1)), :));
end

%==========================================================================
% Helper: build templates + per-direction regression dataset for a lagBins
%==========================================================================
function cache = buildTrainingCache(training_data, lagBins)
    binSizeMs  = 20;
    dt         = binSizeMs/1000;
    numNeurons = 98;
    numDirs    = 8;

    % Direction templates (first 320 ms mean firing)
    dirTemplatesSum = zeros(numNeurons, numDirs);
    dirTemplatesCnt = zeros(1, numDirs);

    % Typical start position
    x0y0Sum = zeros(2, numDirs);
    x0y0Cnt = zeros(1, numDirs);

    % Regression data
    Xall = cell(1, numDirs);
    Yall = cell(1, numDirs);

    nTrials = size(training_data, 1);

    for tr = 1:nTrials
        for d = 1:numDirs
            spikes  = training_data(tr, d).spikes;          % 98 x T
            handPos = training_data(tr, d).handPos(1:2, :); % 2  x T
            T = size(spikes, 2);

            % Start position stats
            x0y0Sum(:, d) = x0y0Sum(:, d) + handPos(:, 1);
            x0y0Cnt(d)    = x0y0Cnt(d) + 1;

            % Direction template stats
            t320 = min(320, T);
            fr320 = mean(spikes(:, 1:t320), 2);
            dirTemplatesSum(:, d) = dirTemplatesSum(:, d) + fr320;
            dirTemplatesCnt(d)    = dirTemplatesCnt(d) + 1;

            % Bin counts
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

            % Bin-end positions
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

    % Final templates
    dirTemplates = zeros(numNeurons, numDirs);
    for d = 1:numDirs
        if dirTemplatesCnt(d) > 0
            dirTemplates(:, d) = dirTemplatesSum(:, d) / dirTemplatesCnt(d);
        end
    end
    norms = sqrt(sum(dirTemplates.^2, 1)) + 1e-12;
    dirTemplatesUnit = dirTemplates ./ norms;

    % Start positions
    x0y0 = cell(1, numDirs);
    for d = 1:numDirs
        if x0y0Cnt(d) > 0
            x0y0{d} = x0y0Sum(:, d) / x0y0Cnt(d);
        else
            x0y0{d} = [0; 0];
        end
    end

    % Precompute centered matrices products for ridge:
    % For each direction store:
    %   muX, XtX = Xc'Xc, XtY = Xc'Y, meanY
    cache = struct();
    cache.binSizeMs = binSizeMs;
    cache.lagBins   = lagBins;
    cache.dt        = dt;

    cache.dirTemplates = dirTemplatesUnit;
    cache.x0y0 = x0y0;

    numDirs = 8;
    cache.muX  = cell(1, numDirs);
    cache.XtX  = cell(1, numDirs);
    cache.XtY  = cell(1, numDirs);
    cache.meanY= cell(1, numDirs);

    for d = 1:numDirs
        X = Xall{d};
        Y = Yall{d};
        nFeat = numNeurons * lagBins;

        if isempty(X)
            cache.muX{d}   = zeros(1, nFeat);
            cache.XtX{d}   = zeros(nFeat, nFeat);
            cache.XtY{d}   = zeros(nFeat, 2);
            cache.meanY{d} = zeros(1, 2);
            continue;
        end

        muX = mean(X, 1);
        Xc  = X - muX;

        cache.muX{d}   = muX;
        cache.XtX{d}   = Xc.' * Xc;
        cache.XtY{d}   = Xc.' * Y;
        cache.meanY{d} = mean(Y, 1);
    end
end

%==========================================================================
% Helper: train a full modelParameters struct from cached matrices
%==========================================================================
function modelParameters = trainFromCache(cache, lambda)
    numDirs = 8;
    numNeurons = 98;
    lagBins = cache.lagBins;
    nFeat = numNeurons * lagBins;

    velW = cell(1, numDirs);
    velB = cell(1, numDirs);

    for d = 1:numDirs
        XtX = cache.XtX{d};
        XtY = cache.XtY{d};
        b0  = cache.meanY{d};

        if ~any(XtX(:))
            velW{d} = zeros(nFeat, 2);
            velB{d} = b0;
            continue;
        end

        W = (XtX + lambda * eye(nFeat)) \ XtY;
        velW{d} = W;
        velB{d} = b0;
    end

    modelParameters = struct();
    modelParameters.binSizeMs    = cache.binSizeMs;
    modelParameters.lagBins      = cache.lagBins;
    modelParameters.dt           = cache.dt;
    modelParameters.lambda       = lambda;

    modelParameters.dirTemplates = cache.dirTemplates;
    modelParameters.x0y0         = cache.x0y0;

    modelParameters.muX          = cache.muX;
    modelParameters.velW         = velW;
    modelParameters.velB         = velB;
end

%==========================================================================
% Helper: evaluate RMSE using the SAME calling convention as testFunction
%==========================================================================
function RMSE = evalRMSE(testData, modelParameters, nBlocks)
    meanSqError = 0;
    n_predictions = 0;

    for tr = 1:nBlocks
        for direc = 1:8
            decodedHandPos = [];
            times = 320:20:size(testData(tr,direc).spikes,2);

            for t = times
                past_current_trial.trialId = testData(tr,direc).trialId;
                past_current_trial.spikes  = testData(tr,direc).spikes(:,1:t);
                past_current_trial.decodedHandPos = decodedHandPos;
                past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1);

                % IMPORTANT: keep the same 3-output calling pattern
                [decodedPosX, decodedPosY, modelParameters] = positionEstimator(past_current_trial, modelParameters);

                decodedPos = [decodedPosX; decodedPosY];
                decodedHandPos = [decodedHandPos decodedPos]; %#ok<AGROW>

                err = testData(tr,direc).handPos(1:2,t) - decodedPos;
                meanSqError = meanSqError + (err.'*err);
            end
            n_predictions = n_predictions + length(times);
        end
    end

    RMSE = sqrt(meanSqError / n_predictions);
end