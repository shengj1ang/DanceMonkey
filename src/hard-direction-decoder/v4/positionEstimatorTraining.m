function modelParameters = positionEstimatorTraining(training_data)
%==========================================================================
% positionEstimatorTraining
%
% Method :
%   template cosine direction + per-dir ridge velocity
%   with standardization but stores a re-classification time for decoding.
%==========================================================================

    cacheFile = 'modelParameters.mat';
    if exist(cacheFile, 'file') == 2
        S = load(cacheFile, 'modelParameters');
        if isfield(S, 'modelParameters')
            mp = S.modelParameters;
            requiredFields = {'binSizeMs','lagBins','dt','lambda', ...
                              'dirTemplatesUnit','velW','velB','muX','sigX','x0y0','reclassMs'};
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

    % Hyperparams
    binSizeMs = 20;
    lagBins   = 3;
    dt        = binSizeMs/1000;
    lambda    = 50;

    numNeurons = 98;
    numDirs    = 8;

    dirTemplatesSum = zeros(numNeurons, numDirs);
    dirTemplatesCnt = zeros(1, numDirs);

    Xall = cell(1, numDirs);
    Yall = cell(1, numDirs);

    x0y0Sum = zeros(2, numDirs);
    x0y0Cnt = zeros(1, numDirs);

    nTrials = size(training_data, 1);

    for tr = 1:nTrials
        for d = 1:numDirs
            spikes  = training_data(tr, d).spikes;
            handPos = training_data(tr, d).handPos(1:2, :);
            T = size(spikes, 2);

            x0y0Sum(:, d) = x0y0Sum(:, d) + handPos(:, 1);
            x0y0Cnt(d)    = x0y0Cnt(d) + 1;

            % Template from first 320 ms
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

            % Bin positions
            P = zeros(2, nBins);
            for b = 1:nBins
                idx2 = b*binSizeMs;
                P(:, b) = handPos(:, idx2);
            end

            V = diff(P, 1, 2) / dt; % mm/s

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

    % Finalise templates
    dirTemplates = zeros(numNeurons, numDirs);
    for d = 1:numDirs
        if dirTemplatesCnt(d) > 0
            dirTemplates(:, d) = dirTemplatesSum(:, d) / dirTemplatesCnt(d);
        end
    end
    templateNorms = sqrt(sum(dirTemplates.^2, 1)) + 1e-12;
    dirTemplatesUnit = dirTemplates ./ templateNorms;

    % Initial position fallback
    x0y0 = cell(1, numDirs);
    for d = 1:numDirs
        if x0y0Cnt(d) > 0
            x0y0{d} = x0y0Sum(:, d) / x0y0Cnt(d);
        else
            x0y0{d} = [0; 0];
        end
    end

    % Ridge regressors with standardization
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

    modelParameters = struct();
    modelParameters.binSizeMs        = binSizeMs;
    modelParameters.lagBins          = lagBins;
    modelParameters.dt               = dt;
    modelParameters.lambda           = lambda;

    modelParameters.dirTemplatesUnit = dirTemplatesUnit;
    modelParameters.velW             = velW;
    modelParameters.velB             = velB;
    modelParameters.muX              = muX;
    modelParameters.sigX             = sigX;
    modelParameters.x0y0             = x0y0;

    % Re-classification time (ms): second direction check
    modelParameters.reclassMs        = 440;

    try
        save(cacheFile, 'modelParameters', '-v7.3');
    catch
        save(cacheFile, 'modelParameters');
    end
end