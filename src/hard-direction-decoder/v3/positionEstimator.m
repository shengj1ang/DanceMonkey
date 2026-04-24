function [decodedPosX, decodedPosY, updatedModelParameters] = positionEstimator(test_data, modelParameters)
%==========================================================================
% positionEstimator
%
% Method #3:
%   - Direction classification: KNN (cosine similarity) at >=320ms.
%   - Velocity: per-direction ridge regression on lag-stacked binned spikes.
%   - Position: light Kalman filter on [x,y,vx,vy] using regressed velocity
%     as measurement (reduces drift/noise).
%
% Fully causal and I/O compatible.
%==========================================================================

    persistent lastT dir pos Sbin nBinsDone lastProcessedBin ...
               kfX kfP kfInit nFeatCached

    spikes = test_data.spikes;
    T = size(spikes, 2);

    %------------------------ Detect trial reset ----------------------------
    if isempty(lastT) || T < lastT
        lastT = 0;
        dir = 1;

        if isfield(test_data, 'startHandPos') && ~isempty(test_data.startHandPos)
            pos = double(test_data.startHandPos(:));
        else
            pos = [0; 0];
        end

        Sbin = [];
        nBinsDone = 0;
        lastProcessedBin = 0;

        kfX = zeros(4,1);
        kfP = eye(4);
        kfInit = false;

        nFeatCached = [];
    end
    lastT = T;

    %------------------------------ Parameters ------------------------------
    binSizeMs  = modelParameters.binSizeMs;
    lagBins    = modelParameters.lagBins;
    dt         = modelParameters.dt;
    numNeurons = 98;

    if isempty(nFeatCached)
        nFeatCached = numNeurons * lagBins;
    end
    nFeat = nFeatCached;

    nBinsNow = floor(T / binSizeMs);

    % Allocate/extend binned spike matrix
    if isempty(Sbin)
        Sbin = zeros(numNeurons, max(nBinsNow, 1));
    elseif size(Sbin, 2) < nBinsNow
        Sbin(:, end+1:nBinsNow) = 0;
    end

    %------------------------- Incremental binning --------------------------
    for b = (nBinsDone+1):nBinsNow
        idx1 = (b-1)*binSizeMs + 1;
        idx2 = b*binSizeMs;
        Sbin(:, b) = sum(spikes(:, idx1:idx2), 2);
    end
    nBinsDone = nBinsNow;

    %---------------------- Direction classification (>=320 ms) -------------
    if nBinsNow*binSizeMs >= 320 && lastProcessedBin == 0
        tClass = min(320, T);
        fr = mean(spikes(:, 1:tClass), 2); % 98 x 1

        % Unit-normalize for cosine similarity
        frNorm = sqrt(sum(fr.^2)) + 1e-12;
        frUnit = (fr / frNorm);           % 98 x 1

        % KNN cosine similarity = dot(knnFrUnit, frUnit)
        simsAll = modelParameters.knnFrUnit * frUnit;  % N x 1
        K = modelParameters.knnK;

        % Take top-K most similar samples
        [simsSorted, idxSorted] = sort(simsAll, 'descend');
        idxTop = idxSorted(1:min(K, numel(idxSorted)));
        simsTop = simsSorted(1:min(K, numel(simsSorted)));

        labelsTop = modelParameters.knnLabels(idxTop); % K x 1

        % Weighted vote by similarity (more robust than plain majority)
        scoreDir = zeros(1, 8);
        for i = 1:numel(labelsTop)
            d = labelsTop(i);
            scoreDir(d) = scoreDir(d) + max(simsTop(i), 0); % ignore negative sims
        end
        [~, dir] = max(scoreDir);

        % Initialize position
        if isfield(test_data, 'startHandPos') && ~isempty(test_data.startHandPos)
            pos = double(test_data.startHandPos(:));
        else
            pos = modelParameters.x0y0{dir};
        end

        % Initialize Kalman state
        kfX = [pos(1); pos(2); 0; 0];
        kfP = diag([50^2, 50^2, 600^2, 600^2]);
        kfInit = true;

        lastProcessedBin = 1;
    end

    %---------------------- Kalman filter model -----------------------------
    A = [1 0 dt 0;
         0 1 0 dt;
         0 0 1  0;
         0 0 0  1];

    qPos = 1.5;    % mm
    qVel = 35;     % mm/s
    Q = diag([qPos^2, qPos^2, qVel^2, qVel^2]);

    H = [0 0 1 0;
         0 0 0 1];

    rVel = 180;    % mm/s
    R = diag([rVel^2, rVel^2]);

    I4 = eye(4);

    %---------------------- Update per new bin ------------------------------
    if nBinsNow >= (lagBins + 1) && lastProcessedBin >= 1 && kfInit
        kStart = max(lagBins, lastProcessedBin);
        kEnd   = nBinsNow - 1;

        W  = modelParameters.velW{dir};
        b0 = modelParameters.velB{dir};
        mu = modelParameters.muX{dir};
        sg = modelParameters.sigX{dir};

        for k = kStart:kEnd
            feat = zeros(nFeat, 1);
            base = 1;
            for lag = 0:(lagBins-1)
                feat(base:base+numNeurons-1) = Sbin(:, k-lag);
                base = base + numNeurons;
            end

            xrow = (feat.' - mu) ./ sg;
            vHat = (xrow * W) + b0;   % 1 x 2
            z = vHat(:);              % 2 x 1

            % Predict
            kfX = A * kfX;
            kfP = A * kfP * A.' + Q;

            % Update
            S = H * kfP * H.' + R;
            K = (kfP * H.') / S;
            innov = z - (H * kfX);
            kfX = kfX + K * innov;
            kfP = (I4 - K * H) * kfP;

            pos = kfX(1:2);
        end

        lastProcessedBin = nBinsNow;
    end

    decodedPosX = pos(1);
    decodedPosY = pos(2);

    updatedModelParameters = modelParameters;
    updatedModelParameters.currentDirection = dir;
    updatedModelParameters.predictedAngle = (dir - 1) * (pi/4);
end