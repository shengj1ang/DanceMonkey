function [decodedPosX, decodedPosY, updatedModelParameters] = positionEstimator(test_data, modelParameters)
%==========================================================================
% positionEstimator
%
% Method:
%   - Direction classification: LSTM on first 320ms binned spike sequence
%     (executed once per trial).
%   - Velocity decoding: per-direction ridge regression on lag-stacked bins.
%   - Position: light Kalman filter on [x,y,vx,vy].
%
% Fully causal and I/O compatible.
%==========================================================================

    persistent lastT dir pos Sbin nBinsDone lastProcessedBin ...
               kfX kfP kfInit nFeatCached dirDone

    spikes = test_data.spikes;
    T = size(spikes, 2);

    %------------------------ Detect trial reset ----------------------------
    if isempty(lastT) || T < lastT
        lastT = 0;
        dir = 1;
        dirDone = false;

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
    if ~dirDone && (nBinsNow*binSizeMs >= 320)
        steps = modelParameters.lstmBinSteps; % 16
        % Build 320ms sequence from the raw spikes (bin into 20ms counts)
        Sseq = zeros(numNeurons, steps);
        for b = 1:steps
            idx1 = (b-1)*binSizeMs + 1;
            idx2 = b*binSizeMs;
            Sseq(:, b) = sum(spikes(:, idx1:idx2), 2);
        end
        Sseq = Sseq / binSizeMs;

        % LSTM expects a cell array sequence (features x time)
        yPred = classify(modelParameters.lstmNet, {single(Sseq)}, ...
            "ExecutionEnvironment","cpu");
        dir = double(yPred);  % categorical labels are 1..8

        % Initialize position
        if isfield(test_data, 'startHandPos') && ~isempty(test_data.startHandPos)
            pos = double(test_data.startHandPos(:));
        else
            pos = modelParameters.x0y0{dir};
        end

        % Initialize Kalman
        kfX = [pos(1); pos(2); 0; 0];
        kfP = diag([50^2, 50^2, 600^2, 600^2]);
        kfInit = true;

        lastProcessedBin = 1;
        dirDone = true;
    end

    %---------------------- Kalman filter model -----------------------------
    A = [1 0 dt 0;
         0 1 0 dt;
         0 0 1  0;
         0 0 0  1];

    qPos = 1.5;
    qVel = 35;
    Q = diag([qPos^2, qPos^2, qVel^2, qVel^2]);

    H = [0 0 1 0;
         0 0 0 1];

    rVel = 180;
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
            vHat = (xrow * W) + b0;  % 1x2
            z = vHat(:);             % 2x1

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