function [decodedPosX, decodedPosY, updatedModelParameters] = positionEstimator(test_data, modelParameters)
%==========================================================================
% positionEstimator
%
% Method :
%   - Direction classification via cosine-template at 320ms (dir1)
%   - One optional re-classification at reclassMs (e.g. 440ms) (dir2)
%     Switch only if confidence improves sufficiently.
%   - Per-direction ridge velocity + light Kalman filter for smooth position.
%
% Fully causal and I/O compatible.
%==========================================================================

    persistent lastT dir pos Sbin nBinsDone lastProcessedBin ...
               kfX kfP kfInit nFeatCached ...
               dirLocked didReclass firstDirScore

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

        dirLocked = false;
        didReclass = false;
        firstDirScore = -Inf;
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

    %---------------------- Helper: cosine template classify ----------------
    function [bestDir, bestScore] = classifyDir(tMs)
        tUse = min(tMs, T);
        fr = mean(spikes(:, 1:tUse), 2);
        frNorm = sqrt(sum(fr.^2)) + 1e-12;
        frUnit = fr / frNorm;
        sims = modelParameters.dirTemplatesUnit.' * frUnit; % 8 x 1
        [bestScore, bestDir] = max(sims);
    end

    %---------------------- First direction classification (>=320 ms) -------
    if nBinsNow*binSizeMs >= 320 && ~dirLocked
        [dir, s1] = classifyDir(320);
        firstDirScore = s1;
        dirLocked = true;

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

    %---------------------- Optional one-time re-classification -------------
    reclassMs = modelParameters.reclassMs;
    if dirLocked && ~didReclass && (nBinsNow*binSizeMs >= reclassMs)
        [dir2, s2] = classifyDir(reclassMs);

        % Switch only if confidence improves enough
        % (prevents unnecessary flips that can hurt)
        if (dir2 ~= dir) && (s2 > firstDirScore + 0.03)
            dir = dir2;
        end
        didReclass = true;
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