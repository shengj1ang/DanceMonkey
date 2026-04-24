function [decodedPosX, decodedPosY, updatedModelParameters] = positionEstimator(test_data, modelParameters)
%==========================================================================
% positionEstimator
%
% Method #6:
%   - Direction classification (once) via cosine similarity to templates.
%   - Velocity: standardize features -> PCA projection -> ridge regression.
%   - Position: light Kalman filter on [x,y,vx,vy] using predicted velocity
%     as measurement.
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
        fr = mean(spikes(:, 1:tClass), 2);

        frNorm = sqrt(sum(fr.^2)) + 1e-12;
        frUnit = fr / frNorm;

        sims = modelParameters.dirTemplatesUnit.' * frUnit;
        [~, dir] = max(sims);

        if isfield(test_data, 'startHandPos') && ~isempty(test_data.startHandPos)
            pos = double(test_data.startHandPos(:));
        else
            pos = modelParameters.x0y0{dir};
        end

        kfX = [pos(1); pos(2); 0; 0];
        kfP = diag([50^2, 50^2, 600^2, 600^2]);
        kfInit = true;

        lastProcessedBin = 1;
    end

    %---------------------- Kalman model (same light cost) ------------------
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

        mu = modelParameters.muX{dir};
        sg = modelParameters.sigX{dir};
        Vr = modelParameters.pcaV{dir};    % nFeat x r
        W  = modelParameters.velW{dir};    % r x 2
        b0 = modelParameters.velB{dir};    % 1 x 2
        r  = size(Vr, 2);

        for k = kStart:kEnd
            feat = zeros(nFeat, 1);
            base = 1;
            for lag = 0:(lagBins-1)
                feat(base:base+numNeurons-1) = Sbin(:, k-lag);
                base = base + numNeurons;
            end

            xrow = (feat.' - mu) ./ sg;    % 1 x nFeat
            zPCA = xrow * Vr;              % 1 x r
            vHat = zPCA * W + b0;          % 1 x 2
            zMeas = vHat(:);               % 2 x 1

            % Predict
            kfX = A * kfX;
            kfP = A * kfP * A.' + Q;

            % Update
            S = H * kfP * H.' + R;
            K = (kfP * H.') / S;
            innov = zMeas - (H * kfX);
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