function [decodedPosX, decodedPosY, updatedModelParameters] = positionEstimator(test_data, modelParameters)
%==========================================================================
% positionEstimator
%
% Method #2 (runtime-light):
%   1) Direction classification (once) via cosine similarity to 320ms templates.
%   2) Velocity regression via per-direction ridge model on lag-stacked binned spikes.
%   3) Light Kalman filter on state [x,y,vx,vy] using regressed [vx,vy] as measurement.
%
% Notes:
%   - Fully causal: uses only spikes available up to current time.
%   - Keeps the exact I/O signature expected by the test script.
%==========================================================================

    %--------------------------- Persistent state ---------------------------
    persistent lastT dir pos Sbin nBinsDone lastProcessedBin ...
               kfX kfP kfInit nFeatCached

    spikes = test_data.spikes;
    T = size(spikes, 2);

    %------------------------ Detect trial reset ----------------------------
    if isempty(lastT) || T < lastT
        lastT = 0;
        dir = 1;

        % Initialize position from true startHandPos if available
        if isfield(test_data, 'startHandPos') && ~isempty(test_data.startHandPos)
            pos = double(test_data.startHandPos(:));
        else
            pos = [0; 0];
        end

        Sbin = [];
        nBinsDone = 0;
        lastProcessedBin = 0;

        % Kalman state reset
        kfX = zeros(4,1);   % [x; y; vx; vy]
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

    % Number of complete bins available
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
        fr = mean(spikes(:, 1:tClass), 2);  % 98 x 1

        % Cosine similarity against unit templates
        frNorm = sqrt(sum(fr.^2)) + 1e-12;
        frUnit = fr / frNorm;

        sims = modelParameters.dirTemplatesUnit.' * frUnit; % 8 x 1
        [~, dir] = max(sims);

        % Initialize position
        if isfield(test_data, 'startHandPos') && ~isempty(test_data.startHandPos)
            pos = double(test_data.startHandPos(:));
        else
            pos = modelParameters.x0y0{dir};
        end

        % Initialize Kalman filter near start position
        kfX = [pos(1); pos(2); 0; 0];
        kfP = diag([50^2, 50^2, 600^2, 600^2]); % wide initial uncertainty
        kfInit = true;

        lastProcessedBin = 1;
    end

    %---------------------- Kalman filter model -----------------------------
    % State: [x; y; vx; vy]
    A = [1 0 dt 0;
         0 1 0 dt;
         0 0 1  0;
         0 0 0  1];

    % Process noise (keep smooth, but allow changes)
    % These values are intentionally conservative to avoid over-reacting.
    qPos = 1.5;    % mm
    qVel = 35;     % mm/s
    Q = diag([qPos^2, qPos^2, qVel^2, qVel^2]);

    % Measurement: z = [vx; vy]
    H = [0 0 1 0;
         0 0 0 1];

    % Measurement noise (trust regression moderately, not too much)
    rVel = 180;    % mm/s
    R = diag([rVel^2, rVel^2]);

    I4 = eye(4);

    %---------------------- Update per new bin ------------------------------
    if nBinsNow >= (lagBins + 1) && lastProcessedBin >= 1 && kfInit
        kStart = max(lagBins, lastProcessedBin);
        kEnd   = nBinsNow - 1;

        % Direction-specific regression params
        W  = modelParameters.velW{dir};   % nFeat x 2
        b0 = modelParameters.velB{dir};   % 1 x 2
        mu = modelParameters.muX{dir};    % 1 x nFeat
        sg = modelParameters.sigX{dir};   % 1 x nFeat

        for k = kStart:kEnd
            % Build lag-stacked feature vector
            feat = zeros(nFeat, 1);
            base = 1;
            for lag = 0:(lagBins-1)
                feat(base:base+numNeurons-1) = Sbin(:, k-lag);
                base = base + numNeurons;
            end

            % Standardize features and predict velocity
            xrow = (feat.' - mu) ./ sg;     % 1 x nFeat
            vHat = (xrow * W) + b0;         % 1 x 2 (mm/s)
            z = vHat(:);                     % 2 x 1

            % Kalman predict
            kfX = A * kfX;
            kfP = A * kfP * A.' + Q;

            % Kalman update
            S = H * kfP * H.' + R;          % 2 x 2
            K = (kfP * H.') / S;            % 4 x 2
            innov = z - (H * kfX);          % 2 x 1
            kfX = kfX + K * innov;
            kfP = (I4 - K * H) * kfP;

            % Output position from filtered state
            pos = kfX(1:2);
        end

        lastProcessedBin = nBinsNow;
    end

    %----------------------------- Output values ----------------------------
    decodedPosX = pos(1);
    decodedPosY = pos(2);

    updatedModelParameters = modelParameters;
    updatedModelParameters.currentDirection = dir;
    updatedModelParameters.predictedAngle = (dir - 1) * (pi/4);
end