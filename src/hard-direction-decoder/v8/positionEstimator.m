function [decodedPosX, decodedPosY, updatedModelParameters] = positionEstimator(test_data, modelParameters)
%==========================================================================
% positionEstimator
%
% Method:
%   - Direction classification (once per trial) via linear SVM (ECOC) on 320ms fr.
%   - Velocity regression: per-direction ridge (standardized lag-stacked spikes).
%   - Kalman filter: 6D constant-acceleration model:
%       state = [x; y; vx; vy; ax; ay]
%       measurement = [vx; vy] from regression
%     with per-direction learned Q and R.
%
% Fully causal and I/O compatible.
%==========================================================================

    persistent lastT dir pos Sbin nBinsDone lastProcessedBin ...
               kfX kfP kfInit nFeatCached dirDone

    spikes = test_data.spikes;
    T = size(spikes, 2);

    % Trial reset
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

        kfX = zeros(6,1);  % [x y vx vy ax ay]
        kfP = eye(6);
        kfInit = false;

        nFeatCached = [];
    end
    lastT = T;

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

    % Incremental binning
    for b = (nBinsDone+1):nBinsNow
        idx1 = (b-1)*binSizeMs + 1;
        idx2 = b*binSizeMs;
        Sbin(:, b) = sum(spikes(:, idx1:idx2), 2);
    end
    nBinsDone = nBinsNow;

    % Direction via SVM at >=320ms
    if ~dirDone && (nBinsNow*binSizeMs >= 320)
        t320 = min(320, T);
        fr = mean(spikes(:, 1:t320), 2);
        frNorm = sqrt(sum(fr.^2)) + 1e-12;
        frUnit = (fr / frNorm).'; % 1 x 98

        dir = predict(modelParameters.svmModel, frUnit);

        if isfield(test_data, 'startHandPos') && ~isempty(test_data.startHandPos)
            pos = double(test_data.startHandPos(:));
        else
            pos = modelParameters.x0y0{dir};
        end

        % Initialize 6D state
        kfX = [pos(1); pos(2); 0; 0; 0; 0];
        kfP = diag([50^2, 50^2, 600^2, 600^2, 500^2, 500^2]);
        kfInit = true;

        lastProcessedBin = 1;
        dirDone = true;
    end

    % 6D constant-acceleration transition matrix
    % x' = x + vx*dt + 0.5*ax*dt^2
    % vx' = vx + ax*dt
    A = eye(6);
    A(1,3) = dt;        A(1,5) = 0.5*dt*dt;
    A(2,4) = dt;        A(2,6) = 0.5*dt*dt;
    A(3,5) = dt;
    A(4,6) = dt;

    % Measurement: z = [vx; vy]
    H = zeros(2,6);
    H(1,3) = 1;
    H(2,4) = 1;

    I6 = eye(6);

    % Update per new bin
    if nBinsNow >= (lagBins + 1) && lastProcessedBin >= 1 && kfInit
        kStart = max(lagBins, lastProcessedBin);
        kEnd   = nBinsNow - 1;

        W  = modelParameters.velW{dir};
        b0 = modelParameters.velB{dir};
        mu = modelParameters.muX{dir};
        sg = modelParameters.sigX{dir};

        Q = modelParameters.kfQ{dir};  % 6x6
        R = modelParameters.kfR{dir};  % 2x2

        for k = kStart:kEnd
            % Build lag-stacked feature vector
            feat = zeros(nFeat, 1);
            base = 1;
            for lag = 0:(lagBins-1)
                feat(base:base+numNeurons-1) = Sbin(:, k-lag);
                base = base + numNeurons;
            end

            % Predict velocity measurement
            xrow = (feat.' - mu) ./ sg;
            vHat = (xrow * W) + b0;      % 1x2
            z = vHat(:);                 % 2x1

            % KF predict
            kfX = A * kfX;
            kfP = A * kfP * A.' + Q;

            % KF update
            S = H * kfP * H.' + R;       % 2x2
            K = (kfP * H.') / S;         % 6x2
            innov = z - (H * kfX);
            kfX = kfX + K * innov;
            kfP = (I6 - K * H) * kfP;

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