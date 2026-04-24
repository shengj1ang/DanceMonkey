function [decodedPosX, decodedPosY, updatedModelParameters] = positionEstimator(test_data, modelParameters)
%==========================================================================
% positionEstimator
%
% Method:
%   - Direction: SVM (ECOC) at 320ms on unit-normalized firing-rate vector.
%   - Measurement: ridge regressed velocity (vx, vy).
%   - Dynamics: 6D constant-acceleration KF with learned Q/R, using two-phase
%     noise model (early vs late).
%
% State: [x; y; vx; vy; ax; ay]
% Measurement: z = [vx; vy]
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

        kfX = zeros(6,1);
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

    % Direction classification at >=320ms (once)
    if ~dirDone && (nBinsNow*binSizeMs >= 320)
        t320 = min(320, T);
        fr = mean(spikes(:, 1:t320), 2);
        frNorm = sqrt(sum(fr.^2)) + 1e-12;
        frUnit = (fr / frNorm).';

        dir = predict(modelParameters.svmModel, frUnit);

        if isfield(test_data, 'startHandPos') && ~isempty(test_data.startHandPos)
            pos = double(test_data.startHandPos(:));
        else
            pos = modelParameters.x0y0{dir};
        end

        % Init state
        kfX = [pos(1); pos(2); 0; 0; 0; 0];
        kfP = diag([50^2, 50^2, 700^2, 700^2, 1200^2, 1200^2]);
        kfInit = true;

        lastProcessedBin = 1;
        dirDone = true;
    end

    % Dynamics matrix (constant acceleration)
    A = eye(6);
    A(1,3) = dt;        A(1,5) = 0.5*dt*dt;
    A(2,4) = dt;        A(2,6) = 0.5*dt*dt;
    A(3,5) = dt;
    A(4,6) = dt;

    % Measurement matrix for velocity
    H = zeros(2,6);
    H(1,3) = 1;
    H(2,4) = 1;

    I6 = eye(6);

    splitBin = modelParameters.splitBin;

    % Update per new bin
    if nBinsNow >= (lagBins + 1) && lastProcessedBin >= 1 && kfInit
        kStart = max(lagBins, lastProcessedBin);
        kEnd   = nBinsNow - 1;

        W  = modelParameters.velW{dir};
        b0 = modelParameters.velB{dir};
        mu = modelParameters.muX{dir};
        sg = modelParameters.sigX{dir};

        for k = kStart:kEnd
            % Choose early/late noise
            if k <= splitBin
                Q = modelParameters.kfQ_early{dir};
                R = modelParameters.kfR_early{dir};
            else
                Q = modelParameters.kfQ_late{dir};
                R = modelParameters.kfR_late{dir};
            end

            % Build lag-stacked features
            feat = zeros(nFeat, 1);
            base = 1;
            for lag = 0:(lagBins-1)
                feat(base:base+numNeurons-1) = Sbin(:, k-lag);
                base = base + numNeurons;
            end

            % Regress velocity measurement
            xrow = (feat.' - mu) ./ sg;
            vHat = (xrow * W) + b0;
            z = vHat(:);

            % KF predict
            kfX = A * kfX;
            kfP = A * kfP * A.' + Q;

            % KF update
            S = H * kfP * H.' + R;
            K = (kfP * H.') / S;
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