function [decodedPosX, decodedPosY, updatedModelParameters] = positionEstimator(test_data, modelParameters)
%==========================================================================
% positionEstimator
%
% Causal online decoder for the continuous-position competition.
%
% IMPORTANT COMPATIBILITY NOTE
%   The provided test script expects that if this function has 3 outputs,
%   the 3rd output is an UPDATED modelParameters struct (it will overwrite
%   modelParameters with it at every time step).
%
%   Therefore this implementation returns:
%       [x, y, updatedModelParameters]
%   and stores any extra information (e.g., predicted angle) INSIDE the
%   returned struct:
%       updatedModelParameters.predictedAngle
%
% Inputs
%   test_data.spikes         : [98 x T] spikes up to current time (ms bins)
%   test_data.decodedHandPos : [2 x N] previously decoded positions (may be empty)
%   test_data.startHandPos   : [2 x 1] starting hand position (mm)
%   modelParameters          : struct produced by positionEstimatorTraining
%
% Outputs
%   decodedPosX, decodedPosY : decoded hand position at current time (mm)
%   updatedModelParameters   : same struct, optionally updated with state
%
% Decoder design (fast + stable)
%   1) Direction classification using first 320 ms spike-rate template match.
%   2) Per-direction ridge regression to predict velocity from recent binned
%      spike counts (with short lag history).
%   3) Integrate velocity in 20 ms steps to obtain position.
%
% Causality
%   All computations use only spikes provided up to time T, and only update
%   position when a full 20 ms bin becomes available.
%==========================================================================

    %--------------------------- Persistent state ---------------------------
    persistent lastT lastProcessedBin dir pos Sbin nBinsDone nFeatCached

    spikes = test_data.spikes;
    T = size(spikes, 2);

    % Detect trial reset: first call OR time goes backwards (new trial)
    if isempty(lastT) || T < lastT
        lastT = 0;
        lastProcessedBin = 0;
        dir = 1;

        % Initialise position using the true start hand position if provided,
        % otherwise fall back to [0;0].
        if isfield(test_data, 'startHandPos') && ~isempty(test_data.startHandPos)
            pos = double(test_data.startHandPos(:));
        else
            pos = [0; 0];
        end

        Sbin = [];
        nBinsDone = 0;
        nFeatCached = [];
    end
    lastT = T;

    %------------------------------ Parameters ------------------------------
    binSizeMs  = modelParameters.binSizeMs;     % 20
    lagBins    = modelParameters.lagBins;       % e.g. 3
    dt         = modelParameters.dt;            % 0.02
    numNeurons = 98;

    % Cache feature length to avoid recomputing each call
    if isempty(nFeatCached)
        nFeatCached = numNeurons * lagBins;
    end
    nFeat = nFeatCached;

    % Number of complete 20 ms bins available at this call
    nBinsNow = floor(T / binSizeMs);

    % Allocate/extend binned spike matrix if needed
    if isempty(Sbin)
        Sbin = zeros(numNeurons, max(nBinsNow, 1));
    elseif size(Sbin, 2) < nBinsNow
        Sbin(:, end+1:nBinsNow) = 0;
    end

    %------------------------- Update binned spikes -------------------------
    % Only compute newly available bins since last call (fast incremental).
    for b = (nBinsDone+1):nBinsNow
        idx1 = (b-1)*binSizeMs + 1;
        idx2 = b*binSizeMs;
        Sbin(:, b) = sum(spikes(:, idx1:idx2), 2);
    end
    nBinsDone = nBinsNow;

    %---------------------- Direction classification (>=320 ms) -------------
    % Run once per trial (when we first have >=320 ms of data).
    if nBinsNow*binSizeMs >= 320 && lastProcessedBin == 0
        tClass = min(320, T);
        fr = mean(spikes(:, 1:tClass), 2);  % 98 x 1

        % Cosine similarity against pre-normalised templates
        frNorm = sqrt(sum(fr.^2)) + 1e-12;
        frUnit = fr / frNorm;

        sims = modelParameters.dirTemplates.' * frUnit; % 8 x 1
        [~, dir] = max(sims);

        % Reset position to a stable initial point:
        % Prefer the true provided startHandPos to avoid bias.
        if isfield(test_data, 'startHandPos') && ~isempty(test_data.startHandPos)
            pos = double(test_data.startHandPos(:));
        else
            pos = modelParameters.x0y0{dir};
        end

        % Mark that we have initialised direction and can start integrating
        lastProcessedBin = 1;
    end

    %---------------------- Integrate velocity updates ----------------------
    % We only update when enough bins exist to build lag features.
    if nBinsNow >= (lagBins + 1) && lastProcessedBin >= 1
        kStart = max(lagBins, lastProcessedBin);
        kEnd   = nBinsNow - 1;

        % Preload direction-specific parameters once (speed)
        W   = modelParameters.velW{dir};    % nFeat x 2
        b0  = modelParameters.velB{dir};    % 1 x 2
        muX = modelParameters.muX{dir};     % 1 x nFeat

        for k = kStart:kEnd
            % Feature vector: [S(:,k); S(:,k-1); ...] (stacked)
            feat = zeros(nFeat, 1);
            base = 1;
            for lag = 0:(lagBins-1)
                feat(base:base+numNeurons-1) = Sbin(:, k-lag);
                base = base + numNeurons;
            end

            % Center features and predict velocity
            xrow = feat.' - muX;            % 1 x nFeat
            v    = xrow * W + b0;           % 1 x 2 (mm/s)

            % Integrate to get position at end of next bin
            pos = pos + (v.' * dt);
        end

        lastProcessedBin = nBinsNow;
    end

    %----------------------------- Output values ----------------------------
    decodedPosX = pos(1);
    decodedPosY = pos(2);

    % Store a representative angle inside the returned model struct
    % (so we keep the information without breaking the test script).
    predictedAngle = (dir - 1) * (pi/4);

    updatedModelParameters = modelParameters;
    updatedModelParameters.predictedAngle = predictedAngle;
    updatedModelParameters.currentDirection = dir;
end