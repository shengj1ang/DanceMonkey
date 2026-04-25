function [x, y, modelParameters] = positionEstimator(test_data, modelParameters)
% Online causal hand-position estimator.
%
% This function is called repeatedly during a test trial.
% At each call, it receives spike data only up to the current time.
% It must output the current predicted hand position x and y.
%
% Main online decoding steps:
% 1. Bin and smooth the available spike train.
% 2. Keep only the neurons selected during training.
% 3. Select the correct time-dependent model.
% 4. Estimate direction confidence using the PCA+LDA nearest-neighbour model.
% 5. Smooth direction confidence across time.
% 6. Predict hand position using all direction-specific regressors.
% 7. Combine those predictions using soft direction weights.
% 8. Apply lightweight causal temporal smoothing.

    % Load basic parameters learned during training.
    binSize = modelParameters.binSize;
    startBin = modelParameters.startBin;
    maxBinCount = modelParameters.maxBinCount;

    % Read spike data available up to the current test time.
    spikes = test_data.spikes;

    % Replace NaNs with zero to avoid numerical errors.
    spikes(isnan(spikes)) = 0;

    % Convert current spike length from milliseconds into number of bins.
    currBinCount = floor(size(spikes,2) / binSize);

    % Do not allow the test bin count to exceed the maximum length used in training.
    currBinCount = min(currBinCount, maxBinCount);

    % If no complete bin is available, return the known starting position.
    if currBinCount < 1
        x = test_data.startHandPos(1);
        y = test_data.startHandPos(2);
        return;
    end

    % Keep only complete bins.
    spikes = spikes(:, 1:currBinCount*binSize);

    % Apply the same preprocessing used during training:
    % binning, square-root transform, EMA smoothing, and rate conversion.
    rate = preprocessSpikes(spikes, binSize, modelParameters.emaAlpha);

    % Keep only the neurons that survived low-rate filtering during training.
    rate = rate(modelParameters.keepIdx, :);

    % Choose which time-dependent model to use.
    % The first model corresponds to startBin.
    % Later calls move forward one model per additional bin.
    if currBinCount <= startBin; tIdx = 1;
    else; tIdx = currBinCount - startBin + 1; end

    % Clamp tIdx to a valid range in case the test trial is longer than training.
    tIdx = max(1, min(tIdx, numel(modelParameters.timeBins)));

    % ---------------------------------------
    % Direction confidence estimation
    % ---------------------------------------
    % dirScore is a probability-like score over all movement directions.
    % It is computed using the current neural feature and the trained
    % PCA+LDA nearest-neighbour model.
    dirScore = getDirScore(rate, tIdx, modelParameters);
    
    % If decodedHandPos is empty, this is the first call of a new trial.
    % Therefore, reset accumulated direction confidence.
    if ~isfield(test_data, 'decodedHandPos') || isempty(test_data.decodedHandPos)
        modelParameters.cumScore = dirScore;
    else
        % Otherwise, update direction confidence using a simple temporal average.
        % This reduces sudden direction changes caused by noisy neural activity.
        modelParameters.cumScore = 0.5 * modelParameters.cumScore + 0.5 * dirScore;
    end

    % Sharpen the accumulated direction confidence.
    % Squaring makes large scores larger relative to small scores,
    % but still keeps all directions active.
    w = modelParameters.cumScore.^2; 

    % Normalise scores so all direction weights sum to 1.
    w = w / sum(w);

    % ---------------------------------------
    % Soft ensemble regression
    % ---------------------------------------
    % usedBins is the number of bins actually used for the current prediction.
    usedBins = min(currBinCount, modelParameters.timeBins(tIdx));

    % Extract all neural features available up to usedBins.
    feat = rate(:,1:usedBins);

    % Flatten neuron-by-time matrix into a single row vector.
    feat = feat(:)';

    % Initialise final predicted position.
    x_pred = 0; 
    y_pred = 0;
    
    % Instead of choosing only one direction, the decoder predicts a candidate
    % position from every direction-specific regression model.
    % The final prediction is a weighted average of all candidates.
    for d = 1:modelParameters.nDirs
        % Load regression model for direction d at time model tIdx.
        R = modelParameters.regModel(d, tIdx);

        % L protects against small length differences between test features
        % and the stored regression feature mean.
        L = min(numel(feat), numel(R.mx));
        
        % Centre the test feature using the training mean.
        f_d = feat(1:L) - R.mx(1:L);
        
        % Predict x and y using the precomputed linear regression weights.
        % PCA projection and ridge-regression weights were already combined
        % during training, so only a direct dot product is needed here.
        x_d = f_d * R.W_x(1:L) + R.x0;
        y_d = f_d * R.W_y(1:L) + R.y0;
        
        % Add this direction's prediction according to its confidence weight.
        x_pred = x_pred + w(d) * x_d;
        y_pred = y_pred + w(d) * y_d;
    end

    % Store ensemble prediction.
    x = x_pred;
    y = y_pred;

    % ---------------------------------------
    % Lightweight causal temporal smoothing
    % ---------------------------------------
    % This smoothing improves trajectory stability.
    % It is causal because it only uses the previous decoded position,
    % never any future ground-truth information.
    if isfield(test_data, 'decodedHandPos') && ~isempty(test_data.decodedHandPos)
        % Previous decoded hand position from the same test trial.
        prev = test_data.decodedHandPos(:, end);

        % Mostly trust the current prediction, but keep a small contribution
        % from the previous prediction to reduce jitter.
        x = 0.97 * x + 0.03 * prev(1);
        y = 0.97 * y + 0.03 * prev(2);
    else
        % At the first prediction step, there is no previous decoded position.
        % Blend with the known starting hand position to stabilise the initial output.
        x = 0.30 * x + 0.70 * test_data.startHandPos(1);
        y = 0.30 * y + 0.70 * test_data.startHandPos(2);
    end
end

% =========================================================
% Helper functions
% =========================================================

function score = getDirScore(rate, tIdx, modelParameters)
% Estimate direction confidence for the current test sample.
%
% This function:
% 1. Takes the current neural feature history.
% 2. Projects it into the trained PCA+LDA space.
% 3. Compares it with training samples in that space.
% 4. Uses the nearest samples to vote for movement direction.
% 5. Returns a normalised score over all directions.

    % Load the classification model for the current time point.
    C = modelParameters.classModel(tIdx);

    % Use the same number of bins as this time-dependent model was trained on.
    feat = rate(:,1:modelParameters.timeBins(tIdx));

    % Flatten the neuron-by-time matrix and subtract the training mean.
    feat = feat(:) - C.mu; 
    
    % Project the test feature into PCA+LDA space.
    % W_cls already combines PCA and LDA, so this is fast.
    zlda = C.W_cls' * feat; 

    % Compute squared Euclidean distance between the test point and
    % every stored training point in the PCA+LDA space.
    d2 = sum((C.Ztrain - zlda).^2, 1);

    % Sort training samples from nearest to farthest.
    [d2s, idx] = sort(d2, 'ascend');
    
    % Use the nearest 20 training samples for voting.
    k = 20;

    % Keep only the available nearest neighbours.
    idx = idx(1:min(k, numel(idx)));
    d2s = d2s(1:min(k, numel(d2s)));
    
    % Convert distances into weights.
    % Closer neighbours receive larger weights.
    % 1e-6 prevents division by zero.
    w_k = 1 ./ (d2s + 1e-6);

    % Initialise one score per direction.
    score = zeros(1, modelParameters.nDirs);

    % Accumulate weighted votes for each direction.
    for i = 1:numel(idx)
        c = C.labels(idx(i));
        score(c) = score(c) + w_k(i);
    end

    % Normalise scores so they sum to 1.
    score = score / sum(score);
end

function rate = preprocessSpikes(spikes, binSize, emaAlpha)
% Convert raw spike trains into smoothed firing-rate features.
%
% This must match the preprocessing used in training.
% The steps are:
% 1. Sum spikes in each 20 ms bin.
% 2. Apply square-root transform.
% 3. Apply causal EMA smoothing.
% 4. Convert counts into firing rates.

    [n, T] = size(spikes);

    % Number of complete bins.
    nb = floor(T / binSize);

    % Store spike counts per neuron per bin.
    rate = zeros(n, nb);

    for b = 1:nb
        % Sum spikes in the current bin.
        rate(:,b) = sum(spikes(:,(b-1)*binSize+1 : b*binSize), 2);
    end

    % Square-root transform reduces spike-count variance.
    rate = sqrt(rate);

    % Apply causal exponential moving average smoothing.
    out = zeros(size(rate));

    % First bin has no history.
    out(:,1) = rate(:,1);

    for b = 2:nb
        % Current smoothed bin depends only on current and previous bins.
        out(:,b) = emaAlpha * rate(:,b) + (1-emaAlpha) * out(:,b-1);
    end

    % Convert from spike count per bin to firing rate per second.
    rate = out / (binSize/1000);
end