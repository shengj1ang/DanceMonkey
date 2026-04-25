function modelParameters = positionEstimatorTraining(training_data)
% Time-Dependent Low-Dimensional Neural Decoder Training
%
% Its job is to learn all parameters needed by positionEstimator.m.
%
% Main idea:
% 1. Convert spike trains into 20 ms firing-rate bins.
% 2. Smooth spike features causally, meaning only current and past bins are used.
% 3. Remove neurons with very low firing activity.
% 4. For each decoding time point, build a separate time-dependent model.
% 5. Use PCA to reduce feature dimensionality.
% 6. Use LDA and nearest-neighbour voting later to estimate direction confidence.
% 7. Train direction-conditioned regression models to predict hand position.
% 8. Store everything in modelParameters for fast online decoding.

    % ----------------------------
    % Hyperparameters
    % ----------------------------
    % binSize defines the length of each neural time bin in milliseconds.
    % The decoder works on binned spike counts rather than raw millisecond spikes.
    binSize = 20;

    % startTime is the first time point at which the decoder starts prediction.
    % In this BMI task, predictions begin at 320 ms.
    startTime = 320;

    % smoothMode records the smoothing type used in preprocessing.
    % The actual smoothing is implemented by the EMA code below.
    smoothMode = 'ema';

    % emaAlpha controls causal exponential moving average smoothing.
    % A larger value gives more weight to the current bin.
    % A smaller value gives more weight to previous bins.
    % This is causal because it never uses future spike data.
    emaAlpha = 0.45;

    % minMeanRate is used to remove neurons with extremely low activity.
    % Such neurons contribute mostly noise and can make the model less stable.
    minMeanRate = 0.5;

    % pcaVarKeep controls how much variance PCA should keep.
    % 0.85 means the smallest number of principal components explaining
    % at least 85% of the variance will be retained.
    pcaVarKeep = 0.85;

    % ldaDim is the target number of LDA dimensions used after PCA.
    % LDA makes different reaching directions easier to separate.
    ldaDim = 7;

    % ridgeLambda controls the strength of ridge regularisation.
    % Ridge regression prevents the regression weights from becoming too large,
    % which helps reduce overfitting.
    ridgeLambda = 10;

    % nTrials is the number of repeated trials per direction.
    % nDirs is the number of movement directions.
    [nTrials, nDirs] = size(training_data);

    % ----------------------------
    % Determine shortest valid trial length
    % ----------------------------
    % Different trials may have slightly different recording lengths.
    % To keep every training sample aligned, all trials are truncated to
    % the shortest trial length.
    allLens = zeros(nTrials, nDirs);
    for i = 1:nTrials
        for d = 1:nDirs
            allLens(i,d) = size(training_data(i,d).spikes, 2);
        end
    end

    % minLen is the shortest spike recording length among all trials.
    minLen = min(allLens(:));

    % maxBinCount is the maximum number of complete 20 ms bins that can be
    % used for every trial.
    maxBinCount = floor(minLen / binSize);

    % startBin converts the start time from milliseconds into a bin index.
    startBin = floor(startTime / binSize);
    if startBin < 1; startBin = 1; end

    % timeBins stores all causal time points at which separate models are built.
    % Example: if startBin = 16, the first model uses bins 1 to 16,
    % the next model uses bins 1 to 17, and so on.
    timeBins = startBin:maxBinCount;
    nTimeModels = numel(timeBins);

    % ----------------------------
    % Preprocess all training trials
    % ----------------------------
    % proc stores the processed spike features for every trial and direction.
    % posAtBins stores the hand position sampled at every bin boundary.
    proc = cell(nTrials, nDirs);
    posAtBins = cell(nTrials, nDirs);

    for i = 1:nTrials
        for d = 1:nDirs
            % Keep only the part of the spike train that fits into complete bins.
            spikes = training_data(i,d).spikes(:, 1:maxBinCount*binSize);

            % Replace NaN spike values with zero so they do not break training.
            spikes(isnan(spikes)) = 0;

            % Convert raw spikes into smoothed firing-rate features.
            proc{i,d} = preprocessSpikes(spikes, binSize, emaAlpha);
            
            % Use only x and y hand position.
            hp = training_data(i,d).handPos(1:2, :);

            % Fill possible missing position values.
            hp = fillHandTrajectory(hp);

            % Truncate hand trajectory to match the spike data length.
            hp = hp(:, 1:maxBinCount*binSize);

            % Sample hand position at the end of each 20 ms bin.
            % These positions are used as regression targets.
            posAtBins{i,d} = hp(:, binSize:binSize:maxBinCount*binSize);
        end
    end

    % Number of recorded neurons before filtering.
    nNeurons = size(proc{1,1}, 1);

    % ----------------------------
    % Remove low-rate neurons
    % ----------------------------
    % Compute the average firing rate of every neuron across all trials
    % and all directions. Neurons below minMeanRate are removed.
    globalRate = zeros(nNeurons, 1);

    for n = 1:nNeurons
        acc = 0;
        cnt = 0;

        for i = 1:nTrials
            for d = 1:nDirs
                % Accumulate all firing-rate values of neuron n.
                acc = acc + sum(proc{i,d}(n, :));

                % Count how many time bins contributed.
                cnt = cnt + size(proc{i,d}, 2);
            end
        end

        % Average activity of this neuron.
        globalRate(n) = acc / cnt;
    end

    % Find neurons that are too inactive.
    dropIdx = find(globalRate < minMeanRate);

    % keepIdx stores the neurons that remain in the decoder.
    keepIdx = setdiff(1:nNeurons, dropIdx);
    keptNeuronCount = numel(keepIdx);

    % ----------------------------
    % Build time-dependent classification models
    % ----------------------------
    % These models are used later to estimate how likely each direction is.
    % A separate model is trained for each time point because the amount of
    % available neural history changes over time.
    classModel = struct([]);

    for tIdx = 1:nTimeModels
        % bNow is the current number of available bins.
        bNow = timeBins(tIdx);

        % Xcls contains one feature vector per training trial.
        % Each feature vector concatenates all kept neurons over bins 1:bNow.
        Xcls = zeros(keptNeuronCount * bNow, nTrials * nDirs);

        % ycls stores the direction label for each feature vector.
        ycls = zeros(1, nTrials * nDirs);

        col = 1;

        for d = 1:nDirs
            for i = 1:nTrials
                % Extract neural features for this trial and direction.
                featMat = proc{i,d}(keepIdx, 1:bNow);

                % Flatten neuron-by-time matrix into one long vector.
                Xcls(:, col) = featMat(:);

                % Store direction label.
                ycls(col) = d;

                col = col + 1;
            end
        end

        % Compute the mean feature vector for centering.
        mu = mean(Xcls, 2);

        % Center the data before PCA.
        X0 = Xcls - mu;

        % PCA reduces the high-dimensional neural history into a compact space.
        [Wpca, scorePca, ~, npc] = runPCAfromCov(X0, pcaVarKeep);

        % LDA projects PCA features into a more direction-discriminative space.
        Wlda = runLDA(scorePca, ycls, ldaDim);
        
        % Store the mean vector needed to centre test features.
        classModel(tIdx).mu = mu;

        % Important speed optimisation:
        % During testing, the feature would normally go through PCA first,
        % then LDA. Since both are linear projections, they can be multiplied
        % together in advance. This makes online decoding faster.
        classModel(tIdx).W_cls = Wpca(:, 1:npc) * Wlda; 

        % Store the training samples after PCA+LDA projection.
        % During testing, the new sample will be compared with these points.
        classModel(tIdx).Ztrain = Wlda' * scorePca;

        % Store direction labels for nearest-neighbour voting.
        classModel(tIdx).labels = ycls;
    end

    % ----------------------------
    % Train direction-conditioned regression models
    % ----------------------------
    % For every direction and every time point, train one regression model.
    % Each model predicts the x and y hand position from neural features.
    regModel = struct([]);

    for d = 1:nDirs
        for tIdx = 1:nTimeModels
            % bNow is the number of spike bins available at this time point.
            bNow = timeBins(tIdx);

            % Xreg contains one row per trial.
            % Each row is the flattened neural feature vector.
            Xreg = zeros(nTrials, keptNeuronCount * bNow);

            % tx and ty are target x and y positions.
            tx = zeros(nTrials,1);
            ty = zeros(nTrials,1);

            for i = 1:nTrials
                % Extract neural activity up to the current bin.
                featMat = proc{i,d}(keepIdx, 1:bNow);

                % Flatten into one row vector for regression.
                Xreg(i,:) = featMat(:)';

                % The regression target is the hand position at the same bin.
                tx(i) = posAtBins{i,d}(1, bNow);
                ty(i) = posAtBins{i,d}(2, bNow);
            end

            % Mean feature vector for centering regression inputs.
            mx = mean(Xreg, 1);

            % Center regression features.
            Xc = Xreg - mx;
            
            % Apply PCA to regression features.
            % Xc' is used because runPCAfromCov expects features as rows.
            [WpcaReg, scoreReg, ~, npcReg] = runPCAfromCov(Xc', pcaVarKeep);

            % Each row of P corresponds to one trial in PCA space.
            P = scoreReg';

            % Train ridge regression for x and y separately.
            % The target is also centered by subtracting its mean.
            bx = ridgeSolve(P, tx - mean(tx), ridgeLambda);
            by = ridgeSolve(P, ty - mean(ty), ridgeLambda);
            
            % Store the feature mean for test-time centering.
            regModel(d, tIdx).mx = mx;

            % Important speed optimisation:
            % Normally, test features would first be projected by PCA and then
            % multiplied by ridge weights. Because both steps are linear,
            % they are combined into one direct weight vector here.
            regModel(d, tIdx).W_x = WpcaReg(:, 1:npcReg) * bx; 
            regModel(d, tIdx).W_y = WpcaReg(:, 1:npcReg) * by;

            % Store the mean x and y positions.
            % These are added back after predicting the centered position.
            regModel(d, tIdx).x0 = mean(tx);
            regModel(d, tIdx).y0 = mean(ty);
        end
    end

    % ----------------------------
    % Package all trained information
    % ----------------------------
    % modelParameters is the only object passed into positionEstimator.m.
    % Therefore, everything needed during online testing must be stored here.
    modelParameters = struct;
    modelParameters.binSize = binSize;
    modelParameters.startTime = startTime;
    modelParameters.startBin = startBin;
    modelParameters.maxBinCount = maxBinCount;
    modelParameters.timeBins = timeBins;
    modelParameters.nDirs = nDirs;
    modelParameters.keepIdx = keepIdx;
    modelParameters.classModel = classModel;
    modelParameters.regModel = regModel;
    modelParameters.smoothMode = smoothMode;
    modelParameters.emaAlpha = emaAlpha;

    % cumScore is used during online testing to remember direction confidence
    % from previous calls within the same trial.
    modelParameters.cumScore = [];
end

% =========================================================
% Helper functions
% =========================================================

function rate = preprocessSpikes(spikes, binSize, emaAlpha)
% Convert raw spike trains into smoothed firing-rate features.
%
% Input:
%   spikes   - raw spike matrix, neurons x time
%   binSize  - number of milliseconds per bin
%   emaAlpha - smoothing coefficient
%
% Output:
%   rate     - smoothed firing-rate matrix, neurons x bins

    [n, T] = size(spikes);

    % Number of complete bins available.
    nb = floor(T / binSize);

    % rate initially stores spike counts per bin.
    rate = zeros(n, nb);

    for b = 1:nb
        % Sum spikes within the current 20 ms bin.
        rate(:,b) = sum(spikes(:,(b-1)*binSize+1 : b*binSize), 2);
    end

    % Square-root transform reduces the effect of very large spike counts
    % and makes the spike-count distribution less skewed.
    rate = sqrt(rate);
    
    % Apply causal exponential moving average smoothing.
    out = zeros(size(rate));

    % The first bin has no previous bin, so it is copied directly.
    out(:,1) = rate(:,1);

    for b = 2:nb
        % Current smoothed value = part current bin + part previous smoothed bin.
        % No future bin is used, so this is valid for causal decoding.
        out(:,b) = emaAlpha * rate(:,b) + (1-emaAlpha) * out(:,b-1);
    end

    % Convert spike count per bin into firing rate per second.
    rate = out / (binSize/1000);
end

function hp = fillHandTrajectory(hp)
% Fill missing hand-position values.
%
% Some hand-position samples may be NaN. Regression cannot train with NaN
% targets, so missing values are replaced using neighbouring valid values.

    for r = 1:size(hp,1)
        % Work on one coordinate at a time: x first, then y.
        v = hp(r,:);

        % If this coordinate has no missing values, keep it unchanged.
        if all(~isnan(v)); hp(r,:) = v; continue; end

        % Forward fill:
        % if a value is missing, copy the previous available value.
        for i = 2:numel(v)
            if isnan(v(i)); v(i) = v(i-1); end
        end

        % Backward fill:
        % if missing values remain at the beginning, copy the next value.
        for i = numel(v)-1:-1:1
            if isnan(v(i)); v(i) = v(i+1); end
        end

        % If the entire row was NaN, replace remaining NaNs with zero.
        v(isnan(v)) = 0;

        % Store the cleaned trajectory coordinate.
        hp(r,:) = v;
    end
end

function [W, score, latent, npc] = runPCAfromCov(X, varKeep)
% Compute PCA and keep enough components to explain varKeep variance.
%
% X is expected to have features as rows and samples as columns.
% W contains PCA directions in the original feature space.
% score contains the low-dimensional PCA representation.
% latent contains retained eigenvalues.
% npc is the number of selected principal components.

    % Compute sample covariance-like matrix in sample space.
    % This is efficient when the feature dimension is very large.
    C = X' * X;

    % Eigen-decomposition of the covariance matrix.
    [V,D] = eig(C);

    % Extract eigenvalues.
    eigvals = diag(D);

    % Sort eigenvalues from largest to smallest.
    [eigvals, order] = sort(eigvals, 'descend');
    V = V(:, order);

    % Remove numerically tiny components.
    valid = eigvals > 1e-10;
    eigvals = eigvals(valid);
    V = V(:, valid);

    % Compute cumulative explained variance ratio.
    ratio = cumsum(eigvals) / sum(eigvals);

    % Select the smallest number of PCs that reaches the threshold.
    npc = find(ratio >= varKeep, 1, 'first');

    % Fallback in case no component reaches the threshold.
    if isempty(npc); npc = numel(eigvals); end

    % Keep only selected PCA directions.
    V = V(:,1:npc);
    latent = eigvals(1:npc);

    % Convert sample-space eigenvectors back to feature-space PCA directions.
    W = X * V .* (1 ./ sqrt(latent(:)')) ;

    % Project original data into PCA space.
    score = W' * X;
end

function Wlda = runLDA(X, y, outDim)
% Compute Linear Discriminant Analysis projection.
%
% PCA reduces dimensionality, while LDA tries to make classes easier to
% separate. Here, classes correspond to movement directions.

    % Unique class labels.
    cls = unique(y);

    % p is the PCA feature dimension.
    p = size(X,1);

    % Global mean across all samples.
    mu = mean(X,2);

    % Sw measures within-class scatter.
    % Sb measures between-class scatter.
    Sw = zeros(p,p);
    Sb = zeros(p,p);

    for k = 1:numel(cls)
        % Select samples belonging to class k.
        idx = (y == cls(k));
        Xk = X(:,idx);

        % Mean of class k.
        muk = mean(Xk,2);

        % Add within-class scatter.
        Dk = Xk - muk;
        Sw = Sw + Dk * Dk';

        % Add between-class scatter.
        diff = muk - mu;
        Sb = Sb + sum(idx) * (diff * diff');
    end

    % Solve the generalized LDA eigenvalue problem.
    % A small identity term improves numerical stability.
    [V,D] = eig(pinv(Sw + 1e-6*eye(p)) * Sb);

    % Sort LDA directions by discriminative strength.
    evals = real(diag(D));
    [~, order] = sort(evals, 'descend');

    % Do not request more dimensions than available.
    outDim = min(outDim, size(V,2));

    % Return the leading LDA directions.
    Wlda = real(V(:, order(1:outDim)));
end

function b = ridgeSolve(X, y, lambda)
% Solve ridge regression.
%
% The objective is:
%   min ||Xb - y||^2 + lambda * ||b||^2
%
% The lambda term prevents unstable or overly large regression weights.

    p = size(X,2);
    b = (X' * X + lambda * eye(p)) \ (X' * y);
end