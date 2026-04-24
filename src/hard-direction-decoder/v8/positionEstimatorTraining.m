function modelParameters = positionEstimatorTraining(training_data)
%==========================================================================
% positionEstimatorTraining
%
% Method:
%   - Direction classification: Linear SVM (ECOC) on 320ms firing-rate vector.
%   - Velocity decoding: per-direction ridge regression with feature standardization.
%   - Kalman filter parameters (per direction) are learned from training data:
%       * 6D constant-acceleration state: [x y vx vy ax ay]
%       * Process noise Q from empirical acceleration statistics
%       * Measurement noise R from ridge residual covariance
%
% Runtime cost: small (SVM once per trial, KF with 6x6 matrices per bin).
% Requires: Statistics and Machine Learning Toolbox for fitcecoc/templateLinear.
%==========================================================================

    %-------------------------- Cache: load if exists -----------------------
    cacheFile = 'modelParameters.mat';
    if exist(cacheFile, 'file') == 2
        S = load(cacheFile, 'modelParameters');
        if isfield(S, 'modelParameters')
            mp = S.modelParameters;
            requiredFields = {'binSizeMs','lagBins','dt','lambda', ...
                              'svmModel','velW','velB','muX','sigX','x0y0', ...
                              'kfQ','kfR'};
            ok = true;
            for i = 1:numel(requiredFields)
                if ~isfield(mp, requiredFields{i})
                    ok = false; break;
                end
            end
            if ok
                modelParameters = mp;
                return;
            end
        end
    end

    %----------------------------- Hyperparams ------------------------------
    binSizeMs = 20;
    lagBins   = 3;
    dt        = binSizeMs/1000;
    lambda    = 50;

    numNeurons = 98;
    numDirs    = 8;

    %--------------------- Containers for training data ---------------------
    % Direction SVM training (first 320ms)
    Xdir = [];  % N x 98
    ydir = [];  % N x 1

    % Velocity regression data
    Xall = cell(1, numDirs);
    Yall = cell(1, numDirs);

    % For learning KF Q from true trajectories (per direction)
    accAll = cell(1, numDirs);  % store [ax ay] samples (mm/s^2)

    % Initial position fallback
    x0y0Sum = zeros(2, numDirs);
    x0y0Cnt = zeros(1, numDirs);

    nTrials = size(training_data, 1);

    for tr = 1:nTrials
        for d = 1:numDirs
            spikes  = training_data(tr, d).spikes;          % 98 x T
            handPos = training_data(tr, d).handPos(1:2, :); % 2  x T
            T = size(spikes, 2);

            x0y0Sum(:, d) = x0y0Sum(:, d) + handPos(:, 1);
            x0y0Cnt(d)    = x0y0Cnt(d) + 1;

            % -------- Direction feature for SVM (unit-normalized fr) --------
            t320 = min(320, T);
            fr = mean(spikes(:, 1:t320), 2);
            frNorm = sqrt(sum(fr.^2)) + 1e-12;
            frUnit = (fr / frNorm).';
            Xdir = [Xdir; frUnit];
            ydir = [ydir; d];

            % -------- Bin spikes for regression and trajectory stats --------
            nBins = floor(T / binSizeMs);
            if nBins <= lagBins
                continue;
            end

            Sbin = zeros(numNeurons, nBins);
            for b = 1:nBins
                idx1 = (b-1)*binSizeMs + 1;
                idx2 = b*binSizeMs;
                Sbin(:, b) = sum(spikes(:, idx1:idx2), 2);
            end

            % Positions at bin ends
            P = zeros(2, nBins);
            for b = 1:nBins
                idx2 = b*binSizeMs;
                P(:, b) = handPos(:, idx2);
            end

            % True velocity and acceleration from positions
            Vtrue = diff(P, 1, 2) / dt;              % 2 x (nBins-1)
            if size(Vtrue,2) >= 2
                Atrue = diff(Vtrue, 1, 2) / dt;      % 2 x (nBins-2)
                accAll{d} = [accAll{d}; Atrue.'];    % append as rows
            end

            % Regression targets: velocity (mm/s)
            kStart = lagBins;
            kEnd   = nBins - 1;

            nSamples = kEnd - kStart + 1;
            nFeat    = numNeurons * lagBins;

            X = zeros(nSamples, nFeat);
            Y = zeros(nSamples, 2);

            row = 1;
            for k = kStart:kEnd
                feat = zeros(nFeat, 1);
                base = 1;
                for lag = 0:(lagBins-1)
                    feat(base:base+numNeurons-1) = Sbin(:, k-lag);
                    base = base + numNeurons;
                end
                X(row, :) = feat.';
                Y(row, :) = Vtrue(:, k).';  % Vtrue(:,k) aligns with step k->k+1
                row = row + 1;
            end

            Xall{d} = [Xall{d}; X];
            Yall{d} = [Yall{d}; Y];
        end
    end

    %---------------------- Train linear SVM (ECOC) -------------------------
    learner = templateLinear('Learner','svm', 'Regularization','ridge', 'Lambda',1e-4);
    svmModel = fitcecoc(Xdir, ydir, 'Learners', learner, 'Coding','onevsone');

    %---------------------- Initial position fallback -----------------------
    x0y0 = cell(1, numDirs);
    for d = 1:numDirs
        if x0y0Cnt(d) > 0
            x0y0{d} = x0y0Sum(:, d) / x0y0Cnt(d);
        else
            x0y0{d} = [0; 0];
        end
    end

    %------------------- Train ridge regressors + learn KF R ----------------
    velW = cell(1, numDirs);
    velB = cell(1, numDirs);
    muX  = cell(1, numDirs);
    sigX = cell(1, numDirs);
    kfR  = cell(1, numDirs);   % 2x2 measurement noise cov per direction
    kfQ  = cell(1, numDirs);   % 6x6 process noise cov per direction

    for d = 1:numDirs
        X = Xall{d};
        Y = Yall{d};
        nFeat = numNeurons * lagBins;

        if isempty(X)
            velW{d} = zeros(nFeat, 2);
            velB{d} = zeros(1, 2);
            muX{d}  = zeros(1, nFeat);
            sigX{d} = ones(1, nFeat);
            kfR{d}  = eye(2) * (200^2);
        else
            mu = mean(X, 1);
            sg = std(X, 0, 1);
            sg(sg < 1e-6) = 1;

            Xz = (X - mu) ./ sg;

            W = (Xz.'*Xz + lambda*eye(size(Xz,2))) \ (Xz.' * Y);
            b = mean(Y, 1);

            velW{d} = W;
            velB{d} = b;
            muX{d}  = mu;
            sigX{d} = sg;

            % Estimate measurement noise R from residuals on training data
            Yhat = Xz * W + b;
            E = Y - Yhat;               % residuals (mm/s)
            if size(E,1) >= 5
                R = cov(E);             % 2x2
            else
                R = eye(2) * (200^2);
            end
            % Regularize R to ensure positive definite
            R = R + eye(2) * 1e-3;
            kfR{d} = R;
        end

        %-------------------- Learn process noise Q from acceleration --------
        % We model acceleration as random walk noise in [ax, ay].
        % Build discrete-time Q for constant-acceleration model using
        % empirical acceleration covariance.
        if isempty(accAll{d}) || size(accAll{d},1) < 10
            Sa = diag([80^2, 80^2]);  % fallback accel variance (mm/s^2)^2
        else
            Sa = cov(accAll{d});      % 2x2 covariance of [ax, ay]
            Sa = Sa + eye(2)*1e-6;
        end

        % Discrete-time process noise for [x v a] in 1D:
        % q * [dt^5/20 dt^4/8 dt^3/6; dt^4/8 dt^3/3 dt^2/2; dt^3/6 dt^2/2 dt]
        % For 2D, block-diagonal with Sa scaling.
        G = [dt^5/20, dt^4/8, dt^3/6;
             dt^4/8,  dt^3/3, dt^2/2;
             dt^3/6,  dt^2/2, dt];

        % Build 6x6 Q as kron(Sa, G) with ordering [x y vx vy ax ay]
        % Expand carefully:
        Q = zeros(6,6);
        % x/v/a for X-dimension
        Q([1 3 5],[1 3 5]) = Sa(1,1) * G;
        % x/v/a for Y-dimension
        Q([2 4 6],[2 4 6]) = Sa(2,2) * G;
        % Cross-cov between X and Y dims (optional, use Sa off-diagonals)
        Q([1 3 5],[2 4 6]) = Sa(1,2) * G;
        Q([2 4 6],[1 3 5]) = Sa(2,1) * G;

        kfQ{d} = Q;
    end

    %------------------------------ Pack output -----------------------------
    modelParameters = struct();
    modelParameters.binSizeMs = binSizeMs;
    modelParameters.lagBins   = lagBins;
    modelParameters.dt        = dt;
    modelParameters.lambda    = lambda;

    modelParameters.svmModel  = svmModel;

    modelParameters.velW      = velW;
    modelParameters.velB      = velB;
    modelParameters.muX       = muX;
    modelParameters.sigX      = sigX;

    modelParameters.x0y0      = x0y0;

    modelParameters.kfQ       = kfQ;  % per-dir 6x6
    modelParameters.kfR       = kfR;  % per-dir 2x2

    try
        save(cacheFile, 'modelParameters', '-v7.3');
    catch
        save(cacheFile, 'modelParameters');
    end
end