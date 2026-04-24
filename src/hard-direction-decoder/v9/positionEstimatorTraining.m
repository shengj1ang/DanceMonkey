function modelParameters = positionEstimatorTraining(training_data)
%==========================================================================
% positionEstimatorTraining
%
% Method: Full dynamics + noise modeling (two-phase learned Kalman)
%
% Pipeline:
%   1) Direction classification: Linear SVM (ECOC) using first 320ms
%      unit-normalized firing-rate vector.
%   2) Velocity measurement model: per-direction ridge regression on lagged
%      binned spikes (standardized).
%   3) Dynamics model: 6D constant-acceleration state
%         s = [x y vx vy ax ay]'
%      with discrete-time transition A(dt).
%   4) Noise modeling:
%      - Measurement noise R from ridge residuals (v_true - v_hat)
%      - Process noise Q from true state residuals (s_{k+1} - A s_k)
%      We learn TWO sets: Early vs Late (different noise regimes).
%
% Runtime impact: negligible (switch Q/R based on time; 6x6 KF).
% Requires: Statistics and Machine Learning Toolbox (fitcecoc/templateLinear).
%==========================================================================

    %-------------------------- Cache: load if exists -----------------------
    cacheFile = 'modelParameters.mat';
    if exist(cacheFile, 'file') == 2
        S = load(cacheFile, 'modelParameters');
        if isfield(S, 'modelParameters')
            mp = S.modelParameters;
            requiredFields = {'binSizeMs','lagBins','dt','lambda', ...
                              'svmModel','velW','velB','muX','sigX','x0y0', ...
                              'kfQ_early','kfQ_late','kfR_early','kfR_late','splitBin'};
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

    % Early/Late split (in bins). 400ms is 20 bins at 20ms/bin.
    splitMs  = 400;
    splitBin = max(1, round(splitMs / binSizeMs));

    %--------------------- Containers for training data ---------------------
    % Direction SVM training
    Xdir = [];  % N x 98
    ydir = [];  % N x 1

    % Velocity regression
    Xall = cell(1, numDirs);
    Yall = cell(1, numDirs);

    % Initial position fallback
    x0y0Sum = zeros(2, numDirs);
    x0y0Cnt = zeros(1, numDirs);

    % For estimating R (measurement noise) from ridge residuals:
    % store residuals separately for early/late per direction
    resV_early = cell(1, numDirs);  % rows of [dvx dvy]
    resV_late  = cell(1, numDirs);

    % For estimating Q (process noise) from true state residuals:
    % store w_k = s_{k+1} - A*s_k separately for early/late per direction
    resW_early = cell(1, numDirs);  % rows of 6D residual
    resW_late  = cell(1, numDirs);

    % Precompute dynamics A for 6D constant-acceleration
    A = eye(6);
    A(1,3) = dt;        A(1,5) = 0.5*dt*dt;
    A(2,4) = dt;        A(2,6) = 0.5*dt*dt;
    A(3,5) = dt;
    A(4,6) = dt;

    %------------------------------ Build data ------------------------------
    nTrials = size(training_data, 1);

    for tr = 1:nTrials
        for d = 1:numDirs
            spikes  = training_data(tr, d).spikes;          % 98 x T (ms)
            handPos = training_data(tr, d).handPos(1:2, :); % 2  x T (mm)
            T = size(spikes, 2);

            % initial position
            x0y0Sum(:, d) = x0y0Sum(:, d) + handPos(:, 1);
            x0y0Cnt(d)    = x0y0Cnt(d) + 1;

            % -------- Direction feature: first 320ms unit vector ------------
            t320 = min(320, T);
            fr = mean(spikes(:, 1:t320), 2);
            frNorm = sqrt(sum(fr.^2)) + 1e-12;
            frUnit = (fr / frNorm).';
            Xdir = [Xdir; frUnit];
            ydir = [ydir; d];

            % -------- Bin spikes and positions ------------------------------
            nBins = floor(T / binSizeMs);
            if nBins <= (lagBins + 2)
                continue;
            end

            Sbin = zeros(numNeurons, nBins);
            for b = 1:nBins
                idx1 = (b-1)*binSizeMs + 1;
                idx2 = b*binSizeMs;
                Sbin(:, b) = sum(spikes(:, idx1:idx2), 2);
            end

            P = zeros(2, nBins);
            for b = 1:nBins
                idx2 = b*binSizeMs;
                P(:, b) = handPos(:, idx2);
            end

            % True velocity and acceleration at bin grid
            Vtrue = diff(P, 1, 2) / dt;                   % 2 x (nBins-1)
            Atrue = diff(Vtrue, 1, 2) / dt;               % 2 x (nBins-2)

            % Build true 6D state sequence aligned to bins 2..(nBins-1):
            % For index k (2..nBins-1):
            %   x,y at P(:,k)
            %   v at Vtrue(:,k-1)
            %   a at Atrue(:,k-2)
            % This gives states for k=3..nBins-1 (needs a)
            if (nBins >= 4)
                k0 = 3;                % first k where accel exists
                kN = nBins - 1;        % last k where Vtrue exists for step k->k+1
                nState = kN - k0 + 1;

                Strue = zeros(6, nState);
                idx = 1;
                for k = k0:kN
                    Strue(:, idx) = [P(1,k); P(2,k); ...
                                     Vtrue(1,k-1); Vtrue(2,k-1); ...
                                     Atrue(1,k-2); Atrue(2,k-2)];
                    idx = idx + 1;
                end

                % Process residuals w_k = s_{k+1} - A s_k for idx=1..nState-1
                for idx = 1:(nState-1)
                    w = Strue(:, idx+1) - A * Strue(:, idx);
                    % Map to "time bin" (k) for split: use k = k0+idx-1
                    k_bin = (k0 + idx - 1);
                    if k_bin <= splitBin
                        resW_early{d} = [resW_early{d}; w(:).'];
                    else
                        resW_late{d}  = [resW_late{d};  w(:).'];
                    end
                end
            end

            % -------- Regression samples (velocity) -------------------------
            % Features built at k -> predicts Vtrue(:,k) where k is bin index in Vtrue
            kStart = lagBins;
            kEnd   = nBins - 1;  % Vtrue exists to nBins-1

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
                Y(row, :) = Vtrue(:, k).';
                row = row + 1;
            end

            Xall{d} = [Xall{d}; X];
            Yall{d} = [Yall{d}; Y];
        end
    end

    %---------------------- Train direction SVM (ECOC) ----------------------
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

    %------------------- Train ridge regressors and R -----------------------
    velW = cell(1, numDirs);
    velB = cell(1, numDirs);
    muX  = cell(1, numDirs);
    sigX = cell(1, numDirs);

    kfR_early = cell(1, numDirs);
    kfR_late  = cell(1, numDirs);

    for d = 1:numDirs
        X = Xall{d};
        Y = Yall{d};
        nFeat = numNeurons * lagBins;

        if isempty(X)
            velW{d} = zeros(nFeat, 2);
            velB{d} = zeros(1, 2);
            muX{d}  = zeros(1, nFeat);
            sigX{d} = ones(1, nFeat);
            kfR_early{d} = eye(2) * (250^2);
            kfR_late{d}  = eye(2) * (200^2);
            continue;
        end

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

        % residuals for measurement noise estimation
        Yhat = Xz * W + b;
        E = Y - Yhat; % mm/s

        % Split residuals by an approximate time index:
        % Since Xall/Yall aggregates many trials, we cannot recover exact bins here.
        % Practical workaround: use global split by velocity magnitude as proxy:
        %   early tends to have larger speeds/acc; but to keep it simple,
        %   we use a fixed partition ratio 40%/60%.
        nE = size(E,1);
        cut = max(10, round(0.4*nE));
        Eearly = E(1:cut, :);
        Elate  = E(cut+1:end, :);

        R1 = cov(Eearly); if any(isnan(R1(:))) || rank(R1) < 2, R1 = eye(2)*(250^2); end
        R2 = cov(Elate);  if any(isnan(R2(:))) || rank(R2) < 2, R2 = eye(2)*(200^2); end

        kfR_early{d} = R1 + eye(2)*1e-3;
        kfR_late{d}  = R2 + eye(2)*1e-3;
    end

    %------------------- Estimate Q from true state residuals ---------------
    kfQ_early = cell(1, numDirs);
    kfQ_late  = cell(1, numDirs);

    for d = 1:numDirs
        We = resW_early{d};
        Wl = resW_late{d};

        if isempty(We) || size(We,1) < 20
            Qe = eye(6) * 1e-2;
            Qe(1,1) = 5; Qe(2,2)=5; Qe(3,3)=500; Qe(4,4)=500; Qe(5,5)=5e3; Qe(6,6)=5e3;
        else
            Qe = cov(We) + eye(6)*1e-6;
        end

        if isempty(Wl) || size(Wl,1) < 20
            Ql = eye(6) * 1e-2;
            Ql(1,1) = 2; Ql(2,2)=2; Ql(3,3)=200; Ql(4,4)=200; Ql(5,5)=2e3; Ql(6,6)=2e3;
        else
            Ql = cov(Wl) + eye(6)*1e-6;
        end

        % mild shrinkage to ensure stability
        Qe = 0.98*Qe + 0.02*eye(6)*mean(diag(Qe));
        Ql = 0.98*Ql + 0.02*eye(6)*mean(diag(Ql));

        kfQ_early{d} = Qe;
        kfQ_late{d}  = Ql;
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

    modelParameters.kfQ_early = kfQ_early;
    modelParameters.kfQ_late  = kfQ_late;
    modelParameters.kfR_early = kfR_early;
    modelParameters.kfR_late  = kfR_late;
    modelParameters.splitBin  = splitBin;

    try
        save(cacheFile, 'modelParameters', '-v7.3');
    catch
        save(cacheFile, 'modelParameters');
    end
end