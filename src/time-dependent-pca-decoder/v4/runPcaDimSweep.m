function results = runPcaDimSweep()
% Run different PCA dimensions and compare final RMSE.
%
% Make sure these files are in the same folder:
%   1. monkeydata_training.mat
%   2. positionEstimatorTraining_pcaDim.m
%   3. runPcaDimSweep.m
%
% Then run:
%   results = runPcaDimSweep();

    load monkeydata_training.mat

    rng(2013);
    ix = randperm(length(trial));

    trainingData = trial(ix(1:50),:);
    testData = trial(ix(51:end),:);

    % ===== Change PCA dimensions here =====
    pcaDims = [2 10 30 40 50 75 100 200 300 400 500];
    % =====================================

    results = table('Size', [numel(pcaDims), 4], ...
        'VariableTypes', {'double','double','double','double'}, ...
        'VariableNames', {'PCA_Dim','RMSE','Runtime','Weighted_rank'});

    fprintf('\nRunning PCA dimension sweep...\n');

    for k = 1:numel(pcaDims)
        pcaDim = pcaDims(k);

        fprintf('\n====================================\n');
        fprintf('Testing PCA dimension = %d\n', pcaDim);
        fprintf('====================================\n');

        totalTimer = tic;

        modelParameters = positionEstimatorTraining_pcaDim(trainingData, pcaDim);

        meanSqError = 0;
        n_predictions = 0;

        for tr = 1:size(testData,1)
            fprintf('Decoding block %d out of %d\n', tr, size(testData,1));

            for direc = randperm(8)
                decodedHandPos = [];
                times = 320:20:size(testData(tr,direc).spikes,2);

                for t = times
                    past_current_trial.trialId = testData(tr,direc).trialId;
                    past_current_trial.spikes = testData(tr,direc).spikes(:,1:t);
                    past_current_trial.decodedHandPos = decodedHandPos;
                    past_current_trial.startHandPos = testData(tr,direc).handPos(1:2,1);

                    [decodedPosX, decodedPosY, newParameters] = ...
                        positionEstimator_pcaDim(past_current_trial, modelParameters);

                    modelParameters = newParameters;

                    decodedPos = [decodedPosX; decodedPosY];
                    decodedHandPos = [decodedHandPos decodedPos];

                    meanSqError = meanSqError + ...
                        norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
                end

                n_predictions = n_predictions + length(times);
            end
        end

        RMSE = sqrt(meanSqError / n_predictions);
        runtime = toc(totalTimer);
        Weighted_rank = RMSE * 0.9 + 0.1 * runtime;

        results.PCA_Dim(k) = pcaDim;
        results.RMSE(k) = RMSE;
        results.Runtime(k) = runtime;
        results.Weighted_rank(k) = Weighted_rank;

        fprintf('PCA Dim = %d | RMSE = %.6f | Runtime = %.4f s | Weighted_rank = %.6f\n', ...
            pcaDim, RMSE, runtime, Weighted_rank);
    end

    disp(results);

    save('pcaDimSweep_results.mat', 'results');

    figure;
    plot(results.PCA_Dim, results.RMSE, '-o', 'LineWidth', 1.5);
    grid on;
    xlabel('PCA Dimension');
    ylabel('RMSE');
    title('RMSE vs PCA Dimension');

    figure;
    plot(results.PCA_Dim, results.Weighted_rank, '-o', 'LineWidth', 1.5);
    grid on;
    xlabel('PCA Dimension');
    ylabel('Weighted rank');
    title('Weighted rank vs PCA Dimension');
end
