function results = runPcaVarKeepSweep_withNpcRmse()
% Sweep PCA variance-retention thresholds and compare final RMSE.
%
% This version DOES NOT directly set npc/PCA dimension.
% It only changes pcaVarKeep, for example 0.90, 0.95, 0.99.
% The actual PCA dimension npc is then selected automatically inside runPCAfromCov().
%
% Extra output figures:
%   1. RMSE vs PCA variance threshold
%   2. Automatically selected NPC vs variance threshold
%   3. Weighted rank vs PCA variance threshold
%   4. RMSE vs Mean NPC
%   5. RMSE vs Max NPC
%
% Required files in the same folder:
%   1. monkeydata_training.mat
%   2. positionEstimatorTraining.m
%   3. positionEstimator.m
%   4. runPcaVarKeepSweep_withNpcRmse.m
%
% Run:
%   results = runPcaVarKeepSweep_withNpcRmse();

    load monkeydata_training.mat

    rng(2013);
    ix = randperm(length(trial));

    trainingData = trial(ix(1:50),:);
    testData = trial(ix(51:end),:);

    % ===== Change PCA variance thresholds here =====
    % Do NOT change npc directly. npc will be automatically selected
    % according to cumulative explained variance.
    varKeepList = [0.80 0.85 0.90 0.92 0.95 0.97 0.99];
    % ==============================================

    % Create a temporary training function that accepts pcaVarKeep as input.
    % This avoids permanently editing positionEstimatorTraining.m.
    makeTempTrainingFunction();

    cleanupObj = onCleanup(@() cleanupTempTrainingFunction()); %#ok<NASGU>

    results = table('Size', [numel(varKeepList), 6], ...
        'VariableTypes', {'double','double','double','double','double','double'}, ...
        'VariableNames', {'PCA_varKeep','Mean_NPC','Max_NPC','RMSE','Runtime','Weighted_rank'});

    fprintf('\nRunning PCA variance-threshold sweep...\n');

    for k = 1:numel(varKeepList)
        pcaVarKeep = varKeepList(k);

        fprintf('\n====================================\n');
        fprintf('Testing PCA varKeep = %.2f\n', pcaVarKeep);
        fprintf('====================================\n');

        totalTimer = tic;

        clear positionEstimatorTraining_varKeep_auto positionEstimator
        modelParameters = positionEstimatorTraining_varKeep_auto(trainingData, pcaVarKeep);

        % Collect the actual automatically selected npc values.
        npcValues = collectNpcValues(modelParameters);
        meanNpc = mean(npcValues, 'omitnan');
        maxNpc = max(npcValues);

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
                        positionEstimator(past_current_trial, modelParameters);

                    modelParameters = newParameters;

                    decodedPos = [decodedPosX; decodedPosY];
                    decodedHandPos = [decodedHandPos decodedPos]; %#ok<AGROW>

                    meanSqError = meanSqError + ...
                        norm(testData(tr,direc).handPos(1:2,t) - decodedPos)^2;
                end

                n_predictions = n_predictions + length(times);
            end
        end

        RMSE = sqrt(meanSqError / n_predictions);
        runtime = toc(totalTimer);
        Weighted_rank = RMSE * 0.9 + 0.1 * runtime;

        results.PCA_varKeep(k) = pcaVarKeep;
        results.Mean_NPC(k) = meanNpc;
        results.Max_NPC(k) = maxNpc;
        results.RMSE(k) = RMSE;
        results.Runtime(k) = runtime;
        results.Weighted_rank(k) = Weighted_rank;

        fprintf('varKeep = %.2f | mean npc = %.2f | max npc = %.0f | RMSE = %.6f | Runtime = %.4f s | Weighted_rank = %.6f\n', ...
            pcaVarKeep, meanNpc, maxNpc, RMSE, runtime, Weighted_rank);
    end

    disp(results);

    save('pcaVarKeepSweep_results.mat', 'results');

    % Figure 1: varKeep vs RMSE
    figure;
    plot(results.PCA_varKeep, results.RMSE, '-o', 'LineWidth', 1.5);
    grid on;
    xlabel('PCA variance retained');
    ylabel('RMSE');
    title('RMSE vs PCA variance threshold');

    % Figure 2: varKeep vs selected NPC
    figure;
    plot(results.PCA_varKeep, results.Mean_NPC, '-o', 'LineWidth', 1.5);
    hold on;
    plot(results.PCA_varKeep, results.Max_NPC, '-s', 'LineWidth', 1.5);
    grid on;
    xlabel('PCA variance retained');
    ylabel('Selected PCA dimension (NPC)');
    title('Automatically selected PCA dimension vs variance threshold');
    legend('Mean NPC', 'Max NPC', 'Location', 'best');

    % Figure 3: varKeep vs weighted rank
    figure;
    plot(results.PCA_varKeep, results.Weighted_rank, '-o', 'LineWidth', 1.5);
    grid on;
    xlabel('PCA variance retained');
    ylabel('Weighted rank');
    title('Weighted rank vs PCA variance threshold');

    % Figure 4: Mean NPC vs RMSE
    figure;
    plot(results.Mean_NPC, results.RMSE, '-o', 'LineWidth', 1.5);
    grid on;
    xlabel('Mean selected PCA dimension (Mean NPC)');
    ylabel('RMSE');
    title('RMSE vs Mean selected PCA dimension');

    % Label each point by its varKeep value.
    for i = 1:height(results)
        text(results.Mean_NPC(i), results.RMSE(i), ...
            sprintf('  %.2f', results.PCA_varKeep(i)), ...
            'VerticalAlignment', 'bottom');
    end

    % Figure 5: Max NPC vs RMSE
    figure;
    plot(results.Max_NPC, results.RMSE, '-o', 'LineWidth', 1.5);
    grid on;
    xlabel('Maximum selected PCA dimension (Max NPC)');
    ylabel('RMSE');
    title('RMSE vs Maximum selected PCA dimension');

    % Label each point by its varKeep value.
    for i = 1:height(results)
        text(results.Max_NPC(i), results.RMSE(i), ...
            sprintf('  %.2f', results.PCA_varKeep(i)), ...
            'VerticalAlignment', 'bottom');
    end

    % Print best setting by RMSE.
    [bestRMSE, bestIdx] = min(results.RMSE);
    fprintf('\nBest RMSE setting:\n');
    fprintf('varKeep = %.2f | Mean NPC = %.2f | Max NPC = %.0f | RMSE = %.6f\n', ...
        results.PCA_varKeep(bestIdx), results.Mean_NPC(bestIdx), ...
        results.Max_NPC(bestIdx), bestRMSE);
end

function makeTempTrainingFunction()
    srcFile = 'positionEstimatorTraining.m';
    tmpFile = 'positionEstimatorTraining_varKeep_auto.m';

    code = fileread(srcFile);

    % Rename function and add pcaVarKeep as an input.
    code = regexprep(code, ...
        'function\s+modelParameters\s*=\s*positionEstimatorTraining\s*\(\s*training_data\s*\)', ...
        'function modelParameters = positionEstimatorTraining_varKeep_auto(training_data, pcaVarKeep)');

    % Remove the hard-coded pcaVarKeep assignment, because it is now an input.
    code = regexprep(code, ...
        '\n\s*pcaVarKeep\s*=\s*[0-9.]+\s*;[^\n]*', ...
        sprintf('\n    %% pcaVarKeep is supplied by runPcaVarKeepSweep_withNpcRmse.m'));

    % Store actual npc values inside the model structs.
    code = strrep(code, ...
        'classModel(tIdx).labels = ycls;', ...
        sprintf('classModel(tIdx).labels = ycls;\n        classModel(tIdx).npc = npc;'));

    code = strrep(code, ...
        'regModel(d, tIdx).y0 = mean(ty);', ...
        sprintf('regModel(d, tIdx).y0 = mean(ty);\n            regModel(d, tIdx).npc = npcReg;'));

    fid = fopen(tmpFile, 'w');
    if fid == -1
        error('Could not create temporary training file: %s', tmpFile);
    end
    fwrite(fid, code);
    fclose(fid);
end

function cleanupTempTrainingFunction()
    tmpFile = 'positionEstimatorTraining_varKeep_auto.m';
    if exist(tmpFile, 'file')
        delete(tmpFile);
    end
end

function npcValues = collectNpcValues(modelParameters)
    npcValues = [];

    if isfield(modelParameters, 'classModel')
        for i = 1:numel(modelParameters.classModel)
            if isfield(modelParameters.classModel(i), 'npc')
                npcValues(end+1) = modelParameters.classModel(i).npc; %#ok<AGROW>
            end
        end
    end

    if isfield(modelParameters, 'regModel')
        for i = 1:numel(modelParameters.regModel)
            if isfield(modelParameters.regModel(i), 'npc')
                npcValues(end+1) = modelParameters.regModel(i).npc; %#ok<AGROW>
            end
        end
    end

    if isempty(npcValues)
        warning('No npc values were found in modelParameters. Mean_NPC and Max_NPC will be NaN.');
        npcValues = NaN;
    end
end
