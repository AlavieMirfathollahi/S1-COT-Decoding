function [y_predict_test_PLSr, y_predict_test_second, response_test, fea_maxRs_secondr,...
    fea_maxRs_PLSr, Rsq_Feas_secondr, Rsq_Feas_PLSr, Rsqtest] = PLS_MLR_ContDecoder...
    (K_fold, Train_responses, predictors, predictors_test, TEST_responses, PLS_mode, second_model)
% INPUTs      ************************************************************************
%   K_fold:                           folds # (test / train)
%   Train_responses:              Actual response of Train data
%   predictors:                     Neural data of Train
%   predictors_test:               Neural data of Test
%   TEST_responses:              Actual response of Test data
%   PLS_mode:                     Mode of PLS decoder, use 'oneResp' if you want to build model for each Response and use 'AllResp' if you want to build one model for all responses
%   second_model:                 second secoding Model, 'MLR' or 'SVM'
% OUTPUTs   ************************************************************************
%   y_predict_test_PLSr:         Predicted response of Test data (PLS Model)
%   y_predict_test_second:       Predicted response of Test data (Second Model)
%   response_test:                  Actual response of Test data
%   fea_maxRs_secondr:          Opt feature #  (Second Model)
%   fea_maxRs_PLSr:              Opt feature #  (PLS Model)
%   Rsq_Feas_secondr:             R^2 in Train Data  (Second Model)
%   Rsq_Feas_PLSr:                 R^2 in Train Data   (PLS Model)
%   Rsqtest:                           R^2 in Test Data  (Struct, both models)
%*************************************************************************
% Alavie Mirfathollahi - Dec 2022 <alaviemir@gmail.com>
%*************************************************************************
% IMPORTANT NOTICE: If you use this code in your work, please cite [1].
%*************************************************************************
%   References
%   [1] A. Mirfathollahi, M.T. Ghodrati, V. Shalchyan, M. R. Zarrindast, M. R. Daliri,
%       "Decoding hand kinetics and kinematics using somatosensory cortex activity
%       in active and passive movement", iScience, Aug. 2023
%*************************************************************************
options = statset();
options.UseParallel = 'true';
Response_size = size(Train_responses, 2);

%% Cross validation in Train data to fit Selection
part_size = round(   size(predictors, 1)  / K_fold  );
%  Validation (to find best feature #)
for i_Kfold2 = 1:K_fold
    % Test
    sample_start = ((i_Kfold2 - 1) * part_size) + 1;
    sample_end = min(  (i_Kfold2 * part_size)   ,   size(predictors, 1)  );
    Test_indx = [sample_start : sample_end];
    data_test2 = predictors(Test_indx, :);
    response_test2 = Train_responses(Test_indx,:);
    % Train
    data_train2 = predictors;
    data_train2(Test_indx, :)= [];
    response_train2 = Train_responses;
    response_train2(Test_indx, :)= [];
    clear sample_start sample_end Test_indx
    
    if PLS_mode == 'oneResp'    % ----------------method 1 by 1 Resp ----------------
        for i_responses = 1:Response_size
            
            % PLS
            ncomp = size(data_train2,2);
            [~,YL,~,~,~,~, ~, stats] =...
                plsregress(data_train2, response_train2(:, i_responses), ncomp, 'cv', K_fold,'options',options);
            
            for i_Features = 1:5:size(data_train2, 2)
                
                % MLR
                regressionsecond=regress(response_train2(:, i_responses), data_train2(:, 1:i_Features));
                y_predict_test2_secondr = data_test2(:, 1:i_Features) * regressionsecond;
                
                Rsq_Feas_secondr{i_responses, i_Kfold2}(i_Features) =...
                    1 - sum(((response_test2(:, i_responses)) - (y_predict_test2_secondr)).^2)/...
                    sum(((response_test2(:, i_responses)) - mean((response_test2(:, i_responses)))).^2);
                
                % PLS
                beta = stats.W(:,1:i_Features)*YL(:, 1:i_Features)';
                b = mean(response_train2(:, i_responses), 1)-mean(data_train2, 1)*beta;
                beta_PLS = [b;beta];
                y_predict_test2_PLSr = [ones(size(data_test2,1),1), data_test2] * beta_PLS;
                
                Rsq_Feas_PLSr{i_responses, i_Kfold2}(i_Features) =...
                    1 - sum(((response_test2(:, i_responses)) - (y_predict_test2_PLSr)).^2)/...
                    sum(((response_test2(:, i_responses)) - mean((response_test2(:, i_responses)))).^2);
                
                clear regressionTree regressionLinM regressionsecond beta_PLS beta b y_predict_test2_tree y_predict_test2_LinM y_predict_test2_secondr y_predict_test2_PLSr predictors_train2
            end
        end
        
    elseif PLS_mode == 'AllResp' % ----------------method AllResp----------------
        
        % PLS
        ncomp = size(data_train2,2);
        [~,YL,~,~,~,~, ~, stats] =...
            plsregress(data_train2, response_train2, ncomp, 'cv', K_fold,'options',options);
        
        for i_Features = 1:5:size(data_train2, 2)
            for i_responses = 1:Response_size
                
                if second_model == 'MLR'
                    % MLR
                    regressionsecond=regress(response_train2(:, i_responses), data_train2(:, 1:i_Features));
                    y_predict_test2_secondr = data_test2(:, 1:i_Features) * regressionsecond;
                    
                    Rsq_Feas_secondr{i_responses, i_Kfold2}(i_Features) =...
                        1 - sum(((response_test2(:, i_responses)) - (y_predict_test2_secondr)).^2)/...
                        sum(((response_test2(:, i_responses)) - mean((response_test2(:, i_responses)))).^2);
                    clear regressionsecond y_predict_test2_secondr
                    
                elseif second_model == 'SVM'
                    % SVM
                    responseScale = iqr(response_train2(:, i_responses));
                    if ~isfinite(responseScale) || responseScale == 0.0
                        responseScale = 1.0;
                    end
                    boxConstraint = responseScale/1.349;   epsilon = responseScale/13.49;
                    regressionsecond = fitrsvm(data_train2(:, 1:i_Features), response_train2(:, i_responses), ...
                        'KernelFunction', 'polynomial', 'PolynomialOrder', 3,'KernelScale', 'auto', ...
                        'BoxConstraint', boxConstraint, 'Epsilon', epsilon, 'Standardize', true);
                    
                    y_predict_test2_secondr = predict(regressionsecond,data_test2(:, 1:i_Features));
                    
                    Rsq_Feas_secondr{i_responses, i_Kfold2}(i_Features)  =...
                        1 - sum(((response_test2(:, i_responses)) - (y_predict_test2_secondr)).^2)/...
                        sum(((response_test2(:, i_responses)) - mean((response_test2(:, i_responses)))).^2);
                    
                    clear regressionsecond y_predict_test2_secondr
                end
            end
            
            % PLS
            beta = stats.W(:, 1:i_Features)*YL(:, 1:i_Features)';
            b = mean(response_train2, 1) - mean(data_train2, 1) * beta;
            beta_PLS = [b;beta];
            y_predict_test2_PLSr = [ones(size(data_test2,1),1), data_test2] * beta_PLS;
            
            for i_responses = 1:Response_size
                Rsq_Feas_PLSr{i_responses, i_Kfold2}(i_Features)  =...
                    1 - sum(((response_test2(:,i_responses)) - (y_predict_test2_PLSr(:,i_responses))).^2)/...
                    sum(((response_test2(:,i_responses)) - mean((response_test2(:,i_responses)))).^2);
            end
            
            clear beta_PLS beta b y_predict_test2_PLSr
        end
    end
end

% Opt Feature #
for i_responses = 1:size(Rsq_Feas_PLSr ,1)
    A = []; B = [];
    for i_Kfold2 = 1:K_fold
        A = vertcat(Rsq_Feas_PLSr{i_responses, i_Kfold2}, A);
        B = vertcat(Rsq_Feas_secondr{i_responses, i_Kfold2}, B);
    end
    Rsq_Feas_PLSrall{i_responses} = A;
    Rsq_Feas_secondall{i_responses} = B;
    % Feature Number
    [~, fea_maxRs_secondr(i_responses)] = max(mean(Rsq_Feas_secondall{i_responses}, 1));
    [~, fea_maxRs_PLSr(i_responses)] = max(mean(Rsq_Feas_PLSrall{i_responses}, 1));
end

%% Final Model in test data -----------------------------------------------------
if PLS_mode == 'oneResp'
    for i_responses = 1:Response_size
        % PLS
        ncomp = size(predictors,2);
        [~,YL,~,~,~,~, ~, stats] = plsregress(predictors, Train_responses(:, i_responses), ncomp, 'cv', K_fold,'options',options);
        beta = stats.W(:,1:fea_maxRs_PLSr)*YL(1:fea_maxRs_PLSr)';
        b = mean(Train_responses(:, i_responses))-mean(predictors)*beta;
        beta_PLS = [b;beta];
        y_predict_test_PLSr{i_responses} = [ones(size(predictors_test,1),1),...
            predictors_test] * beta_PLS;
        
        %Second Model
        if second_model == 'MLR'
            % MLR
            regressionsecond=regress(predictors(:, 1:fea_maxRs_secondr(i_responses)), Train_responses(:, i_responses));
            y_predict_test_second{i_responses} = predictors_test(:, 1:fea_maxRs_secondr(i_responses)) * regressionsecond;
        elseif second_model == 'SVM'
            % SVM
            responseScale = iqr(Train_responses(:, i_responses));
            if ~isfinite(responseScale) || responseScale == 0.0
                responseScale = 1.0;
            end
            boxConstraint = responseScale/1.349;                epsilon = responseScale/13.49;
            regressionsecond = fitrsvm(predictors(:, 1:fea_maxRs_secondr(i_responses)), Train_responses(:, i_responses), ...
                'KernelFunction', 'polynomial', 'PolynomialOrder', 3, 'KernelScale', 'auto', ...
                'BoxConstraint', boxConstraint, 'Epsilon', epsilon, 'Standardize', true);
            y_predict_test_second{i_responses} = predict(regressionsecond, predictors_test(:, 1:fea_maxRs_secondr(i_responses)));
        end
        
        % True Responses
        response_test{i_responses} = TEST_responses(:, i_responses);
        clear regressionsecond beta_PLS beta b
        
        % R2
        Rsqtest.PLS(i_responses) =...
            1 - sum(((TEST_responses(:, i_responses)) - (y_predict_test_PLSr{i_responses})).^2)/...
            sum(((TEST_responses(:,i_responses)) - mean((TEST_responses(:,i_responses)))).^2);
        Rsqtest.second(i_responses) =...
            1 - sum(((TEST_responses(:, i_responses)) - (y_predict_test_second{i_responses})).^2)/...
            sum(((TEST_responses(:, i_responses)) - mean((TEST_responses(:, i_responses)))).^2);
    end
    
    %% AllResp
elseif PLS_mode == 'AllResp'
    % PLS
    clear beta_PLS beta b YL stats ncomp
    ncomp = size(predictors,2);
    [~,YL,~,~,~,~, ~, stats] = plsregress(predictors, Train_responses, ncomp, 'cv', K_fold,'options',options);
    
    for i_responses = 1:Response_size
        beta = stats.W(:,1:fea_maxRs_PLSr(i_responses))*YL(:, 1:fea_maxRs_PLSr(i_responses))';
        b = mean(Train_responses(:, i_responses), 1)-mean(predictors, 1)*beta;
        beta_PLS = [b;beta];
        Ped_allrep = [ones(size(predictors_test,1),1),...
            predictors_test] * beta_PLS;
        
        y_predict_test_PLSr{i_responses} = Ped_allrep(:, i_responses);
        clear beta_PLS beta b
        
        %Second Model
        if second_model == 'MLR'
            % MLR
            regressionsecond = regress(Train_responses(:, i_responses), predictors(:, 1:fea_maxRs_secondr(i_responses)));
            y_predict_test_second{i_responses} = predictors_test(:, 1:fea_maxRs_secondr(i_responses)) * regressionsecond;
            
        elseif second_model == 'SVM'
            % SVM
            responseScale = iqr(Train_responses(:, i_responses));
            if ~isfinite(responseScale) || responseScale == 0.0
                responseScale = 1.0;
            end
            boxConstraint = responseScale/1.349;                epsilon = responseScale/13.49;
            regressionsecond = fitrsvm(predictors(:, 1:fea_maxRs_secondr(i_responses)), Train_responses(:, i_responses), ...
                'KernelFunction', 'polynomial', 'PolynomialOrder', 3, 'KernelScale', 'auto', ...
                'BoxConstraint', boxConstraint, 'Epsilon', epsilon, 'Standardize', true);
            y_predict_test_second{i_responses} = predict(regressionsecond, predictors_test(:, 1:fea_maxRs_secondr(i_responses)));
        end
        clear regressionsecond
        
        % True Responses
        response_test{i_responses} = TEST_responses(:, i_responses);
        
        % R2
        Rsqtest.PLS(:, i_responses) =...
            1 - sum(((TEST_responses(:, i_responses)) - (y_predict_test_PLSr{i_responses})).^2)/...
            sum(((TEST_responses(:,i_responses)) - mean((TEST_responses(:,i_responses)))).^2);
        Rsqtest.second(:, i_responses) =...
            1 - sum(((TEST_responses(:, i_responses)) - (y_predict_test_second{i_responses})).^2)/...
            sum(((TEST_responses(:, i_responses)) - mean((TEST_responses(:, i_responses)))).^2);
    end
end

VariableNames = {'X','Y', 'Vx', 'Vy', 'F1','F2','F3','M1','M2','M3' };
DataR = [Rsqtest.PLS(1:length(VariableNames)); Rsqtest.second(1:length(VariableNames))];
T = array2table(round(DataR.*100)./100,...
    'VariableNames', VariableNames, 'RowName',{'PLS','second'});
disp(T)

end
