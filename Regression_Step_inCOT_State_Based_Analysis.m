function  [perf_classification_TEST_COT, perf_classification_COT, test_sample_length,...
    movement_on_test, Class_test, Labels_Result_TEST_reg, y_predict_test_PLSr,...
    y_predict_test_MLRr, response_test, fea_maxRs_MLRr, fea_maxRs_PLSr,...
    Cofu_mat_TEST, Rsq_Feas_MLRr, Rsq_Feas_PLSr, Rsqtest] =...
    Regression_Step_inCOT_State_Based_Analysis(ind_train, ind_test, Classifier, K_fold_classification, selectionF,...
    Feature_classify, Labels, Prep_data, Delay_sample, num_Lag, remove_very_corr_Feature,...
    feature_regression_selection, K_fold_regression, PLS_mode, need_kernel_Regression, kernel_ord_Regression, second_model_names)
% INPUTs      ************************************************************************
%   ind_train:                               Index of Train Data
%   ind_test:                                 Index of Test Data
%   Classifier:                                Classifier Name
%   K_fold_classification                   folds of classification model training
%   selectionF                                feature selection method
%   Feature_classify                        Extracted feature for classification
%   Labels                                     Labels
%   Prep_data:                              Data
%   num_Lag:                                lag
%   Delay_sample:                          Delay in time, between data and Responses
%   remove_very_corr_Feature:          Removing Corrolated Features (1)
%   feature_regression_selection:         Feature Selection (1)
%   need_kernel_Regression:               Kernel for smoothing data,  'gausswin' / 'tukeywin' / 'chebwinn' / 'gammakrl'
%   kernel_ord_Regression:                order of kernel
%   K_fold_regression:                      folds # (test / train)
%   PLS_mode:                              Mode of PLS decoder, use 'oneResp' if you want to build model for each Response and use 'AllResp' if you want to build one model for all responses
%   second_model:                          second secoding Model, 'MLR' or 'SVM'
% OUTPUTs   ************************************************************************
%   y_predict_test_PLSr:         Predicted response of Test data (PLS Model)
%   y_predict_test_second:       Predicted response of Test data (Second Model)
%   response_test:                  Actual response of Test data
%   fea_maxRs_secondr:          Opt feature #  (Second Model)
%   fea_maxRs_PLSr:              Opt feature #  (PLS Model)
%   movement_on_test:           Movement onset
%   Class_test:                      Labels of test Data
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
%% Classification model     ************************************************************************
[Mdl_classification, num_best_Fea, features, perf_classification_COT] =...
    classification_model_generator_optional(Classifier, selectionF,...
    Feature_classify, ind_train, K_fold_classification, Labels);

%% Classify Test data     ************************************************************************
indx_f = features(1:num_best_Fea);
Y = Feature_classify(:, indx_f);
class = predict(Mdl_classification, Y(ind_test, :));
cp = classperf(Labels);
classperf(cp,class,ind_test);
perf_classification_TEST_COT = (1-cp.ErrorRate)*100;
Cofu_mat_TEST = cp.CountingMatrix;

%% Kernel for smoothing FR     ******************************************************************
if need_kernel_Regression == 'gausswin'
    for i_trial = 1:length(Prep_data)
        Prep_data(i_trial).S1_FR = filtfilt(gausswin(kernel_ord_Regression) , 1 , Prep_data(i_trial).S1_FR);
    end
elseif need_kernel_Regression == 'tukeywin'
    for i_trial = 1:length(Prep_data)
        Prep_data(i_trial).S1_FR = filtfilt(tukeywin(kernel_ord_Regression) , 1 , Prep_data(i_trial).S1_FR);
    end
elseif need_kernel_Regression == 'chebwinn'
    for i_trial = 1:length(Prep_data)
        Prep_data(i_trial).S1_FR = filtfilt(chebwin(kernel_ord_Regression) , 1 , Prep_data(i_trial).S1_FR);
    end
elseif need_kernel_Regression == 'gammakrl'
    t=0:.01:1;     ts = 0;      alpha = 1.5;     beta = kernel_ord_Regression;
    Kernel_R = (beta^alpha)*((t-ts).^(alpha - 1)).*exp(-beta*(t-ts))/gamma(alpha);
    for i_trial = 1:length(Prep_data)
        coet_1 = sum(impz(Kernel_R , 1 ));
        Prep_data(i_trial).S1_FR = filter(Kernel_R./coet_1 , 1 , Prep_data(i_trial).S1_FR);
    end
end

%% sepration of data of each classes - Train Data      ********************************
data_train_reg = Prep_data(ind_train);                         Labels_train_reg = Labels(ind_train);
i_1 = 0; i_2=0; i_3=0; i_4=0;                                      a1 = []; a2 = []; a3 = []; a4 = [];
for i_trials = 1:length(Labels_train_reg)
    if Labels_train_reg(i_trials) == 1
        i_1 = i_1 + 1;
        Fea_classes{1, i_1} = data_train_reg(i_trials).S1_FR;
        a1 = [a1, i_trials];
    elseif Labels_train_reg(i_trials) == 3
        i_2 = i_2 + 1;
        Fea_classes{2, i_2} = data_train_reg(i_trials).S1_FR;
        a2 = [a2, i_trials];
    elseif Labels_train_reg(i_trials) == 5
        i_3 = i_3 + 1;
        Fea_classes{3, i_3} = data_train_reg(i_trials).S1_FR;
        a3 = [a3, i_trials];
    elseif Labels_train_reg(i_trials) == 7
        i_4 = i_4 + 1;
        Fea_classes{4, i_4} = data_train_reg(i_trials).S1_FR;
        a4 = [a4, i_trials];
    end
end

% responses
indx{1} = a1;  indx{2} = a2;  indx{3} = a3;  indx{4} = a4;
resp_classes{1} = data_train_reg(indx{1});                      resp_classes{2} = data_train_reg(indx{2});
resp_classes{3} = data_train_reg(indx{3});                      resp_classes{4} = data_train_reg(indx{4});

%% sepration of data of each classes - Test Data      ************************************
Labels_Result_TEST_reg = class;                                   data_TEST_reg = Prep_data(ind_test);
i_1 = 0; i_2=0; i_3=0; i_4=0;                                      a1 = []; a2 = []; a3 = []; a4 = [];
for i_trials = 1:length(Labels_Result_TEST_reg)
    if Labels_Result_TEST_reg(i_trials) == 1
        i_1 = i_1 + 1;
        Fea_classes_TEST{1, i_1} = data_TEST_reg(i_trials).S1_FR;
        a1 = [a1, i_trials];
    elseif Labels_Result_TEST_reg(i_trials) == 3
        i_2 = i_2 + 1;
        Fea_classes_TEST{2, i_2} = data_TEST_reg(i_trials).S1_FR;
        a2 = [a2, i_trials];
    elseif Labels_Result_TEST_reg(i_trials) == 5
        i_3 = i_3 + 1;
        Fea_classes_TEST{3, i_3} = data_TEST_reg(i_trials).S1_FR;
        a3 = [a3, i_trials];
    elseif Labels_Result_TEST_reg(i_trials) == 7
        i_4 = i_4 + 1;
        Fea_classes_TEST{4, i_4} = data_TEST_reg(i_trials).S1_FR;
        a4 = [a4, i_trials];
    end
end

% response test reg
indx = {}; resp_classes_TEST = {}; indx{1} = a1;  indx{2} = a2;  indx{3} = a3;  indx{4} = a4;
resp_classes_TEST{1} = data_TEST_reg(indx{1});             resp_classes_TEST{2} = data_TEST_reg(indx{2});
resp_classes_TEST{3} = data_TEST_reg(indx{3});             resp_classes_TEST{4} = data_TEST_reg(indx{4});

%% regression for each classes  ************************************************************************
for i_classes = 1:size(Fea_classes,1)
    %% TRAIN  ************************************************************************
    Data_FR = []; positions = []; Velocities = []; Forces = []; joint_angles = [];
    Prep_Resp_data_Train = resp_classes{i_classes};           Num_neurons = length(Prep_Resp_data_Train(1).S1_FR(1, :));
    for i_trial_all = 1:size(Fea_classes, 2)
        if isempty(Fea_classes{i_classes,i_trial_all})~=1
            Data_FR_lag = [];
            move_on_period =  Prep_Resp_data_Train(i_trial_all).idx_movement_on;
            for i_Lag = 1:num_Lag
                Data_lag = [];
                if i_Lag == 1 % No Lag
                    Data_lag = [Fea_classes{i_classes,i_trial_all}(move_on_period + 1 - i_Lag:end  , :)];
                else
                    Data_lag = [zeros(i_Lag - 1 , Num_neurons);...
                        Fea_classes{i_classes,i_trial_all}(move_on_period : end - i_Lag + 1  , :)];
                end
                Data_FR_lag = [Data_FR_lag, Data_lag];
            end
            
            % remove lag related row
            if num_Lag >1
                Data_FR_lag(1: num_Lag, :) = [];
            end
            
            %  all trials
            Data_FR = [Data_FR; Data_FR_lag];
            
            % Responses
            if num_Lag >1
                start_index = move_on_period + num_Lag; % because 1 is no Lag!
            else
                start_index = move_on_period + num_Lag - 1; % because 1 is no Lag!
            end
            
            positions = [positions; Prep_Resp_data_Train(i_trial_all).pos(...
                start_index : end, :)];
            Velocities = [Velocities; Prep_Resp_data_Train(i_trial_all).vel(...
                start_index : end, :)];
            Forces = [Forces; Prep_Resp_data_Train(i_trial_all).force(...
                start_index : end, :)];
            joint_angles = [joint_angles;  Prep_Resp_data_Train(i_trial_all).joint_ang(...
                start_index : end, :)];
        end
    end
    Train_responses = [positions, Velocities, Forces, joint_angles];
    clear positions Velocities Forces accelerations joint_angles joint_Velocities muscle_lengths muscle_Velocities markers
    predictors = Data_FR;
    
    %% TEST  ************************************************************************
    Data_FR = []; positions = []; Velocities = []; Forces = [];  joint_angles = []; Prep_data_TEST = [];
    Prep_data_TEST = resp_classes_TEST{i_classes};
    for i_trial_all = 1:size(Fea_classes_TEST, 2)
        if isempty(Fea_classes_TEST{i_classes,i_trial_all})~=1
            % Lag calcalation
            Data_FR_lag = [];
            move_on_period_test =  Prep_data_TEST(i_trial_all).idx_movement_on;
            for i_Lag = 1:num_Lag
                Data_lag = [];
                if i_Lag == 1 % No Lag
                    Data_lag = [Fea_classes_TEST{i_classes,i_trial_all}(move_on_period_test + 1 - i_Lag:end, :)];
                else
                    Data_lag = [zeros(i_Lag - 1 , Num_neurons);...
                        Fea_classes_TEST{i_classes,i_trial_all}(move_on_period_test : end - i_Lag + 1, :)];
                end
                Data_FR_lag = [Data_FR_lag, Data_lag];
            end
            
            % remove lag related row
            if num_Lag >1
                Data_FR_lag(1: num_Lag, :) = [];
            end
            
            test_sample_length(i_trial_all) = size(Data_FR_lag, 1);
            Data_FR = [Data_FR; Data_FR_lag];
            
            % Responses
            if num_Lag >1
                start_index = move_on_period_test + num_Lag; % because 1 is no Lag!
            else
                start_index = move_on_period_test + num_Lag - 1; % because 1 is no Lag!
            end
            
            positions = [positions; Prep_data_TEST(i_trial_all).pos(start_index : end , :)];
            Velocities = [Velocities; Prep_data_TEST(i_trial_all).vel(start_index : end , :)];
            Forces = [Forces; Prep_data_TEST(i_trial_all).force(start_index : end , :)];
            joint_angles = [joint_angles;  Prep_data_TEST(i_trial_all).joint_ang(start_index : end , :)];
        end
    end
    
    TEST_responses = [];
    TEST_responses = [positions, Velocities, Forces, joint_angles];
    clear positions Velocities Forces accelerations joint_angles joint_Velocities muscle_lengths muscle_Velocities markers
    
    predictors_test = Data_FR;
    
    %% Delay  ************************************************************************
    if Delay_sample > 0
        Train_responses = Train_responses(1 : end - Delay_sample, :);
        TEST_responses = TEST_responses(1:end - Delay_sample, :);
        predictors = predictors(Delay_sample + 1 : end, :);
        predictors_test = predictors_test(Delay_sample + 1 : end, :);
    elseif Delay_sample < 0
        Delay_sampleN = abs(Delay_sample);
        Train_responses = Train_responses(Delay_sampleN + 1 : end, :);
        TEST_responses = TEST_responses(Delay_sampleN + 1 : end, :);
        predictors = predictors(1 : end - Delay_sampleN, :);
        predictors_test = predictors_test(1 : end - Delay_sampleN, :);
    end
    
    %% remove bad features  *****************************************************
    if remove_very_corr_Feature == 1
        [predictors, predictors_test] = remove_bad_features(predictors, predictors_test);
    end
    
    %% fearure Selection  *********************************************************
    if feature_regression_selection == 1
        {add your feature selection (for regression) here!}
    end
    
    %% Model TRAIN  ************************************************************
    for i_second_model = 1:length(second_model_names)
        second_model =  second_model_names{i_second_model};
        [y_predict_test_PLSr{i_classes, i_second_model}, y_predict_test_MLRr{i_classes, i_second_model},...
            response_test{i_classes, i_second_model}, fea_maxRs_MLRr{i_classes, i_second_model}, fea_maxRs_PLSr{i_classes, i_second_model},...
            Rsq_Feas_MLRr{i_classes, i_second_model}, Rsq_Feas_PLSr{i_classes, i_second_model}, Rsqtest{i_classes, i_second_model}] =...
            PLS_MLR_ContDecoder...
            (K_fold_regression, Train_responses, predictors, predictors_test, TEST_responses, PLS_mode, second_model);
    end
    fprintf(['Processing of ', num2str(i_classes), ' Done!\n'])
    
end

%% for next analysis
for i_test = 1:length(find(ind_test==1))
    movement_on_test(i_test) = Prep_data(ind_test).idx_movement_on;
    Class_test(i_test, 1) =  Prep_data(ind_test).tgtDir;
    Class_test(i_test, 2) =  Prep_data(ind_test).bumpDir;
end

end

