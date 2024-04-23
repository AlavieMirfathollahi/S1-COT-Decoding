function [y_predict_test_PLSr, y_predict_test_MLRr, response_test, fea_maxRs_MLRr, fea_maxRs_PLSr, movement_on_test,...
    Class_test, Rsq_Feas_MLRr, Rsq_Feas_PLSr, Rsqtest] = Regression_Step_Conventional(...
    ind_train, ind_test, Prep_data, num_Lag, Delay_sample, remove_very_corr_Feature, feature_regression_selection,...
    K_fold_regression, PLS_mode, need_kernel_Regression, kernel_ord_Regression, second_model)
% INPUTs      ************************************************************************
%   ind_train:                               Index of Train Data
%   ind_test:                                 Index of Test Data
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
%% Kernel for smoothing FR
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

%% regression for each classes *************************************************************************
Data_FR = []; positions = []; Velocities = []; Forces = []; joint_angles = [];
Prep_Resp_data_Train = Prep_data(ind_train);
Num_neurons = length(Prep_Resp_data_Train(1).S1_FR(1, :));

for i_trial_all = 1:length(Prep_Resp_data_Train)
    Data_FR_lag = [];
    move_on_period =  Prep_Resp_data_Train(i_trial_all).idx_movement_on;
    
    for i_Lag = 1:num_Lag
        Data_lag = [];
        if i_Lag == 1 % No Lag
            Data_lag = [Prep_Resp_data_Train(i_trial_all).S1_FR(move_on_period - 1 + i_Lag:end , :)];
        else
            Data_lag = [zeros(i_Lag - 1 , Num_neurons);...
                Prep_Resp_data_Train(i_trial_all).S1_FR(move_on_period  : end - i_Lag + 1 , :)];
        end
        Data_FR_lag = [Data_FR_lag, Data_lag];
    end
    
    % remove lag related row
    if num_Lag > 1
        Data_FR_lag(1: num_Lag, :) = [];
    end
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
Train_responses = [positions, Velocities, Forces, joint_angles];
clear positions Velocities Forces joint_angles
predictors = Data_FR;

%% TEST *************************************************************************
Data_FR = []; positions = []; Velocities = []; Forces = []; joint_angles = [];
Prep_data_TEST = Prep_data(ind_test);

for i_trial_all = 1:length(Prep_data_TEST)
    
    % Lag calcalation
    Data_FR_lag = [];
    move_on_period_test =  Prep_data_TEST(i_trial_all).idx_movement_on;
    for i_Lag = 1:num_Lag
        Data_lag = [];
        if i_Lag == 1 % No Lag
            Data_lag = [Prep_data_TEST(i_trial_all).S1_FR(move_on_period_test - 1 + i_Lag:end, :)];
        else
            Data_lag = [zeros(i_Lag - 1 , Num_neurons);...
                Prep_data_TEST(i_trial_all).S1_FR(move_on_period_test  : end - i_Lag + 1 , :)];
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
    
    positions = [positions; Prep_data_TEST(i_trial_all).pos(...
        start_index : end , :)];
    
    Velocities = [Velocities; Prep_data_TEST(i_trial_all).vel(...
        start_index : end , :)];
    
    Forces = [Forces; Prep_data_TEST(i_trial_all).force(...
        start_index : end , :)];
    
    joint_angles = [joint_angles;  Prep_data_TEST(i_trial_all).joint_ang(...
        start_index : end , :)];
end
TEST_responses = [positions, Velocities, Forces, joint_angles];
clear positions Velocities Forces joint_angles
predictors_test = Data_FR;

%% Delay *************************************************************************
if Delay_sample > 1
    Train_responses = Train_responses(1 : end - Delay_sample, :);
    TEST_responses = TEST_responses(1:end - Delay_sample, :);
    
    predictors = predictors(Delay_sample + 1 : end, :);
    predictors_test = predictors_test(Delay_sample + 1 : end, :);
end

%% remove bad features ***********************************************************
if remove_very_corr_Feature == 1
    [predictors, predictors_test] = remove_bad_features(predictors, predictors_test);
end

%% fearure Selection *************************************************************
if feature_regression_selection == 1
   {add your feature selection (for regression) here!}
end

%% Model TRAINing ****************************************************************
[y_predict_test_PLSr, y_predict_test_MLRr, response_test,...
    fea_maxRs_MLRr, fea_maxRs_PLSr, Rsq_Feas_MLRr,...
    Rsq_Feas_PLSr ,Rsqtest ] = PLS_MLR_ContDecoder...
    (K_fold_regression, Train_responses, predictors, predictors_test, ...
    TEST_responses, PLS_mode, second_model);

%% for next analysis :))
for i_test = 1:length(find(ind_test==1))
    movement_on_test(i_test) = Prep_data(ind_test).idx_movement_on;
    Class_test(i_test, 1) =  Prep_data(ind_test).tgtDir;
    Class_test(i_test, 2) =  Prep_data(ind_test).bumpDir;
end

end

%*************************************************************************
%   References
%   [1] A. Mirfathollahi, M.T. Ghodrati, V. Shalchyan, M. R. Zarrindast, M. R. Daliri,
%       "Decoding hand kinetics and kinematics using somatosensory cortex activity
%       in active and passive movement", IScience, Aug. 2023
%*************************************************************************
