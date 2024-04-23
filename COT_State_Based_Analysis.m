%% COT   -   State-Based Analysis
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
clc
clear all
clearvars
warning off

%% Inputs *************************************************************************
dataroot = {path to Dataset folder};
load({path to index of train and test data, 5 folds 10 runs});      % ind train test

% Classification Inputs
index_before = 0;                                                   index_after = 20;
Fea_name = 'Statf';                                                Classifier = 'LDA';
K_fold_classification = 5;                                         selectionF = 'MI';
need_kernel = 'gausswin';                                         kernel_ord_classification = 11;
window_MTfeaClssification = index_after;                   overlapp_MTfeaClssification = 0;
window_statFeaClssification = index_after/2;              overlapp_statFeaClssification = 0;

% Regression Inputs
num_Lag = 1;                                                       PLS_mode = 'AllResp';
second_model = 'MLR';                                            kernel_ord_Regression = 11;
K_fold_regression = 3;                                             feature_regression_selection = 0;
remove_very_corr_Feature = 1;                                  Delay_sample = 0;
need_kernel_Regression = 'gammakrl';

%% Paths *************************************************************************
path_lib_data = what;
path_lib_data = path_lib_data.path;

% Path lib
ind=find(path_lib_data=='\');
path_lib = path_lib_data(1:ind(end));
addpath(genpath([path_lib, '\lib'])) % add path library
file_info = dir(fullfile(dataroot,'reaching_experiments','*COactpas*.mat')); % COT Task
filenames = horzcat({file_info.name})'; % find filenames (TRT)

% Path Save
ind2=find(dataroot=='\');
path_Save = dataroot(1:ind2(end));

%% define *************************************************************************
K_fold = size(Inds.ind_train_save, 3);                            Runs_time = size(Inds.ind_train_save, 4);
Class_test_Swich = cell(length(filenames), 2, K_fold);        Run_threshold = 0;
response_test_Swich = cell(length(filenames), 2, K_fold);
fea_maxRs_PLSr_Swich = cell(length(filenames), 2, K_fold);
Cofu_mat_TEST_Swich = cell(length(filenames), 2, K_fold);
fea_maxRs_MLRr_Swich = cell(length(filenames), 2, K_fold);
test_sample_length_Swich = cell(length(filenames), 2, K_fold);
movement_on_test_Swich = cell(length(filenames), 2, K_fold);
y_predict_test_PLSr_Swich = cell(length(filenames), 2, K_fold);
y_predict_test_MLRr_Swich = cell(length(filenames), 2, K_fold);
Labels_Result_TEST_reg_Swich = cell(length(filenames), 2, K_fold);
perf_classification_TEST_COT_Swich = cell(length(filenames), 2, K_fold);
perf_classification_COT_train_Swich = cell(length(filenames), 2, K_fold);

%% Run in each datafile *************************************************************************
for filenum = 1:length(filenames)
    td = load(fullfile(dataroot,'reaching_experiments', [filenames{filenum}]));
    [td_act, td_pas] = Preprocessing_COT(td);
    clear td
    
    %% Active vs passive *************************************************************************
    for i_Act_Pas = 1:2
        if (i_Act_Pas == 1)
            Prep_data = td_act;
        elseif (i_Act_Pas == 2)
            Prep_data = td_pas;
            clear td_act
        end
        
        %% Feature Extraction for Classification *****************************************************
        [Prep_data, Feature_classify, Labels] =...
            FeatureExtraction_for_Classification(Prep_data, need_kernel,...
            kernel_ord_classification, i_Act_Pas, Fea_name, window_MTfeaClssification,...
            overlapp_MTfeaClssification, index_before, index_after);
        
        %% Regression step *************************************************************************
        for numRun = 1:Runs_time
            %% Cross validation *************************************************************************
            for i_Kfold = 1:K_fold
                ind_train = [];          ind_train =  Inds.ind_train_save{filenum, i_Act_Pas, i_Kfold, numRun};
                ind_test = [];            ind_test = Inds.ind_test_save{filenum, i_Act_Pas, i_Kfold, numRun};
                
                % State-Based
                [perf_classification_TEST_COT_Swich{filenum, i_Act_Pas, i_Kfold, numRun},...
                    perf_classification_COT_train_Swich{filenum, i_Act_Pas, i_Kfold, numRun},...
                    test_sample_length_Swich{filenum, i_Act_Pas, i_Kfold, numRun},...
                    movement_on_test_Swich{filenum, i_Act_Pas, i_Kfold, numRun},...
                    Class_test_Swich{filenum, i_Act_Pas, i_Kfold, numRun},...
                    Labels_Result_TEST_reg_Swich{filenum, i_Act_Pas, i_Kfold, numRun},...
                    y_predict_test_PLSr_Swich{filenum, i_Act_Pas, i_Kfold, numRun},...
                    y_predict_test_MLRr_Swich{filenum, i_Act_Pas, i_Kfold, numRun},...
                    response_test_Swich{filenum, i_Act_Pas, i_Kfold, numRun},...
                    fea_maxRs_MLRr_Swich{filenum, i_Act_Pas, i_Kfold, numRun},...
                    fea_maxRs_PLSr_Swich{filenum, i_Act_Pas, i_Kfold, numRun}, ...
                    Cofu_mat_TEST_Swich{filenum, i_Act_Pas, i_Kfold, numRun},...
                    Rsq_Feas_MLRr{filenum, i_Act_Pas, i_Kfold, numRun},...
                    Rsq_Feas_PLSr{filenum, i_Act_Pas, i_Kfold, numRun},...
                    Rsqtest{filenum, i_Act_Pas, i_Kfold, numRun}] =Regression_Step_inCOT_State_Based_Analysis...
                    (ind_train, ind_test, Classifier,K_fold_classification, selectionF, Feature_classify, Labels,...
                    Prep_data, Delay_sample, num_Lag, remove_very_corr_Feature,feature_regression_selection,...
                    K_fold_regression, PLS_mode, need_kernel_Regression, kernel_ord_Regression, second_model);
                
                fprintf(['StateBased - task: ', num2str(i_Act_Pas) ' / Run: ' , num2str(numRun), ' / fold: ' , num2str(i_Kfold), ' Done!\n'])
                
            end
        end
    end
    clear td_pas Prep_data
    
    %% Save *************************************************************************
    Results_Swich.K_fold = K_fold;
    Results_Swich.i_run = Runs_time;
    Results_Swich.num_Lag = num_Lag;
    Results_Swich.Class_test_Swich = Class_test_Swich;
    Results_Swich.Rsq_Feas_PLSr_Swich = Rsq_Feas_PLSr;
    Results_Swich.Rsq_Feas_MLRr_Swich = Rsq_Feas_MLRr;
    Results_Swich.response_test_Swich = response_test_Swich;
    Results_Swich.Cofu_mat_TEST_Swich = Cofu_mat_TEST_Swich;
    Results_Swich.movement_on_test_Swich = movement_on_test_Swich;
    Results_Swich.test_sample_length_Swich = test_sample_length_Swich;
    Results_Swich.y_predict_test_PLSr_Swich = y_predict_test_PLSr_Swich;
    Results_Swich.y_predict_test_MLRr_Swich = y_predict_test_MLRr_Swich;
    Results_Swich.fea_max.fea_maxRs_PLSr_Swich = fea_maxRs_PLSr_Swich;
    Results_Swich.fea_max.fea_maxRs_MLRr_Swich = fea_maxRs_MLRr_Swich;
    Results_Swich.Labels_Result_TEST_reg_Swich = Labels_Result_TEST_reg_Swich;
    Results_Swich.perf_classification_COT_train_Swich = perf_classification_COT_train_Swich;
    Results_Swich.perf_classification_TEST_COT_Swich = perf_classification_TEST_COT_Swich;
    
    Results_Swich.Classifier = Classifier;
    Results_Swich.num_Lag = num_Lag;
    Results_Swich.selectionF = selectionF;
    Results_Swich.Fea_name = Fea_name;
    Results_Swich.PLS_mode = PLS_mode;
    Results_Swich.index_after = index_after;
    Results_Swich.need_kernel = need_kernel;
    Results_Swich.index_before = index_before;
    Results_Swich.Delay_sample = Delay_sample;
    Results_Swich.second_model = second_model;
    Results_Swich.K_fold_regression = K_fold_regression;
    Results_Swich.K_fold_classification = K_fold_classification;
    Results_Swich.kernel_ord_Regression = kernel_ord_Regression;
    Results_Swich.need_kernel_Regression = need_kernel_Regression;
    Results_Swich.kernel_ord_classification = kernel_ord_classification;
    Results_Swich.remove_very_corr_Feature = remove_very_corr_Feature;
    Results_Swich.window_MTfeaClssification = window_MTfeaClssification;
    Results_Swich.feature_regression_selection = feature_regression_selection;
    Results_Swich.overlapp_MTfeaClssification = overlapp_MTfeaClssification;
    Results_Swich.window_statFeaClssification = window_statFeaClssification;
    Results_Swich.overlapp_statFeaClssification = overlapp_statFeaClssification;
    
    codename = mfilename;
    Pathsave = [path_Save, '\', 'Results', '\', codename];
    mkdir(Pathsave)
    Result_name_Swich = [codename, '_Swich_Sub' num2str(filenum), '_', num2str(Runs_time) , 'run_date_' date];
    save([Pathsave, '\', Result_name_Swich],'Results_Swich')
    clear Reults Result_name y_predict_test_PLSr y_predict_test_MLRr y_predict_test_LinM y_predict_test_tree response_test fea_maxRs_LinM fea_maxRs_tree fea_maxRs_MLRr fea_maxRs_PLSr movement_on_test Class_test thr_resp Cofu_mat_TEST perf_classification_TEST_COT perf_classification_COT test_sample_length
    
    fprintf(['Analysis of Dataset ' , num2str(filenum) , ' Done!\n'])
    
end

