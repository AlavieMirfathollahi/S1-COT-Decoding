%% COT   -   Conventional Analysis              
%*************************************************************************
% Alavie Mirfathollahi - Dec 2022 <alaviemir@gmail.com>
%*************************************************************************
% IMPORTANT NOTICE: If you use this code in your work, please cite [1].
%*************************************************************************
clc
clear all
clearvars
warning off

%% Inputs *************************************************************************
% Path Dataset (My PC)
dataroot = {path to Dataset folder};
% ind train/test
load({path to index of train and test data, 5 folds 10 runs});

% Regression
num_Lag = 1; % No lag
PLS_mode = 'AllResp'; % All responces as Y in PLS
K_fold_regression = 3;                                             feature_regression_selection = 0;
remove_very_corr_Feature = 1;                                  Delay_sample = 0;
second_model = 'MLR';                                            kernel_ord_Regression = 11;
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
K_fold = size(Inds.ind_train_save, 3);
Runs_time = size(Inds.ind_train_save, 4);     Run_threshold = 0;
y_predict_test_PLSr_General = cell(length(filenames), 2, K_fold);
y_predict_test_MLRr_General = cell(length(filenames), 2, K_fold);
response_test_General = cell(length(filenames), 2, K_fold);
trainall = cell(length(filenames), 2, K_fold);
testall = cell(length(filenames), 2, K_fold);
Class_test_General = cell(length(filenames), 2, K_fold);
fea_maxRs_MLRr_General = cell(length(filenames), 2, K_fold);
fea_maxRs_PLSr_General = cell(length(filenames), 2, K_fold);
movement_on_test_General = cell(length(filenames), 2, K_fold);

%% Run in each datafile *************************************************************************
for filenum = 1:length(filenames)
    
    td = load(fullfile(dataroot,'reaching_experiments', [filenames{filenum}]));
    [td_act, td_pas] = Preprocessing_COT(td);
    clear td
    
    %% Active vs passive
    for i_Act_Pas = 1:2
        
        if (i_Act_Pas == 1)
            Prep_data = td_act;
        elseif (i_Act_Pas == 2)
            Prep_data = td_pas;
            clear td_act
        end
        
        %% Regression step
        for numRun = 1:Runs_time
            
            %% Cross validation
            for i_Kfold = 1:K_fold
                ind_train = []; ind_test = [];
                
                ind_train =  Inds.ind_train_save{filenum, i_Act_Pas, i_Kfold, numRun};
                ind_test = Inds.ind_test_save{filenum, i_Act_Pas, i_Kfold, numRun};

                % Conventional
                [y_predict_test_PLSr_General{filenum, i_Act_Pas,  i_Kfold, numRun},...
                    y_predict_test_MLRr_General{filenum, i_Act_Pas,  i_Kfold, numRun},...
                    response_test_General{filenum, i_Act_Pas,  i_Kfold, numRun},...
                    fea_maxRs_MLRr_General{filenum, i_Act_Pas,  i_Kfold, numRun},...
                    fea_maxRs_PLSr_General{filenum, i_Act_Pas,  i_Kfold, numRun},...
                    movement_on_test_General{filenum, i_Act_Pas,  i_Kfold, numRun},...
                    Class_test_General{filenum, i_Act_Pas,  i_Kfold, numRun},...
                     Rsq_Feas_MLRr{filenum, i_Act_Pas, i_Kfold, numRun},...
                    Rsq_Feas_PLSr{filenum, i_Act_Pas, i_Kfold, numRun},...
                    Rsqtest{filenum, i_Act_Pas, i_Kfold, numRun}] =...
                    Regression_Step_Conventional(...
                    ind_train, ind_test, Prep_data, num_Lag,...
                    Delay_sample, remove_very_corr_Feature,...
                    feature_regression_selection, K_fold_regression, PLS_mode,...
                    need_kernel_Regression, kernel_ord_Regression, second_model);
                
                fprintf(['A3- task:', num2str(i_Act_Pas) ' / Run:' , num2str(numRun), ' / fold:' , num2str(i_Kfold), ' Done!\n'])
            end
        end
    end
    clear td_pas Prep_data
    
    %% Save *************************************************************************  
    Results_General.y_predict_test_PLSr_General = y_predict_test_PLSr_General;
    Results_General.y_predict_test_MLRr_General = y_predict_test_MLRr_General;
    Results_General.response_test_General = response_test_General;
    Results_General.movement_on_test_General = movement_on_test_General;
    Results_General.Class_test_General = Class_test_General;
    Results_General.i_run = Runs_time;
    Results_General.fea_max.fea_maxRs_MLRr_General = fea_maxRs_MLRr_General;
    Results_General.fea_max.fea_maxRs_PLSr_General = fea_maxRs_PLSr_General;
    Results_General.num_Lag = num_Lag;
    Results_General.K_fold = K_fold;
    Results_General.Rsq_Feas_MLRr = Rsq_Feas_MLRr;
    Results_General.Rsq_Feas_PLSr = Rsq_Feas_PLSr;
    Results_General.PLS_mode = PLS_mode;
    Results_General.K_fold_regression = K_fold_regression;
    Results_General.remove_very_corr_Feature = remove_very_corr_Feature;
    Results_General.feature_regression_selection = feature_regression_selection;
    Results_General.Delay_sample = Delay_sample;
    Results_General.second_model = second_model;
    Results_General.kernel_ord_Regression = kernel_ord_Regression;
    Results_General.need_kernel_Regression = need_kernel_Regression;
    
    % name & path save
    codename = mfilename;
    Pathsave = [path_Save, '\', 'Results', '\', codename];
    mkdir(Pathsave)
    
    Result_name_General = [codename, '_General_Sub' num2str(filenum), '_', num2str(Runs_time) , 'run_date_' date];
    save([Pathsave, '\', Result_name_General],'Results_General')
    
    clear Reults Result_name y_predict_test_PLSr y_predict_test_MLRr y_predict_test_LinM y_predict_test_tree response_test fea_maxRs_LinM fea_maxRs_tree fea_maxRs_MLRr  fea_maxRs_PLSr movement_on_test Class_test thr_resp Cofu_mat_TEST perf_classification_TEST_COT nperf_classification_COT test_sample_legth
    fprintf(['Analysis of Dataset ' , num2str(filenum) , ' Done!\n'])
    
end

%*************************************************************************
%   References
%   [1] A. Mirfathollahi, M.T. Ghodrati, V. Shalchyan, M. R. Zarrindast, M. R. Daliri,
%       "Decoding hand kinetics and kinematics using somatosensory cortex activity 
%       in active and passive movement", IScience, Aug. 2023
%*************************************************************************
