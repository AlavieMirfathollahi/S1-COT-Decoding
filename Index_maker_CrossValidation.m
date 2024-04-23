%% COT   -   Index maker for CrossValidation step (both approaches)
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

%% Inputs
K_fold = 5;
Runs_time = 10;

dataroot = {path to Dataset folder};

%% Paths
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

%% Run in each datafile
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
        
        %% Classification Features and Labels
        data_classification = []; Labels =[]; sdxderiv2 = []; Feature_classi = [];
        data_classification = Prep_data;
        
        i_trial = 0;
        for i_trial_all = 1:length(data_classification)
            i_trial = i_trial + 1;
            
            if i_Act_Pas == 1
                % Label of each trial
                if data_classification(i_trial).tgtDir == 0,                     Labels(i_trial, 1) = 1;
                elseif data_classification(i_trial).tgtDir == 45,               Labels(i_trial, 1) = 2;
                elseif data_classification(i_trial).tgtDir == 90,               Labels(i_trial, 1) = 3;
                elseif data_classification(i_trial).tgtDir == 135,             Labels(i_trial, 1) = 4;
                elseif data_classification(i_trial).tgtDir == 180,             Labels(i_trial, 1) = 5;
                elseif data_classification(i_trial).tgtDir == 225,             Labels(i_trial, 1) = 6;
                elseif data_classification(i_trial).tgtDir == 270,             Labels(i_trial, 1) = 7;
                elseif data_classification(i_trial).tgtDir == 315,             Labels(i_trial, 1) = 8;
                end
                
            elseif i_Act_Pas == 2
                % Label of each trial
                if data_classification(i_trial).bumpDir == 0,                  Labels(i_trial, 1) = 1;
                elseif data_classification(i_trial).bumpDir == 45,            Labels(i_trial, 1) = 2;
                elseif data_classification(i_trial).bumpDir == 90,            Labels(i_trial, 1) = 3;
                elseif data_classification(i_trial).bumpDir == 135,          Labels(i_trial, 1) = 4;
                elseif data_classification(i_trial).bumpDir == 180,          Labels(i_trial, 1) = 5;
                elseif data_classification(i_trial).bumpDir == 225,          Labels(i_trial, 1) = 6;
                elseif data_classification(i_trial).bumpDir == 270,          Labels(i_trial, 1) = 7;
                elseif data_classification(i_trial).bumpDir == 315,          Labels(i_trial, 1) = 8;
                end
            end
        end
        
        % remove other classes
        C2 = find(Labels==2);                                       C4 = find(Labels==4);
        C6 = find(Labels==6);                                       C8 = find(Labels==8);
        C_remove = [C2; C4; C6; C8];                           Labels(C_remove) = [];
        
        %% Cross validation
        for numRun = 1:Runs_time
            hpartition = [];                                            hpartition = cvpartition(Labels, 'KFold', K_fold);
            for i_Kfold = 1:K_fold
                ind_train = []; ind_test = [];
                ind_train_save{filenum, i_Act_Pas, i_Kfold, numRun} = training(hpartition, i_Kfold);
                ind_test_save{filenum, i_Act_Pas, i_Kfold, numRun} = test(hpartition, i_Kfold);
            end
        end
    end
    clear td_pas Prep_data
end

%% Save
Inds.ind_train_save = ind_train_save;
Inds.ind_test_save = ind_test_save;

% name & path save
codename = mfilename;
Pathsave = [path_Save, '\', 'Results', '\', codename];
mkdir(Pathsave)

Result_name_General = [codename, '_' , num2str(K_fold), 'folds_' num2str(Runs_time) , 'runs' ];
save([Pathsave, '\', Result_name_General],'Inds')
