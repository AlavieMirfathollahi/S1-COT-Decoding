function [Prep_data, Feature_classify, Labels] = FeatureExtraction_for_Classification( Prep_data, need_kernel,...
    kernel_ord, i_Act_Pas, Fea_name, window_statFeaClssification, overlapp_statFeaClssification, index_before, index_after)
% INPUTs      ************************************************************************
%   Prep_data:                                  Data
%   need_kernel:                               Kernel for smoothing Data,  'gausswin' / 'tukeywin' / 'chebwinn' / 'gammakrl'
%   kernel_ord:                                 Order of Kernel
%   i_Act_Pas                                   Task, 1 for Active  /  2 for Passive
%   Fea_name                                   Feature group name, 'Statf'
%   window_statFeaClssification      Window length
%   overlapp_statFeaClssification    Overlap
%   index_before                               Start time of feature extraction before movement onset
%   index_after                                   End time of feature extraction after movement onset
% OUTPUTs   ************************************************************************
%   Prep_data:                                 Data of each Trial
%   Feature_classify:                          Features
%   Labels:                                      Labels of each Trial
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
data_classification = Prep_data;

%% Kernel for smoothing FR
if need_kernel == 'gausswin'
    for i_trial = 1:length(Prep_data)
        data_classification(i_trial).S1_FR = filtfilt(gausswin(kernel_ord) , 1 , Prep_data(i_trial).S1_FR);
    end
elseif need_kernel == 'tukeywin'
    for i_trial = 1:length(Prep_data)
        data_classification(i_trial).S1_FR = filtfilt(tukeywin(kernel_ord) , 1 , Prep_data(i_trial).S1_FR);
    end
elseif need_kernel == 'chebwinn'
    for i_trial = 1:length(Prep_data)
        data_classification(i_trial).S1_FR = filtfilt(chebwin(kernel_ord) , 1 , Prep_data(i_trial).S1_FR);
    end
elseif need_kernel == 'gammakrl'
    t=0:.01:1;     ts = 0;      alpha = 1.5;     beta = 20;
    Kernel_R = (beta^alpha)*((t-ts).^(alpha - 1)).*exp(-beta*(t-ts))/gamma(alpha);
    for i_trial = 1:length(Prep_data)
        coet_1 = sum(impz(Kernel_R , 1 ));
        data_classification(i_trial).S1_FR = filter(Kernel_R./coet_1 , 1 , Prep_data(i_trial).S1_FR);
    end
end

%% Classification Features and Labels
i_trial = 0;
for i_trial_all = 1:length(data_classification)
    i_trial = i_trial + 1;
    
    if i_Act_Pas == 1
        
        Data = data_classification(i_trial).S1_FR  (...
            data_classification(i_trial).idx_movement_on - index_before :...
            data_classification(i_trial).idx_movement_on + index_after , :   );
        
        % Feature Extraction
        if  Fea_name == 'Statf'
            [~, mea, ~, ~, ~, ~] = StatisticalFeature_double(Data, window_statFeaClssification, overlapp_statFeaClssification);
            Feature_classify(i_trial, :) =mea;
        end
        
        % Label of each trial
        if data_classification(i_trial).tgtDir == 0
            Labels(i_trial, 1) = 1;
        elseif data_classification(i_trial).tgtDir == 45
            Labels(i_trial, 1) = 2;
        elseif data_classification(i_trial).tgtDir == 90
            Labels(i_trial, 1) = 3;
        elseif data_classification(i_trial).tgtDir == 135
            Labels(i_trial, 1) = 4;
        elseif data_classification(i_trial).tgtDir == 180
            Labels(i_trial, 1) = 5;
        elseif data_classification(i_trial).tgtDir == 225
            Labels(i_trial, 1) = 6;
        elseif data_classification(i_trial).tgtDir == 270
            Labels(i_trial, 1) = 7;
        elseif data_classification(i_trial).tgtDir == 315
            Labels(i_trial, 1) = 8;
        end
        
    elseif i_Act_Pas == 2
        
        Data = data_classification(i_trial).S1_FR  (...
            data_classification(i_trial).idx_bumpTime - index_before :...
            data_classification(i_trial).idx_bumpTime + index_after , :   );
        
        % Feature Extraction
        if Fea_name == 'Statf'
            [~, mea, ~, ~, ~, ~] = StatisticalFeature_double(Data, window_statFeaClssification, overlapp_statFeaClssification);
            Feature_classify(i_trial, :) =mea;
        end
        
        % Label of each trial
        if data_classification(i_trial).bumpDir == 0
            Labels(i_trial, 1) = 1;
        elseif data_classification(i_trial).bumpDir == 45
            Labels(i_trial, 1) = 2;
        elseif data_classification(i_trial).bumpDir == 90
            Labels(i_trial, 1) = 3;
        elseif data_classification(i_trial).bumpDir == 135
            Labels(i_trial, 1) = 4;
        elseif data_classification(i_trial).bumpDir == 180
            Labels(i_trial, 1) = 5;
        elseif data_classification(i_trial).bumpDir == 225
            Labels(i_trial, 1) = 6;
        elseif data_classification(i_trial).bumpDir == 270
            Labels(i_trial, 1) = 7;
        elseif data_classification(i_trial).bumpDir == 315
            Labels(i_trial, 1) = 8;
        end
        
    end
end

% 4 Directions - Remove other classes
C2 = find(Labels==2);
C4 = find(Labels==4);
C6 = find(Labels==6);
C8 = find(Labels==8);

C_remove = [C2; C4; C6; C8];
Feature_classify(C_remove, :) = [];
Labels(C_remove) = [];
Prep_data(C_remove) = [];

end

