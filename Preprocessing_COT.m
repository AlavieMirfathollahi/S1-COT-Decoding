function [td_act, td_pas] = Preprocessing_COT(td)
% INPUTs      ************************************
% td:            Input COT Data
% OUTPUTS   ************************************
% td_act:       Firing Rate and movement parameters in Active task
% td_pas:       Firing Rate and movement parameters in Passive task
%*************************************************************************
% Alavie Mirfathollahi - Dec 2022 <alaviemir@gmail.com>
%*************************************************************************
% IMPORTANT NOTICE: If you use this code in your work, please cite [1].
%*************************************************************************
%   References
%   [1] A. Mirfathollahi, M.T. Ghodrati, V. Shalchyan, M. R. Zarrindast, M. R. Daliri,
%       "Decoding hand kinetics and kinematics using somatosensory cortex activity
%       in active and passive movement", IScience, Aug. 2023
%*************************************************************************
td = td.trial_data;

%%  Preprocessing data
% process marker data
for trialnum = 1:length(td)
    markernans = isnan(td(trialnum).markers);
    td(trialnum).markers(markernans) = 0;
    td(trialnum) = smoothSignals(td(trialnum),struct('signals','markers'));
    td(trialnum).markers(markernans) = NaN;
    clear markernans
end

% get marker velocity
td = getDifferential(td,struct('signals','markers','alias','marker_vel'));

% get speed and ds
td = getNorm(td,struct('signals','vel','field_extra','_norm'));
td = getDifferential(td,struct('signals','vel_norm','alias','dvel_norm'));

% prep trial data by getting only rewards and trimming to only movements
% split into trials
td = splitTD(...
    td,...
    struct(...
    'split_idx_name','idx_startTime',...
    'linked_fields',{{...
    'trialID',...
    'result',...
    'bumpDir',...
    'tgtDir',...
    'ctrHoldBump',...
    'ctrHold',...
    }},...
    'start_name','idx_startTime',...
    'end_name','idx_endTime'));

[~,td] = getTDidx(td,'result','R');
td = reorderTDfields(td);

% clean nans out...?
nan_inds = isnan(cat(1,td.tgtDir));
td = td(~nan_inds);
unacc_trials = cat(1,td.ctrHoldBump) & abs(cat(1,td.bumpDir))>360;
td = td(~unacc_trials);

% remove trials where markers aren't present
bad_trial = false(length(td),1);
for trialnum = 1:length(td)
    if any(any(isnan(td(trialnum).markers)))
        bad_trial(trialnum) = true;
    end
end
td(bad_trial) = [];

% remove trials where muscles aren't present
bad_trial = false(length(td),1);
for trialnum = 1:length(td)
    if any(any(isnan(td(trialnum).muscle_len) | isnan(td(trialnum).muscle_vel)))
        bad_trial(trialnum) = true;
    end
end
td(bad_trial) = [];

%% for C_20170912, trial structure is such that active and passive are part of the same trial--split it up
if strcmpi(td(1).monkey,'C') && contains(td(1).date_time,'2017/9/12')
    td_copy = td;
    [td_copy.ctrHoldBump] = deal(false);
    td = cat(2,td,td_copy);
    clear td_copy
end

%% split into active and passive
[~,td_act] = getTDidx(td,'ctrHoldBump',false);
[~,td_pas] = getTDidx(td,'ctrHoldBump',true);

clear td

% find the relevant movmement onsets
td_act = getMoveOnsetAndPeak(td_act,struct(...
    'start_idx','idx_goCueTime',...
    'start_idx_offset',20,...
    'peak_idx_offset',20,...
    'end_idx','idx_endTime',...
    'method','peak',...
    'peak_divisor',10,...
    'min_ds',1));
td_pas = getMoveOnsetAndPeak(td_pas,struct(...
    'start_idx','idx_bumpTime',...
    'start_idx_offset',-5,... % give it some wiggle room
    'peak_idx_offset',-5,... % give it some wiggle room
    'end_idx','idx_goCueTime',...
    'method','peak',...
    'peak_divisor',10,...
    'min_ds',1));

% throw out all trials where bumpTime and movement_on are more than 3 bins apart
bad_trial = isnan(cat(1,td_pas.idx_movement_on)) | abs(cat(1,td_pas.idx_movement_on)-cat(1,td_pas.idx_bumpTime))>3;
td_pas = td_pas(~bad_trial);

%% FR
td_act = addFiringRates(td_act,struct('array','S1'));
td_pas = addFiringRates(td_pas,struct('array','S1'));

fprintf('Data Loaded and Preprocessed \n')

end
