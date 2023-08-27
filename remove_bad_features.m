function [predictors, predictors_test] = remove_bad_features(predictors, predictors_test)
% INPUTs      ************************************************************************
%   predictors:              Neural features, Train data
%   predictors_test:        Neural features, Test data
% OUTPUTs   ************************************************************************
%   predictors:              Neural features, Train data
%   predictors_test:        Neural features, Test data
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
Thr_remove = 0.98;

%% Remove zero Features
std_features = std(predictors);
zero_std_fea = find(std_features == 0);
if isempty(zero_std_fea) ~=1
    predictors(:, zero_std_fea) = [];
    predictors_test(:, zero_std_fea) = [];
end

%% Remove corr Features
Corr_predictors = abs(corr(predictors));
[row_high_corr, ~] = find( (triu(Corr_predictors) - eye(size(Corr_predictors))) > Thr_remove);
unique_row_high_corr = unique(row_high_corr);
if isempty(unique_row_high_corr) ~= 1
    predictors(:, unique_row_high_corr) = [];
    predictors_test(:, unique_row_high_corr) = [];
end

end