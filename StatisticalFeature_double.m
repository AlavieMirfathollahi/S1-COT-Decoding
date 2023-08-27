function [median_feature, mean_feature, variance_feature, kur_feature, skew_feature, power_feature] =...
    StatisticalFeature_double(data_single_trial, windowLength, overlapp)
% INPUTs      ************************************************************************
%   data_single_trial:                          data of each single trial
%   windowLength:                            Window length for feature Extraction
%   overlapp:                                    overlap for feature Extraction
% OUTPUTs   ************************************************************************
%   median_feature:                             Extracted Feature, Median of neural activity in windows of each channel
%   mean_feature:                               Extracted Feature, Mean of neural activity in windows of each channel
%   variance_feature:                            Extracted Feature, Var of neural activity in windows of each channel
%   kur_feature:                                  Extracted Feature, kurtosis of neural activity in windows of each channel
%   skew_feature:                                Extracted Feature, skewness of neural activity in windows of each channel
%   power_feature:                              Extracted Feature, Power of neural activity in windows of each channel
%*************************************************************************
% Alavie Mirfathollahi - 2022 <alaviemir@gmail.com>
%*************************************************************************
% IMPORTANT NOTICE: If you use this code in your work, please cite [1].
%*************************************************************************
%   References
%   [1] A. Mirfathollahi, M.T. Ghodrati, V. Shalchyan, M. R. Zarrindast, M. R. Daliri,
%       "Decoding hand kinetics and kinematics using somatosensory cortex activity
%       in active and passive movement", iScience, Aug. 2023
%*************************************************************************
median_feature =[]; mean_feature = []; variance_feature = []; power_feature = []; variance_feature = []; kur_feature = []; skew_feature = [];
for j = 1:windowLength:size(data_single_trial,1)
    y = data_single_trial(j:min([j+(windowLength-overlapp), size(data_single_trial,1)]),:);
    
    median_feature = [median_feature, median(y)];
    mean_feature = [mean_feature, mean(y)];
    variance_feature = [variance_feature, var(y)];
    power_feature = [power_feature, sum(y.*y)/size(y,1)];
    kur_feature = [kur_feature, kurtosis(y)];
    skew_feature = [skew_feature, skewness(y)];
end

end

