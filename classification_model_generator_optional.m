function [Mdl_classification, num_best_Fea, features, perf_classification_COT] =...
    classification_model_generator_optional(Classifier, selectionF, data_classification,...
    ind_train_main, K_fold, Labels)
% INPUTs      ************************************************************************
%   Classifier:                              Classifier name,   'KNN'  /   'LDA'  /  'Tre'  /  'SVM'  /  'Vot'
%   selectionF:                             Feature Selection Method, 'MI'  /  'SD'  /   'SF'
%   data_classification:                  data
%   ind_train_main:                      Index of Train Data
%   K_fold:                                  folds (train / test)
%   Labels:                                  Labels
% OUTPUTs   ************************************************************************
%   Mdl_classification:                    Classification Model
%   num_best_Fea:                         # of opt Features
%   features:                                Index of sorted Features
%   perf_classification_COT:             Accuracy of train data
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
steps_of_Features = 5;
LDA_Type = 'pseudoLinear';                                       %  'linear' / 'pseudoquadratic'
K_KNN = 5;
Q = 12;

template = templateSVM(...
    'KernelFunction', 'gaussian', ...
    'PolynomialOrder', [], ...
    'KernelScale', 36, ...
    'BoxConstraint', 1, ...
    'Standardize', true);

%% classification
Feature_classify_classification = data_classification(ind_train_main,:);
Labels_classification = Labels(ind_train_main);

hpartition = cvpartition(Labels_classification, 'KFold', K_fold); % Nonstratified partition

for num_Kfold = 1:K_fold
    ind_train = training(hpartition, num_Kfold);
    ind_test = test(hpartition, num_Kfold);
    
    %% Feature Selection
    if selectionF == 'SD'
        [features, ~] = SD(Feature_classify_classification(ind_train,:), Labels_classification(ind_train), Q);
    elseif selectionF == 'MI'
        [features, ~] = MI(Feature_classify_classification(ind_train,:), Labels_classification(ind_train));
    elseif selectionF == 'SF'
        for n_fea = 1:length(Feature_classify_classification(1,:)) % should all Features added!
            cp = classperf(Labels_classification);
            Y=[];
            Y=Feature_classify_classification(:,n_fea);
            if(Classifier=='KNN')
                Mdl_classification = fitcknn(Y(ind_train, :),Labels_classification(ind_train), 'NumNeighbors', K_KNN); %'OptimizeHyperparameters','auto');
            elseif(Classifier=='LDA')
                Mdl_classification = fitcdiscr(Y(ind_train, :),Labels_classification(ind_train),'discrimType', LDA_Type);
            elseif(Classifier=='Tre')
                Mdl_classification = fitctree(Y(ind_train, :),Labels_classification(ind_train));
            elseif(Classifier=='SVM')
                Mdl_classification = fitcecoc(Y(ind_train, :),Labels_classification(ind_train),...
                    'Learners', template, ...
                    'Coding', 'onevsone',...
                    'ClassNames', unique(Labels));
            elseif(Classifier=='Vot')
                myopts.ShowPlots = false;
                Mdl_classification_knn = fitcknn(Y(ind_train, :),Labels_classification(ind_train),'NumNeighbors', K_KNN); % 'OptimizeHyperparameters','auto', 'HyperparameterOptimizationOptions',myopts);
                Mdl_classification_lda = fitcdiscr(Y(ind_train, :),Labels_classification(ind_train),'discrimType', LDA_Type );
                Mdl_classification_SVM = fitcecoc(Y(ind_train, :),Labels_classification(ind_train),...
                    'Learners', template, ...
                    'Coding', 'onevsone',...
                    'ClassNames', unique(Labels));
                
                class_knn = predict(Mdl_classification_knn,Y(ind_test, :));
                class_lda = predict(Mdl_classification_lda,Y(ind_test, :));
                class_svm = predict(Mdl_classification_SVM,Y(ind_test, :));
                
                for i_class = 1:length(class_knn)
                    Determ = [class_knn(i_class), class_lda(i_class), class_svm(i_class)];
                    class = mode(Determ);
                end
            end
            
            if(Classifier~='Vot')
                class = predict(Mdl_classification,Y(ind_test, :));
            end
            
            class1 = predict(Mdl_classification,Y(ind_test,:));
            classperf(cp, class1, ind_test);
            perf_singleFeature(n_fea)=(1-cp.ErrorRate)*100;
        end
        
        [~,features] = sort(perf_singleFeature,'descend');
    end
    
    %% Classification
    for n_fea = 1 :steps_of_Features: min(length(Labels_classification), length(Feature_classify_classification(1,:)))
        
        Y = [];
        indx_f = features(1:n_fea);
        Y = Feature_classify_classification(:,indx_f);
        
        if(Classifier=='KNN')
            myopts.ShowPlots = false;
            Mdl_classification = fitcknn(Y(ind_train, :),Labels_classification(ind_train),'NumNeighbors', K_KNN); % 'OptimizeHyperparameters','auto', 'HyperparameterOptimizationOptions',myopts);
        elseif(Classifier=='LDA')
            Mdl_classification = fitcdiscr(Y(ind_train, :),Labels_classification(ind_train),'discrimType', LDA_Type );
        elseif(Classifier=='Tre')
            Mdl_classification = fitctree(Y(ind_train, :),Labels_classification(ind_train));
        elseif(Classifier=='SVM')
            Mdl_classification = fitcecoc(Y(ind_train, :),Labels_classification(ind_train),...
                'Learners', template, ...
                'Coding', 'onevsone',...
                'ClassNames', unique(Labels));
        elseif(Classifier=='Vot')
            myopts.ShowPlots = false;
            Mdl_classification_knn = fitcknn(Y(ind_train, :),Labels_classification(ind_train),'NumNeighbors', K_KNN); % 'OptimizeHyperparameters','auto', 'HyperparameterOptimizationOptions',myopts);
            Mdl_classification_lda = fitcdiscr(Y(ind_train, :),Labels_classification(ind_train),'discrimType', LDA_Type );
            Mdl_classification_SVM = fitcecoc(Y(ind_train, :),Labels_classification(ind_train),...
                'Learners', template, ...
                'Coding', 'onevsone',...
                'ClassNames', unique(Labels));
            
            clear class_knn class_lda class_svm Determ class
            class_knn = predict(Mdl_classification_knn,Y(ind_test, :));
            class_lda = predict(Mdl_classification_lda,Y(ind_test, :));
            class_svm = predict(Mdl_classification_SVM,Y(ind_test, :));
            
            for i_class = 1:length(class_knn)
                Determ = [class_knn(i_class), class_lda(i_class), class_svm(i_class)];
                class(i_class) = mode(Determ);
            end
        end
        
        if(Classifier~='Vot')
            class = predict(Mdl_classification,Y(ind_test, :));
        end
        
        cp = classperf(Labels_classification);
        classperf(cp,class,ind_test);
        perf_classification_COT(num_Kfold, n_fea)=(1-cp.ErrorRate)*100;
        
    end
end

perf_classification_COT = mean(perf_classification_COT);
[~, num_best_Fea] = max(perf_classification_COT);

%% Final Model
if(Classifier=='KNN')
    Mdl_classification = fitcknn(Y(:, 1: num_best_Fea),Labels_classification,'NumNeighbors', K_KNN); %'OptimizeHyperparameters','auto', 'HyperparameterOptimizationOptions',myopts);
elseif(Classifier=='LDA')
    Mdl_classification = fitcdiscr(Y(:, 1:num_best_Fea),Labels_classification,'discrimType',LDA_Type );
elseif(Classifier=='Tre')
    Mdl_classification = fitctree(Y(:, 1:num_best_Fea),Labels_classification);
elseif(Classifier=='SVM')
    Mdl_classification = fitcecoc(Y(ind_train, :),Labels_classification(ind_train),...
        'Learners', template, ...
        'Coding', 'onevsone',...
        'ClassNames', unique(Labels));
elseif(Classifier=='Vot')
    myopts.ShowPlots = false;
    Mdl_classification_knn = fitcknn(Y(ind_train, :),Labels_classification(ind_train),'NumNeighbors', K_KNN); % 'OptimizeHyperparameters','auto', 'HyperparameterOptimizationOptions',myopts);
    Mdl_classification_lda = fitcdiscr(Y(ind_train, :),Labels_classification(ind_train),'discrimType', LDA_Type );
    Mdl_classification_SVM = fitcecoc(Y(ind_train, :),Labels_classification(ind_train),...
        'Learners', template, ...
        'Coding', 'onevsone',...
        'ClassNames', unique(Labels));
    
    class_knn = predict(Mdl_classification_knn,Y(ind_test, :));
    class_lda = predict(Mdl_classification_lda,Y(ind_test, :));
    class_svm = predict(Mdl_classification_SVM,Y(ind_test, :));
    
    for i_class = 1:length(class_knn)
        Determ = [class_knn(i_class), class_lda(i_class), class_svm(i_class)];
        class = mode(Determ);
    end
    
    Mdl_classification = {Mdl_classification_knn, Mdl_classification_lda, Mdl_classification_SVM};
end

end


