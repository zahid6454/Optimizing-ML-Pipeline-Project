# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
For this problem, we are given a Scikit-learn Logistic Regression model for which we have to tune the hyperparameters using the Hyperdive. Also, we are given an AutoML for building and optimizing a model. Finally, we have to compare the results of those two models. For this, we are given a training benchmark dataset called "bankmarketing_train.csv". The data in the dataset is related to bank marketing campaigns. The marketing campaigns of a bank were based on phone calls. Often, more than one contact to the same client was required, in order to access if the product (bank term deposit) would be (or not) subscribed. The classification goal is to predict if the client will subscribe a term deposit (variable y).

**In 1-2 sentences, explain the solution: e.g. "The best performing model was a ..."**

## Scikit-learn Pipeline
For this, we are given a "train.py" file, a "udacity-jupyter.ipynb" file, and a benchmark dataset "bankmarketing_train.csv" file. The dataset contains 32950 instances, 20 features or attributes and 1 class column.

For the "train.py" file: 
-   We used the benchmark dataset for training purpose. Using these data and features we have to classify whether client will subscribe a term 
    deposit or not. 
-   A data cleaning process takes place that converts all categorical value to numberical values.
-   Logistic Regression was used as a classfication algorithm. Two arguements or hyperparameter were used for Logistic Regression. One 
    is "--C", another is "--max_iter". These two arguments will be tuned using Hyperdrive.
-   Dataset splitted into train and test set and using the test set the model is trained. Test set was used to evaluate the trained model   
    based on metric "Accuracy" as our model is classification based.

For Hyperdrive in "udacity-jupyter.ipynb" file:
-   As a Parameter Sampler, RandomParameterSampling was used to tune the hyperparameter "--C" and "--max_iter".
-   As an Early Termination Policy, BanditPolicy was used with argument "evaluation_interval" and "slack_factor".
-   As estimator, SKlearn was used where "train.py" was provided as a value to an argument "entry_script". 

The goal of the Hyperdrive is to maximize the primary metric which is "Accuracy".

After the compltiong of the parameter tuning the best parameter was selected "--C" = 0.1 and "--max_iter" = 100 to achieve the maximum "Accuracy" = 0.91456686.

I have choosen RandomParameterSampling as parameter sampler. For tuning a model, initially RandomParameterSampler is a good choice as the hyperparameter values for the model are randomly selected from the defined search space. Also, supports discrete and continuous hyperparameters and early termination of low-performance runs.

I have choosen BanditPolicy as an early stopping policy. BanditPolicy uses slack factor or slack amount and evaluation interval for early stopping. BanditPolicy terminates any runs where the primary metric is not within the specified slack factor/slack amount with respect to the best performing training run.


## AutoML
For AutoML Configuration, we have choosen "classification" as task, "accuracy" as primary metric, "y" variable as label column, and finally 5 fold cross validation.
After using AutoML to select the best model, "VotingEnsemble" model was choosen by AutoML which is an ensemble machine learning model that combines the predictions from 
multiple other models. After completion of VotingEnsemble model execution, the Accuracy is calculated as 0.91756.

## Pipeline comparison
Upon comparison, VotingEnsemble which was choosen by AutoML has better "Accuracy" than Logistic Regression. Though the difference is very small (0.00299314) but VotingEnsemble outperformed Logistic Regression. The reason might be that in Hyperdrive as parameter sampler we are using RandomParameterSampling that chooses hyperparameter values for the model randomly. Also, in parameter sampler, we are passing 5 values for each of the hyperparameter/arguments. Adding more values might solve the issue if optimal combination of hyperparamter values are found. Where as AutoML is using predefined models to find the best model that maximizes primary metric.

## Future work
-   Changing classification model in train.py file, as for this dataset other classification algorithms might be more efficient such as SVM.
-   Adding more choices in hyperparameter for parameter sampler. In this case for every arguement, there were five different values. Adding 
    more values for each argument might be better for finding the most optimal parameter also adding new more arguments (if available) will also help.
-   Using different values for evaluation_interval and slack_factor might become useful.
-   Increasing the AutoML model timeout will be a better option.
-   Judging a model bu only Accuracy is not always optimal. Several other metrices such as AUROC, AUPR, MCC, Precision, Recall, F1-Score are 
    also useful to evaluate a model. So using different metric will be beneficial.

## Proof of cluster clean up
In the last cell of "udacity-project.ipynb" file, compute cluster was deleted using "AmlCompute.delete(cpu_cluster)". This proofs that clean up was done after the whole project was completed.
