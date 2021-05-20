% sequence-to-sequence 회귀 LSTM: 숫자형 시퀀스로 구성된 Nx1 셀형 배열로, 여기서 N은 관측값의 개수
% R은 응답 변수의 개수


%% Load Sequence Data
clc;    % Clear the command window.
close all;  % Close all figures (except those of imtool.)
clear;  % Erase all existing variables. Or clearvars if you want.
clear global;

clear all; close all; clc;
% [status,sheets] = xlsfinfo('demand.xlsx'); % 엑셀 파일 불러오기
% raw_data = xlsread('demand.xlsx',sheets{1}); % raw 파일 불러오기
% data = raw_data(:,135)';

[status,sheets] = xlsfinfo('112.xlsx'); % 엑셀 파일 불러오기
raw_data = xlsread('112.xlsx',sheets{1}); % raw 파일 불러오기
data = raw_data(:,1)';


% FILE{1}.F =  'demands2012/p10015.txt';
% fileName = strcat(FILE{1}.F);
% raw_data = (readTS(fileName));
% 
% 
% data = raw_data';

% figure
% plot(data)
% xlabel("time(h)")
% ylabel("discharge")
% title("water consumption")
%% 
training = 0.7; %training set
total_length = round(length(data)/24*training);
numTimeStepsTrain = floor(total_length*24);

dataTrain = data(1:numTimeStepsTrain+1);
dataTest = data(numTimeStepsTrain+1:end);
%% Standardize Data

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;
%% Prepare Predictors and Responses
% To forecast the values of future time steps of a sequence, specify the responses 
% to be the training sequences with values shifted by one time step. That is, 
% at each time step of the input sequence, the LSTM network learns to predict 
% the value of the next time step. The predictors are the training sequences without 
% the final time step.

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);
%% *Define LSTM Network Architecture*
% Create an LSTM regression network. Specify the LSTM layer to have 
% 168 hidden units.
 
numFeatures = 1;
numResponses = 1;
numHiddenUnits1 = 168;
numHiddenUnits2 = 24;

% layers = [ ...
%     sequenceInputLayer(numFeatures)
%     lstmLayer(numHiddenUnits)
%     fullyConnectedLayer(numResponses)
%     regressionLayer];
layers = [ ...
sequenceInputLayer(numFeatures)
lstmLayer(numHiddenUnits1,'OutputMode','sequence')
dropoutLayer(0.2)
lstmLayer(numHiddenUnits2,'OutputMode','sequence')
dropoutLayer(0.2)
fullyConnectedLayer(numResponses)
regressionLayer];
%% Specify the training options
% Set the solver to |'adam'| and train for 250 epochs. 
% To prevent the gradients from exploding, set the gradient threshold  to 1. 
% Specify the initial learn rate 0.005, and drop the learn rate 
% after 125 epochs by multiplying by a factor of 0.2.

options = trainingOptions('adam', ...% Adaptive Moment Estimation: 적응적 모멘트 추정, 최적화 함수의 훈련 옵션 
    'MaxEpochs',250, ...
    'GradientThreshold',1, ...% 기울기 임계값
    'InitialLearnRate',0.005, ...% 초기학습률
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',125, ... % 학습률을 낮출 Epoch의 주기- 전역 학습률에 감소 인자를 곱함
    'LearnRateDropFactor',0.2, ...% 학습률 감소 인자
    'Verbose',0, ...
    'Plots','training-progress');
%% Train LSTM Network
% Train the LSTM network with the specified training options by using |trainNetwork|.

[net, info] = trainNetwork(XTrain,YTrain,layers,options);
%% Forecast Future Time Steps
% To forecast the values of multiple time steps in the future, use the |predictAndUpdateState| 
% function to predict time steps one at a time and update the network state at 
% each prediction. For each prediction, use the previous prediction as input to 
% the function.
% Standardize the test data using the same parameters as the training data.

dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end);
%% 
% To initialize the network state, first predict on the training data |XTrain|. 
% Next, make the first prediction using the last time step of the training response 
% |YTrain(end)|. Loop over the remaining predictions and input the previous prediction 
% to |predictAndUpdateState|.
% 
% For large collections of data, long sequences, or large networks, predictions 
% on the GPU are usually faster to compute than predictions on the CPU. Otherwise, 
% predictions on the CPU are usually faster to compute. For single time step predictions, 
% use the CPU. To use the CPU for prediction, set the |'ExecutionEnvironment'| 
% option of |predictAndUpdateState| to |'cpu'|.

[net, info2] = predictAndUpdateState(net,XTrain);
[net,YPred1] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred1(:,i)] = predictAndUpdateState(net,YPred1(:,i-1),'ExecutionEnvironment','cpu');
end
%% 
% Unstandardize the predictions using the parameters calculated earlier.

YPred1 = sig*YPred1 + mu;
%% 
% The training progress plot reports the root-mean-square error (RMSE) calculated 
% from the standardized data. Calculate the RMSE from the unstandardized predictions.

YTest = dataTest(1:end);
rmse1_1 = sqrt(mean((YPred1-YTest).^2));
ypred1_24 = YPred1(end-23:end)';
rmse1_2 = sqrt(mean((YPred1(end-23:end)-YTest(end-23:end)).^2));
%% 
% Plot the training time series with the forecasted values.

% figure
% plot(dataTrain(1:end-1))
% hold on
% idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
% plot(idx,[data(numTimeStepsTrain) YPred],'.-')
% hold off
% xlabel("Month")
% ylabel("Cases")
% title("Forecast")
% legend(["Observed" "Forecast"])
%% 
% Compare the forecasted values with the test data.

% figure
% subplot(2,1,1)
% plot(YTest)
% hold on
% plot(YPred,'.-')
% hold off
% legend(["Observed" "Forecast"])
% ylabel("Cases")
% title("Forecast")
% 
% subplot(2,1,2)
% stem(YPred - YTest)
% xlabel("Month")
% ylabel("Error")
% title("RMSE = " + rmse)
%% Update Network State with Observed Values
% If you have access to the actual values of time steps between predictions, 
% then you can update the network state with the observed values instead of the 
% predicted values.
% 
% First, initialize the network state. To make predictions on a new sequence, 
% reset the network state using |resetState|. Resetting the network state prevents 
% previous predictions from affecting the predictions on the new data. Reset the 
% network state, and then initialize the network state by predicting on the training 
% data.

net = resetState(net);
[net, info3] = predictAndUpdateState(net,XTrain);
%% 
% Predict on each time step. For each prediction, predict the next time step 
% using the observed value of the previous time step. Set the |'ExecutionEnvironment'| 
% option of |predictAndUpdateState| to |'cpu'|.

YPred2 = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred2(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end
%% 
% Unstandardize the predictions using the parameters calculated earlier.

YPred2 = sig*YPred2 + mu;
YPred2(YPred2 < 0) = 0;
%% 
% Calculate the root-mean-square error (RMSE).

rmse2_1 = sqrt(mean((YPred2-YTest).^2));

rmse2_2 = sqrt(mean((YPred2(end-23:end)-YTest(end-23:end)).^2));

ypred2_24 = YPred2(end-23:end)';
ytest_24 = YTest(end-23:end)';

figure
plot(ytest_24)
hold on
plot(ypred2_24,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")
%% 
% Compare the forecasted values with the test data.

figure
subplot(2,1,1)
plot(ytest_24)
hold on
plot(ypred2_24,'.-')
hold off
legend(["Observed" "Predicted"])
xlabel("Time (hr)")
ylabel("Discharge")
title("Forecast with Updates")

subplot(2,1,2)
stem(ypred2_24 - ytest_24)
xlabel("Time (hr)")
ylabel("Error")
title("RMSE = " + rmse2_2)
