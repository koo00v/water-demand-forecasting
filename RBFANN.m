clear all;

addpath('package_qmmp/');

% [status,sheets] = xlsfinfo('demand.xlsx'); % 엑셀 파일 불러오기
% raw_data = xlsread('demand.xlsx',sheets{1}); % raw 파일 불러오기
[status,sheets] = xlsfinfo('112.xlsx'); % 엑셀 파일 불러오기
raw_data = xlsread('112.xlsx',sheets{1}); % raw 파일 불러오기

FILE{1}.F = raw_data([1:end],1); %38 

nts = length(FILE);

for i = 1:nts
    FILE{i}.m = 24;
    FILE{i}.tau = 1;
end

TSPercentage = 0.7;
h = 24;
N = floor(length(FILE{1}.F)*(2/3))+h;
% N = 92;

for fix = 1:nts
    residuals = [];
    E = [];
    m = FILE{fix}.m;
    tau = FILE{fix}.tau;
    
    %normalize data from 0 to 1
    range = max(FILE{fix}.F) - min(FILE{fix}.F);
    TS = (FILE{fix}.F - min(FILE{fix}.F))./range; % Normalized data
    
    nInputs = (m-1)*tau+1;
    nOutputs = h;
    
    i = []; o = [];
    for ii = nInputs: length(TS)-(nOutputs)
        i = [i  TS(ii-(m-1)*tau:tau:ii)];
        o = [o  TS(ii+1:ii+h) ];
    end
    
    iTr = 1:floor(length(i)*TSPercentage);
    oTr = 1:floor(length(o)*TSPercentage);
    netS = newrb(i(:,iTr),o(:,iTr),0,1,N);
    
    for ix = floor(length(i)*TSPercentage)+1: length(i)
        ip = i(:,ix);
        yHat = netS(ip)*range+min(FILE{fix}.F);
        y = o(:,ix)*range+min(FILE{fix}.F);
    end
    
end

rmse = sqrt(mean((yHat-y).^2))


figure
plot(y)
hold on
plot(yHat,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")