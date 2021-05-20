clear all;

addpath('package_qmmp/');

[status,sheets] = xlsfinfo('demand.xlsx'); % 엑셀 파일 불러오기
raw_data = xlsread('demand.xlsx',sheets{1}); % raw 파일 불러오기
% [status,sheets] = xlsfinfo('112.xlsx'); % 엑셀 파일 불러오기
% raw_data = xlsread('112.xlsx',sheets{1}); % raw 파일 불러오기
FILE{1}.F = raw_data([1:end],401);

initial_p_max = 5;
initial_d_max = 1;
initial_q_max = 5;

TSPercentage = 0.7;
h = 24; 

% FILE{1}.M = arima('D',1, 'MALags',[1 2 3 4 5] ,'SMALags',[],'ARLags',[1 2 3 4 5]);
    
tsn = length(FILE);
for fix = 1:tsn
    residuals = [];
    E = [];
    fileName = FILE{fix}.F;
    
    % normalize data from 0 to 1
    range = max(FILE{fix}.F) - min(FILE{fix}.F);
    TS = (FILE{fix}.F - min(FILE{fix}.F))./range; % Normalized data
    lastTRData = floor(length(TS)*TSPercentage);
    train = (TS(1:lastTRData));
    para = checkArima(train,initial_p_max,initial_q_max);
    [FILE{fix}.p,FILE{fix}.q] = find(para==min(min(para)));
    FILE{fix}.M = arima(FILE{fix}.p,initial_d_max,FILE{fix}.q);
%     FILE{fix}.M = arima(5,initial_d_max,FILE{fix}.5);
    [EstMdl,EstParamCov] = estimate(FILE{fix}.M, TS(1:lastTRData));
    for d = lastTRData: length(TS)
        % sprintf( '%d / %d ', d,(N-h) )
         FILE{fix}.y = TS(d-h+1:d)*range+min(FILE{fix}.F);
        [yHat1,YMSE] = forecast(EstMdl,h,'Y0',TS(d-55:d));
         FILE{fix}.yHat = yHat1*range+min(FILE{fix}.F);
         FILE{fix}.rmse = sqrt(mean((FILE{fix}.yHat-FILE{fix}.y).^2));
    end

end


% figure
% plot(y)
% hold on
% plot(yHat,'.-')
% hold off
% legend(["Observed" "Predicted"])
% ylabel("Cases")
% title("Forecast with Updates")

