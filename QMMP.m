clear all; close all; clc;

% [status,sheets] = xlsfinfo('demand.xlsx'); % 엑셀 파일 불러오기
% raw_data = xlsread('demand.xlsx',sheets{1}); % raw 파일 불러오기
[status,sheets] = xlsfinfo('112.xlsx'); % 엑셀 파일 불러오기
raw_data = xlsread('112.xlsx',sheets{1}); % raw 파일 불러오기

no_m = length(raw_data(1,:));
Holidays = [1 46 47 60 142 157 227 267 268 276 282 359];
training = 0.7; %training set
H = 2; % prediction horizon daily scale
tau = 1; % tau
gamma = 24; % gamma
epsilonRanges = 0:0.1:1; % epsilon Ranges of Eq ()
mRanges = 1:20; % m ranges of Eq ()
mma = 1:20; % m' ranges of Eq. ()
ncmin = 2; ncmax = 2; % number of clusters tested
repetitions = 100; % 계산 반복 수

for fix = 1 %: no_m  % 수용가 개수
    FILE{fix}.F = raw_data([1:end],1);
    % FILE{fix}.F = raw_data([1:end],fix);
    FILE{fix}.M = arima('D',1, 'MALags',[1 2 3] ,'Seasonality',7,'SMALags',[1],'ARLags',[1 2 3]);
    FILE{fix}.T = {fix};

    Errors_QMMP = []; Errors_Cal = []; Errors_Knn = []; Errors_Naive = []; Errors_Pp = [];
    residuals_QMMP = []; residuals_Cal = []; residuals_Knn = []; residuals_Naive = []; residuals_Pp = [];
    YY1_total = []; YY2_total = [];

    %normalize data from 0 to 1
    range = max(FILE{fix}.F) - min(FILE{fix}.F);
    FILE{fix}.TN = (FILE{fix}.F - min(FILE{fix}.F))./range; % Normalized data
    trainingDaysNumber = round((length(FILE{fix}.TN)/gamma)*training); % training 날짜 수

    % Learns unitary patterns of the time series using k means
    totalDays = length(FILE{fix}.TN)/gamma; % 총 날짜 수
    dailyPatterns = reshape(FILE{fix}.TN,gamma,totalDays); % 전체 물 사용 패턴
%     dailyPatterns_m = mean(dailyPatterns');
%     x = [1:1:24];
%     y = dailyPatterns_m;
%     f = fit(x.',y.','gauss5');
%     plot(f,x,y);
    
    q_day_total = sum(dailyPatterns)'; % 일별 물사용량 합산
    q_day_total_repmat = repmat(q_day_total',24,1); %일별 물사용량 반복
    q_hourly_pattern = dailyPatterns./q_day_total_repmat; % 전 기간 물사용 비율
    
    [TF,LL,UU,CCC] = isoutlier(q_day_total,'grubbs');
    q_day_total2 = q_day_total;
    q_day_total2(TF == 1) = NaN;
    
    %일별 SARIMA 계수 추정 및 Ljung-Box Q-test for residual autocorrelation
    [arima_para1,EstParamCov1] = estimate(FILE{fix}.M,q_day_total); % SARIMA 계수 추정 (전체)
    
    [YYx,YMSEx] = forecast(arima_para1,365,q_day_total2);
    
    errPredCons = [];
    %training
    for i1 = 15 : length(q_day_total)*training % from 50 to end point training - H (255)
        [YY1,YMSE1] = forecast(arima_para1,1,'Y0',q_day_total(i1-14:i1));
        %consecutive forecasted responses YY1, mean square errors YMSE1
        YY1_total = [YY1_total; YY1];
        errPredCons = [errPredCons YY1(1) - q_day_total(i1+1)];
    end
    passed = lbqtest(errPredCons); %Ljung-Box Q-test

    %Forecasting (training 이후)
    start = floor(length(q_day_total)*training)+1; %시작 지점
    sarimafcast= [];
    for i2 = start:length(q_day_total)
        [YY2,YMSE2] = forecast(arima_para1,2,'Y0',q_day_total(i2-14:i2));
        YY2_total = [YY2_total; YY2];
        sarimafcast(1,i2-start+1) = YY2(1);
        sarimafcast(2,i2-start+1) = YY2(2);
    end
    
    % SARIMA 계수 추정 1:trainingDaysNumber
    [arima_para2, EstParamCov2] = estimate(FILE{fix}.M, q_day_total(1:trainingDaysNumber,:));
    
    %Find the most suitable number of classes using silhuette index
    K = []; % 군집 수
    msi = -1; % silhuette_evaluation
    M = [];  P = []; % P: 패턴
   
    for k = ncmin : ncmax % sumdist = 점 - 중심 간 거리의 군집 내 합
        [classes, P, sumdist] = kmeans(q_hourly_pattern(:,1:trainingDaysNumber)',k,'distance','city','display','final','replicates',repetitions);
        [silhuetteIndex, nomm]= silhouette(q_hourly_pattern(:,1:trainingDaysNumber)', classes); %silhuette Index
        M = [M mean(silhuetteIndex)];
        K = k;
        C1 = classes; % 클러스터링 군집 결과_1 to training point
        if mean(silhuetteIndex) > msi
            msi = mean(silhuetteIndex);
        end
    end
    
    FILE{fix}.K = K; 
    
    C2 = zeros(totalDays - trainingDaysNumber,1); %training 이후
    for i3 = trainingDaysNumber + 1 : totalDays
        [a1, b1] = min(sum((repmat(q_hourly_pattern(:,i3),1,K) - P').^2));
        C2(i3 - trainingDaysNumber) = b1;
        p_min(i3 - trainingDaysNumber) = a1;
    end
    
    C = [C1; C2]; % 일별 군집 결과
    
    auxClasses = C;
    auxP = P; % 24시간별 패턴
    
    % 군집 개수 많은 순으로 정렬
    k_f = []; 
    for k_number = 1 : K
        k_f(k_number,1) = k_number;
        k_f(k_number,2) = sum(C == k_number);
    end
    k_sort = sortrows(k_f,2, 'descend');
    for k_number = 1 : K
        C(find(auxClasses == k_sort(k_number,1))) = k_number;
    end

    %% sliding window, find kNN parameters
    fit = zeros(length(mRanges),length(epsilonRanges)); 
    for mR = mRanges % 1:20
        nInputs1 = (mR-1)*tau+1; % 1:20
        ip = []; op = [];
        for it = nInputs1: length(C) - H % 1 to 363
            ip = [ip C(it-(mR-1)*tau:tau:it)]; % 20*344
            op = [op C(it+1:it+H)]; % 2*344
        end
        
        s1 = floor(size(ip,2)*training*training); % 168
        f1 = floor(size(ip,2)*training); % 240
        ex = 1;
        
        for epsilon1 = epsilonRanges % 0:0.1:1
            error = 0;
            for j1 = s1 : f1 % 168:240
                Vector = ip(:,j1); % 20*1
                
                % prediction kNN
                ni1 = size(ip(:,1:j1-H),2); % 238
                nj1 = size(ip(:,1:j1-H),1); % 20
                distances1 = sum(repmat(Vector,1,ni1) ~= ip(:,1:j1-H))/nj1; % 1*238
                ix1 = find(distances1 < epsilon1);
                if length(ix1) == 0
                    [v1,ix1] = min(distances1);
                    ix1 = [ix1 ix1]; % 1*238
                end
                if length(ix1) == 1
                    ix1 = [ix1 ix1];
                end
                oo = op(:,1:j1-H); % 2*238
                modePrediction1 = mode(oo(:,ix1)'); 
                error = error + sum(op(:,j1) ~= modePrediction1'); 
            end
            fit(nj1,ex) =  error; % 20*11
            ex = ex+1;
        end
    end
    
    [ax, bx] = min(fit);
    [a2, b2] = min(ax);
    
    [a3, b3] = min(fit(:,b2));
    epsilon2 = b2*0.1-0.1;
    
    FILE{fix}.m = b3;
    FILE{fix}.eps = epsilon2;
    
    nInputs2 = (b3-1)*tau+1;  % (19-1)*1+1
    im = [];
    om = [];
    
    for i4 = nInputs2 : length(C)- H % 19:363
        im = [im C(i4-(b3-1)*tau:tau:i4)]; % 19*345
        om = [om C(i4+1:i4+H)]; % 2*345
    end
    
    s2 = floor(size(im,2)*training*training); % 169
    f2 = floor(size(im,2)*training); % 241
    
    bTable1 = []; 
    
    %% get Best MA
    CC = C(1:trainingDaysNumber); % 1:255
    q_hourly_pattern_training = q_hourly_pattern(:,1:trainingDaysNumber);% 24*255
    
    errorMA = zeros(1,max(mma)); % 1*20
    iv = 1;
    ma = 1;
    besetMAVal = 999;
    
    PP_Total = [];
    for mm = mma % 1:20
        for i5 = floor(length(CC)*training):floor(length(CC))-1 % 178:254
            PP = CC(i5+1); % 1*1540
            PP_Total = [PP_Total; PP'];
            oneClass = q_hourly_pattern_training(:,CC(1:i5) == PP); 
            
            predictionPattern = (mean(oneClass(:,end-mm:end)')/sum(mean(oneClass(:,end-mm:end)')))';
            errorMA(iv) = errorMA(iv) + mean(abs(q_hourly_pattern_training(:,i5+1)-predictionPattern));
        end
        if  errorMA(iv) < besetMAVal
            ma = mm;
            besetMAVal = errorMA(iv);
        end
        iv = iv+1;
    end
    
    FILE{fix}.MA = ma;
 
    %%
    for d1 = mm + H : f2 % 22 : 252
        currentVector1 = im(:,d1-mm+1);
        realFutureModes1 =  C(d1+1:d1+H);
        
        % prediction kNN 1
        jx1 =  im(:,1:d1-mm+1-H);
        kx1 =  om(:,1:d1-mm+1-H);
        
        ni2 = size(jx1,2);
        nj2 = size(jx1,1);
        distances2 = sum(repmat(currentVector1,1,ni2)~=jx1)/nj2;
        ix2 = find(distances2 < epsilon2);
        if length(ix2) == 0
            [v2,ix2] = min(distances2);
            ix2 = [ix2 ix2];
        end
        if length(ix2)==1
            ix2 = [ix2 ix2];
        end
        modePrediction2 = mode(kx1(:,ix2)');
        
        % calendar Prediction 1
        calP1 = zeros(H,1);
        cont1 = 0;
        for j2 = d1+1:d1+2 % d1 = mm + H : f2
            cont1 = cont1+1;
            dayNumber1 = j2;
            day1 = mod(dayNumber1,7);
            if ~isempty(find(Holidays==dayNumber1, 1)) || day1==1 || day1==0
                if C(1)==1
                    calP1(cont1)=1;
                else
                    calP1(cont1)=2;
                end
            else
                if C(1)==1
                    calP1(cont1)=2;
                else
                    calP1(cont1)=1;
                end
            end
        end
        
        for j3 = 0 : gamma-1
            hour1 = d1*gamma+j3;
            periodSection1 = mod(hour1-1,gamma)+1;
            lagV1 = FILE{fix}.TN(hour1-gamma+1:hour1); % 물사용량
            lagV2 = lagV1/sum(lagV1); % 물사용 비율
            
            % nearest Pattern
            global ov1;
            global ds1;
            K1 = size(P,1);
            tMag1 = sum(lagV2);
            daySample1 = lagV2/tMag1;
            ds1 = daySample1;
            observedVals1 = daySample1(end-periodSection1+1:end);
            nbDist1 = (P(:,1:periodSection1)'- repmat(observedVals1,1,K1)).^2;
            ov1 = observedVals1;
            
            if periodSection1==1
                [a4,b4] = min(nbDist1);
            else
                [a4,b4] = min(sum(nbDist1));
            end
            
            dist1 = a4;
            k1 = b4;
            bTable1 = [bTable1; mod(d1,7)+1 mod(hour1,gamma)+1 k1 modePrediction2 calP1' realFutureModes1'];
        end
    end
    
    for d2 = f2+1 : length(C) - H % 242:363
        currentVector2 = im(:,d2-mm+1); % 19*1
        realFutureModes2 =  C(d2+1:d2+H);

        % prediction kNN 2
        jx2 = im(:,1:d2-mm+1-H);
        kx2 = om(:,1:d2-mm+1-H);
        ni3 = size(jx2,2);
        nj3 = size(jx2,1);
        distances3 = sum(repmat(currentVector2,1,ni3)~=jx2)/nj3;
        ix3 = find(distances3 < epsilon2);
        if length(ix3) == 0
            [v3,ix3] = min(distances3);
            ix3 = [ix3 ix3];
        end
        if length(ix3)==1
            ix3 = [ix3 ix3];
        end
        modePrediction3 = mode(kx2(:,ix3)');
        
        % calendar Prediction 2
        calP2 = zeros(H,1);
        cont2 = 0;
        
        for j4 = d2+1:d2+2
            cont2 = cont2+1;
            dayNumber2 = j4;
            day2 = mod(dayNumber2,7);
            if ~isempty(find(Holidays == dayNumber2, 1)) || day2==1 || day2==0
                if C(1) ==1
                    calP2(cont2)=1;
                else
                    calP2(cont2)=2;
                end
            else
                if C(1) ==1
                    calP2(cont2)=2;
                else
                    calP2(cont2)=1;
                end
            end
        end
        
        %% forecasting===========================================
        Y_total = [];
        d_x = totalDays-2;% 추정일: 가장 마지막 날짜
        [Y,YMSE] = forecast(arima_para2, H,'Y0',q_day_total(d_x-55:d_x)); % d1 = mm + H:f2
        Y_total = [Y_total; Y];
        
        for j5 = 0 : gamma-1
            hour2 = d_x*gamma+j5+1;  
            realQL = FILE{fix}.TN(hour2+1:hour2+gamma); % Observed data

            periodSection2 = mod(hour2-1,gamma)+1;
            lagV3 = FILE{fix}.TN(hour2-gamma+1:hour2);
            lagV4 = lagV3/sum(lagV3);

            for k_number = 1 : K
                q_training = q_hourly_pattern_training(:,C(1:d1) == k_number);
                PP1(k_number,:) = mean(q_training(:,end-ma:end)');
            end

            % nearest Pattern 2
            global ov2;
            global ds2;
            K2 = size(P,1);
            tMag2 = sum(lagV4);
            daySample2 = lagV4/tMag2;
            ds2 = daySample2;
            observedVals2 = daySample2(end-periodSection2+1:end);
            nbDist2 = (P(:,1:periodSection2)'-repmat(observedVals2,1,K2)).^2;
            ov2 = observedVals2;
            
            if periodSection2 == 1
                [a5,b5] = min(nbDist2);
            else
                [a5,b5] = min(sum(nbDist2));
            end
            
            dist2 = a5;
            k2 = b5;
            
            % nearest Pattern 3
            global ov3;
            global ds3;
            K3 = size(PP1,1);
            tMag3 = sum(lagV4);
            
            daySample3 = lagV4/tMag3;
            ds3 = daySample3;
            observedVals3 = daySample3(end-periodSection2+1:end);
            nbDist3 = (PP1(:,1:periodSection2)'-repmat(observedVals3,1,K3)).^2;
            ov3 = observedVals3;
            
            if periodSection2 == 1
                [a6,b6] = min(nbDist3);
            else
                [a6,b6] = min(sum(nbDist3));
            end

            predictedQL1 = [PP1(modePrediction3(1),:)*Y(1) PP1(modePrediction3(2),:)*Y(2)];
            FILE{fix}.y = realQL*range+min(FILE{fix}.F);
            FILE{fix}.yQMMP = predictedQL1(mod(hour2,gamma)+1:mod(hour2,gamma)+gamma)'*range+min(FILE{fix}.F);

            
        end
    end
    
end

rmse = sqrt(mean((FILE{fix}.y-FILE{fix}.yQMMP).^2))


figure
plot(FILE{1,1}.y)
hold on
plot(FILE{1,1}.yQMMP,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")