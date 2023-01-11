from re import L
import pandas as pd
import numpy as np
import torch.nn as nn
import torch
import os
from sklearn import preprocessing
import copy

class stock_csv_read:
    def __init__(self, data, x_frames, y_frames):
        self.data = data
        self.x_frames = x_frames
        self.y_frames = y_frames
        self.stock_data = self.data_loader()
        
    def data_loader(self):
        stock_data = pd.read_csv("C:/Users/lab/Desktop/Multimodal_Transformer_map\data/kdd17/price_long_50_/"  + self.data ,header=0)
        modality2_data = pd.read_csv(r"C:\Users\lab\Desktop\Multimodal_Transformer_map\data\kdd17\modality2.csv",header=0)
        modality3_data = pd.read_csv(r"C:\Users\lab\Desktop\Multimodal_Transformer_map\data\kdd17\modality3.csv",header=0)

        # modality3_data['Date'] = pd.to_datetime(modality3_data["Date"])
        # modality3_data['Date'] = modality3_data["Date"].dt.strftime('%m/%d/%Y')

        stock_data['Date'] = pd.to_datetime(stock_data["Date"])
        stock_data['Date'] = stock_data["Date"].dt.strftime('%Y/%m/%d')
        modality3_data['Date'] = pd.to_datetime(modality3_data['Date'])
        modality3_data['Date'] = modality3_data['Date'].dt.strftime('%Y/%m/%d') 

        ## inner join은 on 에서 교집합만 출력
        stdata_modality3 = pd.merge(left = stock_data , right = modality3_data, how = "inner", on = "Date") 

        open = stdata_modality3.loc[:,"Open"]
        high = stdata_modality3.loc[:,"High"]
        low = stdata_modality3.loc[:,"Low"]
        close = stdata_modality3.loc[:,"Adj Close"]
        volume = stdata_modality3.loc[:,"Volume"]

        nasdaq100 = stdata_modality3.loc[:,"nasdaq100"]
        us_2y_bond = stdata_modality3.loc[:,"us_2y_bond"]
        us_10y_bond = stdata_modality3.loc[:,"us_10y_bond"]
        us_30y_bond = stdata_modality3.loc[:,"us_30y_bond"]
        us_dollars = stdata_modality3.loc[:,"us_dollars"]
        WTI_oil = stdata_modality3.loc[:,"WTI_oil"]

        m1 = modality2_data.loc[:,"m1"]
        m2 = modality2_data.loc[:,"m2"]
        m3 = modality2_data.loc[:,"m3"]
        m4 = modality2_data.loc[:,"m4"]
        m5 = modality2_data.loc[:,"m5"]
        m6 = modality2_data.loc[:,"m6"]
        m7 = modality2_data.loc[:,"m7"]
        m8 = modality2_data.loc[:,"m8"]
        m9 = modality2_data.loc[:,"m9"]
        m10 = modality2_data.loc[:,"m10"]
        m11 = modality2_data.loc[:,"m11"]
        m12 = modality2_data.loc[:,"m12"]
        MON = modality2_data.loc[:,"MON"]
        TUE = modality2_data.loc[:,"TUE"]
        WED = modality2_data.loc[:,"WED"]
        THU = modality2_data.loc[:,"THU"]
        FRI = modality2_data.loc[:,"FRI"]

        modality2 = pd.concat([m1,m2,m3,m4,m5,m6,m7,m8,m9,m10,m11,m12,MON,TUE,WED,THU,FRI],axis=1,ignore_index=True)

        modality3 = pd.concat([nasdaq100,us_2y_bond,us_10y_bond,us_30y_bond,us_dollars,WTI_oil],axis=1,ignore_index=True)


        upndown = copy.copy(stdata_modality3.loc[:, "Adj Close"] - stdata_modality3.loc[:, "Adj Close"].shift(periods=-1, axis=0))
        change = copy.copy(((stdata_modality3.loc[:, "Adj Close"] - stdata_modality3.loc[:, "Adj Close"].shift(periods=-1, axis=0))/stdata_modality3.loc[:, "Adj Close"].shift(periods=-1, axis=0))*100)
        tgt = np.where(stdata_modality3.loc[:, "Adj Close"] >= stdata_modality3.loc[:, "Adj Close"].shift(periods=-1, axis=0), 1.0, 0.0)
        tgt = pd.DataFrame(tgt)
        # technical_indicator

        # 1) 10일 이동평균
        ten_day_ma = copy.copy(stdata_modality3.loc[:, "Adj Close"])
        for i in range(9):
            ten_day_ma += stdata_modality3.loc[:, "Adj Close"].shift(periods = -i-1, axis = 0)
        
        ten_day_ma = ten_day_ma/10 ## 마지막 nan 9개가 생김
            
        # 2) 10일 가중 이동평균
        w_ten_day_ma = copy.copy(stdata_modality3.loc[:, "Adj Close"])
        w_ten_day_ma_10 = copy.copy(w_ten_day_ma*10)
        for i in range(9):
            w_ten_day_ma_10 += (9-i)*stdata_modality3.loc[:, "Adj Close"].shift(periods = -i-1, axis = 0)
        
        wma = w_ten_day_ma_10/((10*9)/2)

        del w_ten_day_ma
        del w_ten_day_ma_10

        # 3) momentum
        momentum = stdata_modality3.loc[:, "Adj Close"] - stdata_modality3.loc[:, "Adj Close"].shift(periods = -10, axis = 0)


        # 4) stochastic_K%
        init_low = copy.copy(stdata_modality3.loc[:, "Low"])
        init_high = copy.copy(stdata_modality3.loc[:, "High"])
        
        for i in range(9):
            second_low = copy.copy(stdata_modality3.loc[:, "Low"].shift(periods = -i-1, axis = 0))
            second_high = copy.copy(stdata_modality3.loc[:, "High"].shift(periods = -i-1, axis = 0))

            if i == 0:
                lows = pd.concat([init_low,second_low],axis = 1,ignore_index=True)
                highs = pd.concat([init_high,second_high],axis = 1,ignore_index=True)
            else:
                lows = pd.concat([lows,second_low],axis = 1,ignore_index=True)
                highs = pd.concat([highs,second_high],axis = 1,ignore_index=True)
        
        row_low = lows.min(axis=1)
        row_high = highs.max(axis=1)

        stochastic_K = ((stdata_modality3.loc[:,"Close"]-row_low)/(row_high-row_low))*100

        del row_low
        del row_high
        del lows
        del highs
        del second_low
        del second_high
        del init_low
        del init_high

        # 4) stochastic_D%
        stochastic_D = copy.copy(stochastic_K)
        for i in range(9):
            stochastic_D += stochastic_K.shift(periods = -i-1, axis = 0)
        stochastic_D = stochastic_D/10    

        # RSI
        difference = stdata_modality3.loc[:, "Adj Close"] - stdata_modality3.loc[:, "Adj Close"].shift(periods = -1, axis = 0)

        u = abs(difference.where(difference>0,0))
        d = abs(difference.where(difference<0,0))
        init_u = copy.copy(u)
        init_d = copy.copy(d)

        for i in range(9):
            init_u += u.shift(periods = -i-1, axis = 0)
            init_d += d.shift(periods = -i-1, axis = 0)
        
        AU = init_u/10
        AD = init_d/10

        RSI = 100-100/(1+AU/AD)

        del AU
        del AD
        del init_u
        del init_d
        del u
        del d
        del difference

        df1 = stdata_modality3.loc[:, "Adj Close"]

        df1 = df1.iloc[::-1]
        

        ema_12 = df1.ewm(span=12,min_periods=11,adjust = True).mean()
        ema_26 = df1.ewm(span=26,min_periods=25,adjust = True).mean()

        # MACD
        MACD =  ema_12 - ema_26

        MACD = MACD[::-1]

        del ema_12
        del ema_26
        del df1

        # Larry_williams_R 
        LWR = ((stdata_modality3.loc[:,"High"]-stdata_modality3.loc[:,"Close"])/(stdata_modality3.loc[:,"High"]-stdata_modality3.loc[:,"Low"]))*100

        # A_D_Oscillator 
        A_D = (stdata_modality3.loc[:,"High"]-stdata_modality3.loc[:,"Close"].shift(periods=-1, axis=0))/(stdata_modality3.loc[:,"High"]-stdata_modality3.loc[:,"Low"])
        
        # CCI
        MT = stdata_modality3.loc[:,"High"]+stdata_modality3.loc[:,"Low"]+stdata_modality3.loc[:,"Close"]/3
        SMT = copy.copy(MT)
        for i in range(9):
            SMT += MT.shift(periods=-1-i, axis=0)
        
        SMT = SMT/10

        DT = abs(MT.shift(periods=-9, axis=0) - SMT)
        for i in range(9):
            DT += abs(MT.shift(periods=-i, axis=0) - SMT)
        
        DT = DT/10

        CCI = (MT -SMT)/(0.015*DT)
        
        del MT
        del DT
        del SMT

        d_len = len(RSI)

        modality1  = pd.concat([open, high, low, close, volume, upndown, change, ten_day_ma, wma, momentum, stochastic_K, stochastic_D, RSI, MACD, LWR, A_D, CCI], axis=1,ignore_index=True)
        df = pd.concat([modality1,modality2,modality3], axis=1,ignore_index=True)

        data = pd.concat([df,tgt], axis=1,ignore_index=True)
        data = data.iloc[:(d_len-24)][:(d_len-24)][::-1].copy()
        
        return pd.DataFrame(data)
    
    def spliter(self, data):
        self.dd = data
        data_list = []
        for i in range(len(self.dd)-self.x_frames-self.y_frames+1):
            xy = []
            X = self.dd.iloc[i : i +self.x_frames, 0:40].values
            scaler  = preprocessing.MinMaxScaler().fit(X)
            X = scaler.transform(X)
            y = self.dd.iloc[i +self.x_frames : i +self.x_frames +self.y_frames, 40:].values
            xy.append(X)
            xy.append(y)
            data_list.append(xy)
        return data_list
    
    def cv_split(self):
        stock_data = self.stock_data
        
        data_len = len(stock_data)
        mok = data_len//19
        
        adder = 0
        data_list = []
        for i in range(10):
            sp_data = stock_data.iloc[0+adder:10*mok+adder,:]
            
            tvt = []
            train_sp_data = sp_data[0:8*mok]
            validation_sp_data = sp_data[8*mok:9*mok]
            test_sp_data = sp_data[9*mok:10*mok]
            
            train_sp_data_ = self.spliter(train_sp_data)
            validation_sp_data_ = self.spliter(validation_sp_data)
            test_sp_data_ = self.spliter(test_sp_data)

            tvt.append(train_sp_data_)
            tvt.append(validation_sp_data_)
            tvt.append(test_sp_data_)
            adder += mok
            data_list.append(tvt)
            
        return data_list

