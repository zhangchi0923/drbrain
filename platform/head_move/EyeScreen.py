"""
 author: zhangchi
 email: zcsjtu0923@gmail.com
 create date: 2023-03-13 11:27:27
"""
import pandas as pd
import numpy as np
import eyeMovement
from settings import *
import itertools
import random
import logging, datetime, os
from sklearn.linear_model import LinearRegression

class EyeScreen(object):
    base_cat = ['abs', 'calc4', 'calc5', 'calc6', 'exec', 'mem8', 'mem9', 'mem10', 'recall']
    eye_cat = ['_aoi_ratio', '_ffd', '_ttff', '_nfbff']

    cog_cat = ['att'] + [x[0] + x[1] for x in itertools.product(base_cat, ['_aoi_ratio'])]
    num_cols = ['att'] + [x[0] + x[1] for x in itertools.product(base_cat, eye_cat)]
    cat_cols = ['gender', 'education', 'age']
    feat_cols = cat_cols + num_cols

    def __init__(self, s, gender, education, age):
        self._s = s
        self._gender = gender
        self._education = education
        self._age = age
    
    def text2DF(self) -> pd.DataFrame:
        lines=self._s.split('\n')
        head=lines[0].split(',')
        arr = [x.split(',') for x in lines[1:] if not not x]
        df = pd.DataFrame(arr, columns=head)
        df['timestamp'] = df['timestamp'].astype('uint64', errors='ignore')
        df['timestamp'] = df['timestamp'] - df.loc[0, 'timestamp']
        df['level'] = df['level'].astype(int, errors='ignore')
        df['state'] = df['state'].astype(int, errors='ignore')
        df['pos_x'] = df['pos_x'].astype(float, errors='ignore')
        df['pos_y'] = df['pos_y'].astype(float, errors='ignore')
        df['left'] = df['left'].astype(float, errors='ignore')
        df['right'] = df['right'].astype(float, errors='ignore')
        return df
    
    def get_lvl_state(self, df: pd.DataFrame, level: int, state: int):
        idx1 = df['level'] == level
        idx2 = df['state'] == state
        idx = idx1 & idx2
        tmpDf = df[idx].copy()
        x, y, time = np.array(tmpDf['pos_x']), np.array(tmpDf['pos_y']), np.array(tmpDf['timestamp'])
        return x, y, time
    
    def preprocess_feat(self, df) -> pd.DataFrame:
        levels = list(range(3, 12, 1))
        x, y, time = self.get_lvl_state(df, 2, 2)

        bez_x, bez_y = self.get_live_position(time/1000, time[0]/1000)
        model = self.corrModel(x, y, bez_x, bez_y)
        corr_x = np.corrcoef(x, bez_x, rowvar=False)[0, 1]; corr_y = np.corrcoef(y, bez_y, rowvar=False)[0, 1]
        print(corr_x, corr_y)
        if corr_x < 0.9 or corr_y < 0.8:
            model.coef_ = np.array([[1,0],[0,1]])
            model.intercept_ = np.array([0,0])
        print(model.coef_, model.intercept_)
        X = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        # print(X.shape)
        # print(model.predict(X))
        x = model.predict(X)[:, 0]
        y = model.predict(X)[:, 1]
        tmpdf = pd.DataFrame({'timestamp':time, 'pos_x': x, 'pos_y': y})

        detector_l2 = eyeMovement.EyeMovement(x, y, time, AOIs, BEZIER_POINTS)
        att = detector_l2.fanbo_follow_rate(tmpdf)
        feats = [att]
        # other
        for level in levels:
            x, y, time = self.get_lvl_state(df, level, 2)
            X = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
            time = time.reshape(-1, 1)
            # print(model_x.coef_, model_y.coef_)
            # print(model_x.intercept_, model_y.intercept_)
            x = model.predict(X)[:, 0]
            y = model.predict(X)[:, 1]
            x = x.flatten(); y = y.flatten(); time = time.flatten()
            detector = eyeMovement.EyeMovement(x, y, time, AOIs[level], BEZIER_POINTS)
            fix_data = detector.eye_movements_detector(x, y, time)
            _, _, merged = detector.merge_fixation(fix_data)
            # feats.append(detector.AOI_fixation_ratio(merged))

            # 烦勃要求的
            tmpdf = pd.DataFrame({'timestamp':time, 'pos_x': x, 'pos_y': y})
            feats.append(detector.measureGazeRate(tmpdf, AOIs[level]))


            ffd, ttff, nfbff = detector.AOI_first_fixation(merged)
            feats += [ffd, ttff, nfbff]
        result = pd.DataFrame([self._gender, self._education, self._age] + feats).T
        result.columns = self.feat_cols
        return result
    
    def predict(self, df):
        cog_score = self.cog_score(df)
        ovr_score = np.mean(cog_score)
        if ovr_score < 30:
            moca = 20 - (30 - ovr_score)*3/10
        elif ovr_score < 50:
            moca = 24 - (50 - ovr_score)*3/10
        elif ovr_score < 70:
            moca = 26 + (ovr_score - 50)/10
        else:
            moca = 30 - (100 - ovr_score)/10
        mmse = min(30, moca + 2 + 0.1*random.randint(0, 10))
        return moca, mmse
    
    def cog_score(self, df) -> list:
        result = df.loc[:, self.cog_cat].values[0].tolist()
        # 伟大的烦勃要求的
        return [x*100 for x in result]
        return [x*100 if x > 0.05 else x*100 + 0.5*random.randint(0, 10) for x in result]

    def bezier_order3(self, t, points):
        p0 = points[0,:]
        p1 = points[1,:]
        p2 = points[2,:]
        p3 = points[3,:]
        y = (1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t**2*p2+t**3*p3
        return y

    def get_live_position(self, trueTime, start_time):
        bez_x = []
        bez_y = []
        allPoints = np.array(BEZIER_POINTS)
        for time in trueTime:
            rel_time = (time-start_time)/2.5 
            i = int(rel_time)                                                          # for detecting current section in 4 bezier curves
            t = rel_time-i                                                             # time parameter of bezier curves
            k = i%4                                                                    # detect current section in 4 bezier curves
            points = allPoints[4*k:4*k+4,:]
            x, y = self.bezier_order3(t,points)
            bez_x.append(x); bez_y.append(y)
        return bez_x, bez_y

    def corrModel(self, x, y, bez_x, bez_y):
        subDf = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), np.array(bez_x).reshape(-1, 1), np.array(bez_y).reshape(-1,1)), axis=1)
        # print(subDf.shape)
        X = subDf[:, 0: 2]; Y = subDf[:, 2: 4]
        model = LinearRegression()
        model.fit(X, Y)
        # print(model.coef_)
        # print(model.intercept_)
        # model_x.fit(x.reshape(-1, 1), np.array(bez_x).reshape(-1, 1))
        # model_y.fit(y.reshape(-1, 1), np.array(bez_y).reshape(-1, 1))
        return model