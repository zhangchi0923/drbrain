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
import joblib
import random

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
        detector_l2 = eyeMovement.EyeMovement(x, y, time, AOIs, BEZIER_POINTS)
        att = detector_l2.measureFollowRate()
        feats = [att]
        # other
        for level in levels:
            x, y, time = self.get_lvl_state(df, level, 2)
            detector = eyeMovement.EyeMovement(x, y, time, AOIs[level], BEZIER_POINTS)
            fix_data = detector.eye_movements_detector(x, y, time)
            _, _, merged = detector.merge_fixation(fix_data)
            feats.append(detector.AOI_fixation_ratio(merged))
            ffd, ttff, nfbff = detector.AOI_first_fixation(merged)
            feats += [ffd, ttff, nfbff]
        result = pd.DataFrame([self._gender, self._education, self._age] + feats).T
        result.columns = self.feat_cols
        return result
    
    def predict(self, df):
        moca_model = joblib.load(MOCA_MODEL_PATH)
        mmse_model = joblib.load(MMSE_MODEL_PATH)
        moca = float(moca_model.predict(df)[0])
        mmse = float(mmse_model.predict(df)[0])
        return moca, mmse
    
    def cog_score(self, df) -> list:
        result = df.loc[:, self.cog_cat].values[0].tolist()
        return [x*100 if x > 0.05 else x*100 + 0.5*random.randint(0, 10) for x in result]
