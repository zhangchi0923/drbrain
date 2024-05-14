"""
 author: zhangchi
 email: zcsjtu0923@gmail.com
 create date: 2023-03-13 11:27:27
"""
import os
from fastapi import BackgroundTasks, Request
import pandas as pd
import numpy as np
from pydantic import BaseModel
import requests
from sklearn import linear_model
from utils.auth import auth_validate
import utils.eyeMovement as eyeMovement
from config.settings import settings
import itertools
import random
from sklearn.linear_model import LinearRegression
from utils.logger import get_logger

from utils.response_template import GeneralResponseModel

class EyeScreenReqeustModel(BaseModel):
    sex: str
    age: int
    education: str
    url: str
    saveResourcesPath: str
    questionVersion: str


base_cat = ['abs', 'calc4', 'calc5', 'calc6', 'exec', 'mem8', 'mem9', 'mem10', 'recall']
eye_cat = ['_aoi_ratio', '_ffd', '_ttff', '_nfbff']
cog_cat = ['att'] + [x[0] + x[1] for x in itertools.product(base_cat, ['_aoi_ratio'])]
num_cols = ['att'] + [x[0] + x[1] for x in itertools.product(base_cat, eye_cat)]
cat_cols = ['gender', 'education', 'age']
feat_cols = cat_cols + num_cols
allPoints = np.array(settings.bezier_points)


def findMinMax(a,b):
    if a>b:
        return b,a
    else:
        return a,b
    
def extended_ROI(ROIs,percent = 0.1):
    newROIs = {}
    for key in ROIs:
        ROI= ROIs[key]
        X0,X1 = findMinMax(ROI[0][0],ROI[1][0])
        Y0,Y1 = findMinMax(ROI[0][1],ROI[1][1])
        width = X1-X0
        hight = Y1-Y0
        margin_x = percent*width
        margin_y = percent*hight
        X0 = X0 - margin_x
        X1 = X1 + margin_x
        Y0 = Y0 - margin_y
        Y1 = Y1 + margin_y
        newROIs[key] = [(X0,Y0),(X1,Y1)]
    return newROIs
ROIs = extended_ROI(settings.aois)

def bezier_order3(t, points):
    p0 = points[0,:]
    p1 = points[1,:]
    p2 = points[2,:]
    p3 = points[3,:]
    y = (1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t**2*p2+t**3*p3
    return y

def bezier_pts(start_time, end_time, interval):
    time = np.linspace(start_time/1000, end_time/1000, interval)
    bez_x = []; bez_y = []
    s_time = time[0]
    for t in time:
        x, y = get_live_position(t, s_time)
        bez_x.append(x); bez_y.append(y)
    return bez_x, bez_y


def get_live_position(trueTime, start_time):                                                      
    rel_time = (trueTime - start_time)/2.5                                       # time parameter of bezier curves
    i = int(rel_time)                                                          # for detecting current section in 4 bezier curves
    t = rel_time-i                                                             # time parameter of bezier curves
    k = i%4                                                                    # detect current section in 4 bezier curves
    points = allPoints[4*k:4*k+4,:]                                            
    y = bezier_order3(t, points)
    return y
    
def measureFollowRate(subDf):
    nObserv = len(subDf)
    cnt = 0
    eps = 0.55                                                              # distance to rocket
    rec = []
    # print(subDf)
    start_time = subDf['timestamp'].iat[0]/1000
    for i in range(nObserv):
        x = subDf['pos_x'].iat[i]
        y = subDf['pos_y'].iat[i]
        t = subDf['timestamp'].iat[i]/1000
        pos_rocket = get_live_position(t,start_time)
        X0 = pos_rocket[0]
        Y0 = pos_rocket[1]
        rec.append([t,x,y,X0,Y0])
        if (x-X0)**2+(y-Y0)**2 < eps**2:
            cnt += 1
    ratio = cnt/nObserv
    rec = np.array(rec)
    return ratio, rec
            
def measureGazeRate(subDf,ROI):
    """
    measure the ratio of the gaze time on ROI 

    Parameters
    ----------
    subDf : pandas dataFrame with column names of 'timestamp','pos_x','pos_y'             
    ROI: list of 4 float numbers, Rectangular diagonal vertex coordinates. 
         just as [(x0,y0),(x1,y1)],express a rectangle with 4 corners(x0,y0),(x1,y0),(x1,y1),(x0,y1)
              
    Returns
    -------
    ratio :     float, time ratio of gaze time

    Example
    -------
    >>> measureGazeRate(df,[(5,6),(0,1)])
    0.5
    """                                                               # margin of ROI
    X0,X1 = findMinMax(ROI[0][0],ROI[1][0])
    Y0,Y1 = findMinMax(ROI[0][1],ROI[1][1])
    nObserv = len(subDf)
    total_Time = subDf['timestamp'].iat[nObserv-1] - subDf['timestamp'].iat[0]
    gaze_time = 0
    for i in range(1,nObserv):
        x = subDf['pos_x'].iat[i]
        y = subDf['pos_y'].iat[i]
        #t = subDf['timestamp'].iat[i]
        if (x-X0)*(x-X1)<0 and (y-Y0)*(y-Y1)<0:
            gaze_time += subDf['timestamp'].iat[i]-subDf['timestamp'].iat[i-1]
    ratio = gaze_time/total_Time
    return ratio
    
def mkdir_new(path):
    if not os.path.exists(path):
        os.makedirs(path) 
        
def sectionSplit2(myList1,myList2):
    """
    find a list of index when state changed according two variables

    Parameters
    ----------
    myList1 : list, a sequence of states variable A             
    myList2 : list, a sequence of states variable B
              
    Returns
    -------
    rec :     list, list of index

    Example
    ------- 
    >>> sectionSplit2([0,0,0,1,1],['a','b','b','b','b'])
    [0,1,3,5]
    """
    rec = [0]
    n = len(myList1)
    for i in range(1,n):
        if (myList1[i] != myList1[i-1]) | (myList2[i] != myList2[i-1]) :
            onset = i
            rec.append(onset)
    rec.append(n)
    return rec

def correctDataEstimate(subDf):
    x =subDf[:,1:3]
    y =subDf[:,3:5]
    model = linear_model.LinearRegression()
    model.fit(x, y)
    return model

def deSpike(arr,threshold=1):
    x = arr[:,1:3]
    dx = np.diff(x,axis=0)
    idx0 = np.abs(dx)<threshold
    idx1 = idx0[:,1]&idx0[:,0]
    idx2 = np.array([True]+list(idx1))
    newArr = arr[idx2,:]
    return newArr

def compute_cog_score(df):

    try:
        # # # data preprocessing
        startTime = df['timestamp'].min()                                          # start time, time unit: ms
        df.loc[:,'timestamp'] = df['timestamp']-startTime                          # set start time with 0 ms
        idx = df.isna().sum(axis=1)==0                                             # detect index of rows with nan data
        df = df[idx].copy()                                                        # discard rows with nan data
        df.sort_values(by = 'timestamp',inplace=True)                              # sort data by time
        df.reset_index(drop=True,inplace=True)

        # # # index of onset for data split
        onset = sectionSplit2(df.level,df.state)
        scoreList = []
        nSection = len(onset)-1

        idx1 = df['state']==2
        idx2 = df['level']==2
        idx = idx1&idx2
        tmpDf = df[idx].copy()
        time = tmpDf['timestamp'].tolist()
        score,trail_level2 = measureFollowRate(tmpDf)
        rec_trail = deSpike(trail_level2,1.5)
        corrMat = np.corrcoef(rec_trail[:,1:],rowvar=False)
        model = correctDataEstimate(rec_trail)
        if corrMat[0,2] < 0.9 or corrMat[1,3] < 0.8:                #相关系数
            model.coef_ = np.array([[1,0],[0,1]])
            model.intercept_ = np.array([0,0])

        for i in range(nSection):
            if i == 0:
                startLevel = df['level'].iat[0]
            subDf = df.loc[onset[i]:onset[i+1],:].copy()

            # corrected for eye tracking data
            xx = np.array(subDf[['pos_x','pos_y']])
            yy = model.predict(xx)
            subDf.loc[:,'pos_x'] = yy[:,0]
            subDf.loc[:,'pos_y'] = yy[:,1]
            maker = 'corrected'
            x = subDf['pos_x']
            y = subDf['pos_y']
            state = df.loc[onset[i],'state']
            level = df.loc[onset[i],'level']
            if state ==2 and level == 2:
                score,trail_level2 = measureFollowRate(subDf)
                # scoreList.append([level,score])
                scoreList.append(score)

            if state ==2 and level > 2:
                ROI= ROIs[level]
                score = measureGazeRate(subDf, ROI)
                # scoreList.append([level,score])
                scoreList.append(score)

    except Exception as e:
        import traceback
        traceback.print_exc()
        # print(e)
        # logger.error("Faild to open sklearn.txt from logger.error",exc_info = True)
        raise
    return scoreList

def text2DF(s) -> pd.DataFrame:
    lines = s.split('\n')
    head = lines[0].split(',')
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

    df.dropna(how='any', axis=0, inplace=True)
    return df

def get_lvl_state(df: pd.DataFrame, level: int, state: int):
    idx1 = df['level'] == level
    idx2 = df['state'] == state
    idx = idx1 & idx2
    tmpDf = df[idx].copy()
    x, y, time = np.array(tmpDf['pos_x']), np.array(
        tmpDf['pos_y']), np.array(tmpDf['timestamp'])
    return x, y, time

def preprocess_feat(df, gender, education, age) -> pd.DataFrame:
    levels = list(range(3, 12, 1))
    x, y, time = get_lvl_state(df, 2, 2)

    bez_x, bez_y = get_live_bezier(time/1000, time[0]/1000)
    model = corrModel(x, y, bez_x, bez_y)
    corr_x = np.corrcoef(x, bez_x, rowvar=False)[0, 1]
    corr_y = np.corrcoef(y, bez_y, rowvar=False)[0, 1]
    if corr_x < 0.9 or corr_y < 0.8:
        model.coef_ = np.array([[1, 0], [0, 1]])
        model.intercept_ = np.array([0, 0])
    X = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
    x = model.predict(X)[:, 0]
    y = model.predict(X)[:, 1]
    tmpdf = pd.DataFrame({'timestamp': time, 'pos_x': x, 'pos_y': y})

    detector_l2 = eyeMovement.EyeMovement(x, y, time, settings.aois, settings.bezier_points)
    att = detector_l2.fanbo_follow_rate(tmpdf)
    feats = [att]
    # other
    for level in levels:
        x, y, time = get_lvl_state(df, level, 2)
        X = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1)), axis=1)
        time = time.reshape(-1, 1)
        # print(model_x.coef_, model_y.coef_)
        # print(model_x.intercept_, model_y.intercept_)
        x = model.predict(X)[:, 0]
        y = model.predict(X)[:, 1]
        x = x.flatten()
        y = y.flatten()
        time = time.flatten()
        detector = eyeMovement.EyeMovement(
            x, y, time, settings.aois[level], settings.bezier_points)
        fix_data = detector.eye_movements_detector(x, y, time)
        _, _, merged = detector.merge_fixation(fix_data)
        # feats.append(detector.AOI_fixation_ratio(merged))

        # 烦勃要求的
        tmpdf = pd.DataFrame({'timestamp': time, 'pos_x': x, 'pos_y': y})
        feats.append(detector.measureGazeRate(tmpdf, settings.aois[level]))

        ffd, ttff, nfbff = detector.AOI_first_fixation(merged)
        feats += [ffd, ttff, nfbff]
    result = pd.DataFrame(
        [gender, education, age] + feats).T
    result.columns = feat_cols
    return result

def predict(df):
    scores = cog_score(df)
    ovr_score = np.mean(scores)
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

def cog_score(df) -> list:
    result = df.loc[:, cog_cat].values[0].tolist()
    # 烦勃要求的
    return [x*100 for x in result]
    return [x*100 if x > 0.05 else x*100 + 0.5*random.randint(0, 10) for x in result]

def bezier_order3(t, points):
    p0 = points[0, :]
    p1 = points[1, :]
    p2 = points[2, :]
    p3 = points[3, :]
    y = (1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t**2*p2+t**3*p3
    return y

def get_live_bezier(trueTime, start_time):
    bez_x = []
    bez_y = []
    allPoints = np.array(settings.bezier_points)
    for time in trueTime:
        rel_time = (time-start_time)/2.5
        # for detecting current section in 4 bezier curves
        i = int(rel_time)
        # time parameter of bezier curves
        t = rel_time-i
        # detect current section in 4 bezier curves
        k = i % 4
        points = allPoints[4*k:4*k+4, :]
        x, y = bezier_order3(t, points)
        bez_x.append(x)
        bez_y.append(y)
    return bez_x, bez_y

def corrModel(x, y, bez_x, bez_y):
    subDf = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), np.array(
        bez_x).reshape(-1, 1), np.array(bez_y).reshape(-1, 1)), axis=1)
    # print(subDf.shape)
    X = subDf[:, 0: 2]
    Y = subDf[:, 2: 4]
    model = LinearRegression()
    model.fit(X, Y)
    # print(model.coef_)
    # print(model.intercept_)
    # model_x.fit(x.reshape(-1, 1), np.array(bez_x).reshape(-1, 1))
    # model_y.fit(y.reshape(-1, 1), np.array(bez_y).reshape(-1, 1))
    return model

def eye_screen(model: EyeScreenReqeustModel, request: Request, background_tasks: BackgroundTasks):

    auth_resp = auth_validate(model, request)
    if auth_resp:
        return auth_resp
    gender = model.sex
    age = model.age
    education = model.education
    url = model.url
    # src = args['backupResources']
    save_pth = model.saveResourcesPath
    q_ver = model.questionVersion
    src = './assets/design-{}/'.format(q_ver)

    _, sid = os.path.split(url)
    if not os.path.exists('./log/eyescreen_log'):
        os.mkdir('./log/eyescreen_log')
    logger = get_logger(sid, './log/eyescreen_log')
    logger.info('Authorization succeed.')

    # executer = ProcessPoolExecutor(1)
    # executer.submit(draw_eye_screen, url, save_pth, src)
    background_tasks.add_task(draw_eye_screen, url, save_pth, src)
    logger.info('Eye screen plot submitted.')
    # print(os.getcwd(), src, save_pth)
    results = calc_eye_screen(url, gender, education, age, logger)
    logger.info('Eye screen results: {}'.format(results))
    return results

def calc_eye_screen(url, gender, education, age, logger):
    if settings.deploy_mode == 'offline':
        with open(url) as f:
            txt = f.read()
    else:
        with requests.get(url) as r:
            if r.status_code != 200 :
                logger.error("Cannot access url data!")
            txt = r.text
    df = text2DF(txt)
    
    try:
        data = preprocess_feat(df, gender, education, age)
        # print(1)
        moca, mmse = predict(data)
        # print(2)
        scores = compute_cog_score(df)
        # print(scores)
        scores = [x*100 for x in scores]
        logger.info('Cog score: {}\nMoCA: {} MMSE: {}'.format(scores, moca, mmse))
    except Exception as e:
        logger.exception(e)
        return GeneralResponseModel(
            code=500,
            # 'msg':'Error during score predicting.',
            msg=str(e),
            body=None
        )
    resp = GeneralResponseModel(
        code=200,
        msg="AI prediction succeed.",
        body={
            'mmse':round(mmse, 1),
            'moca':round(moca, 1),
            'resultScores':[
                {'level':1, 'score':round(scores[0], 1)},{'level':2, 'score':round(scores[1], 1)},{'level':3, 'score':round(scores[2], 1)},
                {'level':4, 'score':round(scores[3], 1)},{'level':5, 'score':round(scores[4], 1)},{'level':6, 'score':round(scores[5], 1)},
                {'level':7, 'score':round(scores[6], 1)},{'level':8, 'score':round(scores[7], 1)},{'level':9, 'score':round(scores[8], 1)},
                {'level':10, 'score':round(scores[9], 1)}
            ]
        }
    )
    logger.info('Eye screen prediction succeed.')
    return resp

def draw_eye_screen(url, out_pth, design_pth):
    os.system('python ./utils/justscore_bySection_4urldata_final.py {} {} {}'.format(url, out_pth, design_pth))
