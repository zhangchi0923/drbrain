"""
Created on: Wed Jun 15 17:46:34 2022
Email:      pbb_194@163.com
@author:    panBaobao
"""
import sys, time, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib
matplotlib.use('Agg')
from sklearn import linear_model
import requests
myFont = fm.FontProperties(fname='C:\Windows\Fonts\simkai.ttf')
import warnings
warnings.filterwarnings('ignore')


import logging
import datetime

# # # - - - - - - - - coordinate of region of object 
ROIs = {
        3:[(1.644000, -1.742000),(2.844000, -0.692000)],
        4:[(-0.594000, -0.340000),(0.594000, 0.848000)],
        5:[(-0.594000, -1.788000),(0.594000, -0.600000)],
        6:[(-2.682000, -1.788000),(-1.494000, -0.600000)],
        7:[(-2.750000, -1.700000),(-1.200000, -0.600000)],
        8:[(0.678000, -0.458000),(2.570000, 0.736000)],
        9:[(-2.570400, -1.856000),(-0.678400, -0.672000)],
        10:[(-2.570400, -0.448000),(-0.678400, 0.736000)],
        11:[(-2.570400, -1.856000),(-0.678400, -0.672000)],
        }

# # # - - - - - - - - coordinate of parameters of bezier curves
ps=[[0.000, -0.040],
    [0.376, -1.440],
    [2.284, -1.440],
    [2.620, -0.040],
    [2.620, -0.040],
    [2.284, 1.360],
    [0.376, 1.360],
    [0.000, -0.040],
    [0.000, -0.040],
    [-0.376, -1.440],
    [-2.284, -1.440],
    [-2.620, -0.040],
    [-2.620, -0.040],
    [-2.284, 1.360],
    [-0.376, 1.360],
    [0.000, -0.040]]
# ps = a[::-1]
allPoints = np.array(ps)


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
ROIs = extended_ROI(ROIs)

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

def text2Df(myStr):
    lines=myStr.split('\n')
    head=lines[0].split(',')
    arr = [x.split(',') for x in lines[1:] if not not x ]
    df = pd.DataFrame(arr,columns = head)
    df['timestamp'] = df['timestamp'].astype('uint64',errors='ignore')
    df['level'] = df['level'].astype(int,errors='ignore')
    df['state'] = df['state'].astype(int,errors='ignore')
    df['pos_x'] = df['pos_x'].astype(float,errors='ignore')
    df['pos_y'] = df['pos_y'].astype(float,errors='ignore')
    df['left' ] = df['left' ].astype(float,errors='ignore')
    df['right'] = df['right'].astype(float,errors='ignore')
    return df 

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

#%%
def main(url,outputPth,designPth):

    mkdir_new(outputPth)
    logger = None
    logger = logging.getLogger()
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(outputPth,'log') )
    handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # console = logging.StreamHandler()
    # console.setLevel(logging.INFO)

    logger.addHandler(handler)
    # logger.addHandler(console)


    try:
        maker = 'origin'

        stateDict = {0:'读题',1:'预览图文',2:'答题'}                                # state of subject
        response = requests.get(url,verify=False)
        if response.__getstate__()['status_code'] != 200:
            logger.info('Http errr for data loading!')
            pass
        df = text2Df(response.text)

        response.close()

        # # # data preprocessing
        startTime = df['timestamp'].min()                                          # start time, time unit: ms
        df.loc[:,'timestamp'] = df['timestamp']-startTime                          # set start time with 0 ms
        idx = df.isna().sum(axis=1)==0                                             # detect index of rows with nan data
        df = df[idx].copy()                                                        # discard rows with nan data
        #cnt_nan = len(df) - sum(idx)                                                # count nan in eye-tracking data
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
        # print(tmpDf)
        score,trail_level2 = measureFollowRate(tmpDf)
        rec_trail = deSpike(trail_level2,1.5)
    #     print(np.min(rec_trail))
        logger.info(str(np.min(rec_trail)))
        corrMat = np.corrcoef(rec_trail[:,1:],rowvar=False)
    #     print('corrcoef: \n',corrMat[0,2],corrMat[1,3])
        logger.info('corrcoef: \n'+str(corrMat[0,2])+'   '+str(corrMat[1,3]))
        model = correctDataEstimate(rec_trail)
        # if corrMat[0,2] < 0.9 or corrMat[1,3] < 0.8:                #相关系数
        if corrMat[0,2] < 0 or corrMat[1,3] < 0:                #相关系数
            model.coef_ = np.array([[1,0],[0,1]])
            model.intercept_ = np.array([0,0])
        logger.info("regression coefficients:\n"+str(model.coef_))
        logger.info("regression intercept:\n"+str(model.intercept_))

        #%%

        for i in range(nSection):
            if i == 0:
                startLevel = df['level'].iat[0]
            subDf = df.loc[onset[i]:onset[i+1],:].copy()

            #%% corrected for eye tracking data
            xx = np.array(subDf[['pos_x','pos_y']])
            yy = model.predict(xx)
            subDf.loc[:,'pos_x'] = yy[:,0]
            subDf.loc[:,'pos_y'] = yy[:,1]
            maker = 'corrected'
        #%%
            x = subDf['pos_x']
            y = subDf['pos_y']
            state = df.loc[onset[i],'state']
            level = df.loc[onset[i],'level']
            if state ==2 and level == 2:
                score,trail_level2 = measureFollowRate(subDf)
                scoreList.append([level,score])
                # print(level,'\t',score)
                logger.info('question:'+str(level)+' \t score:'+str(score))
                #print(subDf)
                rec = trail_level2
                fig = plt.figure(figsize=(8.5,9))
                plt.subplot(2,1,1)
                plt.plot(rec[:,0],rec[:,1:],'-')
                plt.title('score: '+str(round(score,4)))
                plt.xlabel('Times(s)')
                plt.legend(['pos_x','pos_y','rocket_x','rocket_y'])
                fig = plt.figure(figsize=(8.5, 6))
    #             plt.subplot(2,1,2)
                img=plt.imread(str(designPth)+"/2.png")         # reading backgram photos with png format
                plt.imshow(img,extent=[-3.84,3.84,-2.16,2.16])
                plt.plot(rec[:,1],rec[:,2],'r-')
                bez_x, bez_y = bezier_pts(time[0], time[-1], len(time))
                plt.plot(bez_x, bez_y, 'y-')
                # plt.plot(rec[:,3],rec[:,4],'y-')
                plt.xticks([])
                plt.yticks([])
                plt.xlim(-5,5)
                plt.ylim(-3,3)
                plt.tight_layout()
                outfileName = os.path.join(outputPth,'trail_'+str(level)+'.jpg')
                fig.savefig(outfileName,dpi = 300)
                plt.close()
                # en version
                # fig_en = plt.figure(figsize=(8.5, 6))
                # img_en=plt.imread(str(designPth)+"en/en_2.jpg")         # reading backgram photos with png format
                # plt.imshow(img_en,extent=[-3.84,3.84,-2.16,2.16])
                # plt.plot(rec[:,1],rec[:,2],'r-')
                # # plt.plot(rec[:,3],rec[:,4],'y-')
                # plt.plot(bez_x, bez_y, 'y-')
                # plt.xticks([])
                # plt.yticks([])
                # plt.xlim(-5,5)
                # plt.ylim(-3,3)
                # plt.tight_layout()
                # outfileName = os.path.join(outputPth,'trail_en_'+str(level)+'.jpg')
                # fig_en.savefig(outfileName,dpi = 300)
                # plt.close()

            if state == 2 and level == 1:
                fig = plt.figure(figsize=(8.5, 6))                                       # plot interface
                plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97)
                img=plt.imread(str(designPth)+"1.png")         # reading backgram photos with png format
                plt.imshow(img,extent=[-3.84,3.84,-2.16,2.16])                           # location of canvas
                plt.plot(x,y,'r-',linewidth=0.5)
                plt.xticks([])
                plt.yticks([])
                plt.xlim(-5,5)
                plt.ylim(-3,3)
                outfileName = os.path.join(outputPth,'trail_'+str(level)+'.jpg')
                fig.savefig(outfileName,dpi = 300)
                plt.close()
                # en version
                # fig_en = plt.figure(figsize=(8.5, 6))                                       # plot interface
                # plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97)
                # img_en=plt.imread(str(designPth)+"en/en_1.png")         # reading backgram photos with png format
                # plt.imshow(img_en,extent=[-3.84,3.84,-2.16,2.16])                           # location of canvas
                # plt.plot(x,y,'r-',linewidth=0.5)
                # plt.xticks([])
                # plt.yticks([])
                # plt.xlim(-5,5)
                # plt.ylim(-3,3)
                # outfileName = os.path.join(outputPth,'trail_en_'+str(level)+'.jpg')
                # fig_en.savefig(outfileName,dpi = 300)
                # plt.close()

            if state ==2 and level > 2:
                ROI= ROIs[level]
                score = measureGazeRate(subDf,ROI)
                scoreList.append([level,score])
                # print(level,'\t',score)
                logger.info('question:'+str(level)+' \t score:'+str(score))

        # # # - - - - - - - - visualization - - - - - - - - - -
                fig = plt.figure(figsize=(8.5, 6))                                       # plot interface
                plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97)
                img=plt.imread(str(designPth)+str(level)+".png")         # reading backgram photos with png format
                plt.imshow(img,extent=[-3.84,3.84,-2.16,2.16])                           # location of canvas
                plt.plot(x,y,'r-',linewidth=0.5)
                x0 = ROI[0][0]
                y0 = ROI[0][1]
                x1 = ROI[1][0]
                y1 = ROI[1][1]
                box_x =  [x0,x1,x1,x0,x0]
                box_y =  [y0,y0,y1,y1,y0]
                plt.plot(box_x,box_y,c='yellow')
                thing = stateDict[state]
                if state==2:
                    mycolor = 'r'
                else:
                    mycolor = 'b'
    #             plt.title(f'第{level}题({thing})')
                plt.xticks([])
                plt.yticks([])
                plt.xlim(-5,5)
                plt.ylim(-3,3)
                outfileName = os.path.join(outputPth,'trail_'+str(level)+'.jpg')
                fig.savefig(outfileName,dpi = 300)
                plt.close()
                # en version
    #             fig_en = plt.figure(figsize=(8.5, 6))                                       # plot interface
    #             plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.97)
    #             img_en=plt.imread(str(designPth)+'en/en_'+str(level)+".png")         # reading backgram photos with png format
    #             plt.imshow(img_en,extent=[-3.84,3.84,-2.16,2.16])                           # location of canvas
    #             plt.plot(x,y,'r-',linewidth=0.5)
    #             x0 = ROI[0][0]
    #             y0 = ROI[0][1]
    #             x1 = ROI[1][0]
    #             y1 = ROI[1][1]
    #             box_x =  [x0,x1,x1,x0,x0]
    #             box_y =  [y0,y0,y1,y1,y0]
    #             plt.plot(box_x,box_y,c='yellow')
    #             thing = stateDict[state]
    #             if state==2:
    #                 mycolor = 'r'
    #             else:
    #                 mycolor = 'b'
    # #             plt.title(f'第{level}题({thing})')
    #             plt.xticks([])
    #             plt.yticks([])
    #             plt.xlim(-5,5)
    #             plt.ylim(-3,3)
    #             outfileName = os.path.join(outputPth,'trail_en_'+str(level)+'.jpg')
    #             fig_en.savefig(outfileName,dpi = 300)
    #             plt.close()
        logger.info('Plotting succeed.')
    except (SystemExit,KeyboardInterrupt):
        raise
    except Exception:
        logger.error("Faild to open sklearn.txt from logger.error",exc_info = True)
    return

#%%                           
if __name__ == "__main__":
    t0 = time.time()
    url = sys.argv[1]
    outputPth = sys.argv[2]
    designPth = sys.argv[3]
    # url = 'https://cos.drbrain.net/profile/tj/2022/8/19/5b894af5-5aee-4e47-be11-8d39516cc529.txt'
    # outputPth = r"E:\spyder_projects\drbrain\my_code\gaze_preprocessing\backup\figures\test_5b894af5-5aee-4e47-be11-8d39516cc529"
    main(url,outputPth,designPth)
    t1 = time.time()
