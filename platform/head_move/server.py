"""
 author: zhangchi
 email: zcsjtu0923@gmail.com
 create date: 2022-12-15 11:36:19
"""

import os
import sys
import requests
import json
import hashlib
import logging
import datetime
import pandas as pd
from utils.Pingpong import Pingpong
from utils.Balance import Balancer
from utils.EyeScreen import EyeScreen
from utils.Firefly import Firefly
import utils.PCAT as PCAT
from utils.Cervical import Cervical
import utils.pbb_score as pbb_score
from utils.auth import auth_validate
from utils.response_template import GeneralResponseModel
import warnings
warnings.filterwarnings('ignore')

from fastapi import FastAPI, Request, BackgroundTasks
from pydantic import BaseModel
from typing import Union

import uvicorn

# from flask import Flask, request, jsonify
from concurrent.futures import ProcessPoolExecutor

# app = Flask(__name__)
app = FastAPI()


from config.settings import SALT, RESOURCE_PATH

def mkdir_new(path):
        if not os.path.exists(path):
            os.makedirs(path) 

def get_logger(log_id, pth):
        date_time = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
        exe_logger = logging.getLogger()
        exe_logger.setLevel(level=logging.INFO)
        handler = logging.FileHandler(os.path.join(pth, 'log_' + date_time + '_' + log_id))
        handler.setLevel(logging.INFO)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)

        exe_logger.addHandler(handler)
        return exe_logger


# Balance request model
class BalanceRequestModel(BaseModel):
    urlHead: str
    urlBall: str
    outPath: str
    mode: str

# @app.route('/balance', methods=['POST'])
@app.post("/balance")
def balance(model: BalanceRequestModel, request: Request):
    # args = request.get_json(force=True)
    auth_resp = auth_validate(model, request)
    if auth_resp:
        return auth_resp

    url_h = model.urlHead
    url_b = model.urlBall
    out_path = model.outPath
    mode = model.mode

    balancer = Balancer(out_path, mode)
    # log setting
    _, sid = os.path.split(url_h)
    # print(sid)
    mkdir_new(out_path + '/log')
    logger = balancer.get_logger(sid, out_path)
    logger.info('Authorization succeed.')

    try:
        # requests to get data
        with requests.get(url_h) as url_data:
            if url_data.status_code != 200:
                logger.error("Cannot access url data!")
                sys.exit()
            txt_head_data = balancer.text2Df(url_data.text)
            des_head_data = txt_head_data.describe()

        ball_data = pd.read_csv(url_b)
        train_res = balancer.train_time(ball_data)

        mkdir_new(out_path + '/result_fig')
        for n in [1, 2, 3]:
            balancer.draw_sav(txt_head_data, mode, n, './assets/src_fig/', out_path+'/result_fig/'+"traj{}.png".format(n))
            logger.info("traj{} completed.".format(n))
        vel_list = balancer.calc_vel(txt_head_data)
        with open(out_path+'/vel.json', 'w') as f:
            json.dump(vel_list, f)

        resp = GeneralResponseModel(
            code=200,
            msg='Computation completed successfully.',
            body={
                'forth': '{:.2f}'.format(des_head_data.loc['max', 'pos_y']), 
                'back': '{:.2f}'.format(abs(des_head_data.loc['min', 'pos_y'])), 
                'left': '{:.2f}'.format(des_head_data.loc['max', 'pos_x']), 
                'right': '{:.2f}'.format(abs(des_head_data.loc['min', 'pos_x'])), 
                'up': '{:.2f}'.format(des_head_data.loc['max', 'pos_z']), 
                'down': '{:.2f}'.format(abs(des_head_data.loc['min', 'pos_z'])),
                
                'rightForthSuc': '{:.2f}%'.format(train_res.loc['succeed', 1]*100), 
                'leftForthSuc': '{:.2f}%'.format(train_res.loc['succeed', 2]*100), 
                'leftBackSuc': '{:.2f}%'.format(train_res.loc['succeed', 3]*100),
                'rightBackSuc': '{:.2f}%'.format(train_res.loc['succeed', 4]*100),
                
                'rightForthPct': '{:.2f}%'.format(train_res.loc['rate', 1]*100), 
                'leftForthPct': '{:.2f}%'.format(train_res.loc['rate', 2]*100), 
                'leftBackPct': '{:.2f}%'.format(train_res.loc['rate', 3]*100),
                'rightBackPct': '{:.2f}%'.format(train_res.loc['rate', 4]*100)
            }
        )

        logger.info("Velocity calculated.")
        logger.info("--------------------------------")
        logger.info("Forth: {:.2f}".format(des_head_data.loc['max', 'pos_y']))
        logger.info("Back: {:.2f}".format(des_head_data.loc['min', 'pos_y']))
        logger.info("Left: {:.2f}".format(des_head_data.loc['max', 'pos_x']))
        logger.info("Right: {:.2f}".format(des_head_data.loc['min', 'pos_x']))
        logger.info("Up: {:.2f}".format(des_head_data.loc['max', 'pos_z']))
        logger.info("Down: {:.2f}".format(des_head_data.loc['min', 'pos_z']))
        logger.info("--------------------------------")
        logger.info("Succeed rates calculated.")
        logger.info("Right_forth: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 1]*100, train_res.loc['rate', 1]*100))
        logger.info("Left_forth: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 2]*100, train_res.loc['rate', 2]*100))
        logger.info("Left_back: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 3]*100, train_res.loc['rate', 3]*100))
        logger.info("Right_back: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 4]*100, train_res.loc['rate', 4]*100))
        logger.info("--------------END---------------")

        return resp
    except Exception as e:
        logger.exception(e, stacklevel=1)
        return GeneralResponseModel(code=500, msg=str(e), body=None)

class PingpongRequestModel(BaseModel):
    dataUrl: str
    holdType: str
    path: str
# @app.route('/pingpong', methods=['POST'])
@app.post('/pingpong')
def pingpong(model: PingpongRequestModel, request: Request):

    auth_resp = auth_validate(model, request)
    if auth_resp:
        return auth_resp
    url = model.dataUrl
    hold_type = model.holdType
    read_pth = RESOURCE_PATH
    write_pth = model.path

    pp = Pingpong(hold_type, read_pth, write_pth + "/pp_result_fig")

    try:
        # log setting
        _, sid = os.path.split(url)
        # print(sid)
        mkdir_new('./log/pingpong_log')
        logger = pp.get_logger(sid, './log/pingpong_log')
        logger.info('Authorization succeed.')
        # requests to get data
        with requests.get(url) as url_data:
            if url_data.status_code != 200:
                logger.error("Cannot access url data!")
                sys.exit()
            raw_data = pp.text2Df(url_data.text)
            des_head_data = raw_data.describe()
        
        mkdir_new(write_pth + "/pp_result_fig")
        pp.traj_draw(data=raw_data)
        pp.rot_draw(data=raw_data)
        vel_list = pp.calc_vel(raw_data)
        with open(write_pth+'/vel.json', 'w') as f:
            json.dump(vel_list, f)

        resp = GeneralResponseModel(code=200,
                    msg='Computation completed successfully.',
                    body={
                        'forth': '{:.2f}'.format(abs(des_head_data.loc['max', 'handPosZ'])), 
                        'back': '{:.2f}'.format(abs(des_head_data.loc['min', 'handPosZ'])), 
                        'left': '{:.2f}'.format(abs(des_head_data.loc['max', 'handPosX'])), 
                        'right': '{:.2f}'.format(abs(des_head_data.loc['min', 'handPosX'])), 
                        'up': '{:.2f}'.format(abs(des_head_data.loc['max', 'handPosY'])), 
                        'down': '{:.2f}'.format(abs(des_head_data.loc['min', 'handPosY'])),
                    }          
        )

        logger.info("Ranges calculated.")
        logger.info("--------------------------------")
        logger.info("Forth: {:.2f}".format(abs(des_head_data.loc['max', 'handPosZ'])))
        logger.info("Back: {:.2f}".format(abs(des_head_data.loc['min', 'handPosZ'])))
        logger.info("Left: {:.2f}".format(abs(des_head_data.loc['max', 'handPosX'])))
        logger.info("Right: {:.2f}".format(des_head_data.loc['min', 'handPosX']))
        logger.info("Up: {:.2f}".format(abs(des_head_data.loc['max', 'handPosY'])))
        logger.info("Down: {:.2f}".format(abs(des_head_data.loc['min', 'handPosY'])))
        logger.info("--------------END---------------")

        return resp
    except Exception as e:
        logger.exception(e, stacklevel=1)
        return GeneralResponseModel(code=500, msg=str(e), body=None)

class EyeScreenReqeustModel(BaseModel):
    sex: str
    age: int
    education: str
    url: str
    saveResourcesPath: str
    questionVersion: str

@app.post('/eye/screen')
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
    mkdir_new('./log/eyescreen_log')
    logger = get_logger(sid, './log/eyescreen_log')
    logger.info('Authorization succeed.')

    # executer = ProcessPoolExecutor(1)
    # executer.submit(draw_eye_screen, url, save_pth, src)
    background_tasks.add_task(draw_eye_screen, url, save_pth, src)
    logger.info('Eye screen plot submitted.')
    # print(os.getcwd(), src, save_pth)
    results = calc_eye_screen(url, gender, education, age, save_pth, src, logger)
    logger.info('Eye screen results: {}'.format(results))
    return results

def calc_eye_screen(url, gender, education, age, save_pth, src, logger):
    with requests.get(url) as r:
        if r.status_code != 200 :
            logger.error("Cannot access url data!")
        txt = r.text
    
    try:
        es = EyeScreen(txt, gender, education, age)
        data = es.preprocess_feat(es.text2DF())
        moca, mmse = es.predict(data)
        cog_score = pbb_score.main(url, save_pth, src)
        cog_score = [x*100 for x in cog_score]
        logger.info('Cog score: {}\nMoCA: {} MMSE: {}'.format(cog_score, moca, mmse))
    except Exception as e:
        logger.exception(e)
        return GeneralResponseModel(
            code=500,
            # 'msg':'Error during score predicting.',
            msg=str(e),
            body=None
        )

    body = {
        'mmse':round(mmse, 1),
        'moca':round(moca, 1),
        'resultScores':[
        {'level':1, 'score':round(cog_score[0], 1)},{'level':2, 'score':round(cog_score[1], 1)},{'level':3, 'score':round(cog_score[2], 1)},
        {'level':4, 'score':round(cog_score[3], 1)},{'level':5, 'score':round(cog_score[4], 1)},{'level':6, 'score':round(cog_score[5], 1)},
        {'level':7, 'score':round(cog_score[6], 1)},{'level':8, 'score':round(cog_score[7], 1)},{'level':9, 'score':round(cog_score[8], 1)},
        {'level':10, 'score':round(cog_score[9], 1)}
        ]
    }

    results = {
        'code':200,
        'msg':'AI prediction succeed.',
        'body':body
    }
    logger.info('Eye screen prediction succeed.')
    return results

def draw_eye_screen(url, out_pth, design_pth):
    os.system('python ./utils/justscore_bySection_4urldata_final.py {} {} {}'.format(url, out_pth, design_pth))


class FireflyRequestModel(BaseModel):
    url: str
    savePath: str
# create app 'eye/train/firefly' with POST method. Related class was created in eye/train/Firefly.py
# @app.route('/eye/train/firefly', methods=['POST'])
@app.post('/eye/train/firefly')
def firefly(model: FireflyRequestModel, request: Request):

    auth_resp = auth_validate(model, request)
    if auth_resp:
        return auth_resp
    url = model.url
    savePath = model.savePath
    mkdir_new(savePath)

    _, sid = os.path.split(url)
    mkdir_new('./log/firefly_log')
    logger = get_logger(sid, './log/firefly_log')
    try:
        df = pd.read_csv(url, index_col=0)
        # 加了012字段，匹配新老版本机器
        if df.shape[1] == 4:
            df = df.loc[df.iloc[:, -1] == 0, ['timestamp', 'state', 'x']]
            df.columns = ['state', 'x', 'y']

        logger.info('Url data read successfully.')
        firefly = Firefly(df, savePath)
        firefly.plot()
        logger.info('Firefly plot succeed.')
        return GeneralResponseModel(code=200, msg='Firefly plot succeed.', body=None)

    except Exception as e:
        logger.exception(e)
        return GeneralResponseModel(code=500, msg=str(e), body=None)

class PcatRequestModel(BaseModel):
    id: str
    type: str
    url: str
# @app.route('/eye/pcat', methods=['POST'])
@app.post('/eye/pcat')
def eye_pcat(model: PcatRequestModel, request: Request, background_tasks: BackgroundTasks):
    '''
    :param id: 筛查ID
    :param type: 筛查类型
    :param url: 眼动数据cos地址
    :return code: 返回码 e.g.200, 404
    :return objects_url: 对象存储地址列表
    '''

    auth_resp = auth_validate(model, request)
    if auth_resp:
        return auth_resp
    id, type, url = model.id, model.type, model.url
    mkdir_new('./log/pcat_log')
    _, sid = os.path.split(url)
    logger = get_logger(sid, './log/pcat_log')
    try:
        pcat = PCAT.Pcat(id, type, url)
        objects_urls = pcat.make_cos_urls()
        # executer = ProcessPoolExecutor(1)
        # executer = ThreadPoolExecutor(2)
        # executer.submit(draw_pcat, id, type, url)
        background_tasks.add_task(draw_pcat, id, type, url)
        logger.info("PCAT Plot task submitted.")
        resp = GeneralResponseModel(
            code=200,
            body={
                'objectsUrls': objects_urls
            },
            msg='success'
        )
        return resp
    except Exception as e:
        logger.exception(str(e))
        if isinstance(e, ConnectionError) or isinstance(e, KeyError):
            return GeneralResponseModel(code=503, body={'objectUrls': []}, msg=str(e))
        
        return GeneralResponseModel(code=500, body={'objectUrls': []}, msg=str(e))
        # return jsonify({
        #     'code':500, 
        #     'body':{'objectsUrls': []}, 
        #     'msg': str(e)
        # })

def draw_pcat(id, type, url):
    os.system('python ./utils/pcat_draw.py {} {} {}'.format(id, type, url))

class CervicalReuqestModel(BaseModel):
    id: str
    url: str

# @app.route('/rehab/sd/cervical', methods=['POST'])
@app.post('/rehab/sd/cervical')
def sd_cervical(model: CervicalReuqestModel, request: Request):

    auth_resp = auth_validate(model, request)
    if auth_resp:
        return auth_resp
    id, url = model.id, model.url
    mkdir_new('./log/sd_cervical_log')
    _, sid = os.path.split(url)
    logger = get_logger(sid, './log/sd_cervical_log')
    try:
        sd_cervical = Cervical(url, id)
        sd_cervical.get_url_data()
        magnitudes = sd_cervical.get_magnitude()
        json_keys = sd_cervical.get_vel_ang()
        img_keys = sd_cervical.draw()

        resp = GeneralResponseModel(
            code=200, 
            body={
            'magnitude': magnitudes,
                'urls': {
                    'img': img_keys,
                    'vel': json_keys
                }
            },
            msg='success'
        )

        return resp
    except Exception as e:
        logger.exception(str(e))
        if isinstance(e, ConnectionError) or isinstance(e, KeyError):
            return GeneralResponseModel(code=503, body={'magnitude':[]}, msg=str(e))
        return GeneralResponseModel(code=500, body={'magnitude': []}, msg=str(e))

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8101)
    