"""
 author: zhangchi
 email: zcsjtu0923@gmail.com
 create date: 2022-12-15 11:36:19
"""

import os
import logging
import datetime
from utils.Pingpong import pingpong
from utils.Balance import balance
from utils.EyeScreen import eye_screen
from utils.Firefly import firefly
import utils.PCAT as PCAT
from utils.Cervical import Cervical
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


# from config.settings import SALT, RESOURCE_PATH

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

app.post("/balance")(balance)
app.post("/pingpong")(pingpong)
app.post("/eye/screen")(eye_screen)


app.post("/eye/train/firefly")(firefly)

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
    