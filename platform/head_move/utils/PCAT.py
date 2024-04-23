import datetime
import logging
import os
from fastapi import BackgroundTasks, Request
import pandas as pd
from pydantic import BaseModel
import requests
from config.settings import PREFIX, URL_PREFIX
from utils.auth import auth_validate
from utils.response_template import GeneralResponseModel

class PcatRequestModel(BaseModel):
    id: str
    type: str
    url: str

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
    
def text2DF(txt):
    lines = txt.strip().split('\n')
    head = lines[0].split(',')
    arr = [x.split(',') for x in lines[1:] if not not x]
    df = pd.DataFrame(arr, columns=head)
    df['timestamp'] = df['timestamp'].astype('uint64', errors='raise')
    df['timestamp'] = df['timestamp'] - df.loc[0, 'timestamp']
    df['team'] = df['team'].astype(int, errors='raise')
    df['id'] = df['id'].astype(int, errors='raise')
    df['x'] = df['x'].astype(float, errors='raise')
    df['y'] = df['y'].astype(float, errors='raise')

    df.dropna(how='any', axis=0, inplace=True)
    return df

def get_url_and_img_num(url):

    try:
        with requests.get(url) as url_data:
            assert url_data.status_code == 200
            txt = url_data.text
        df = text2DF(txt)
        # TODO: 根据眼动数据格式确定 img_num
        img_num = df.loc[df['team'] >=0, 'id'].unique().shape[0]
        return img_num

    except Exception as e:
        if isinstance(e, AssertionError) or isinstance(e, requests.ConnectionError):
            raise ConnectionError("Error during fetching eye tracking url data.")
        elif isinstance(e, KeyError):
            raise KeyError("Mismatched eye tracking data columns!")
        else:
            raise e

def make_cos_urls(id, type, url):
    img_num = get_url_and_img_num(url)
    cos_urls = []
    now = datetime.datetime.now()
    year, month, day = str(now.year), str(now.month), str(now.day)
    base_key = '/'.join([PREFIX, year, month, day, str(id), type])
    for i in range(1, img_num + 1):
        key = base_key + '/{}.jpg'.format(i)
        cos_urls.append(URL_PREFIX + key)
    return cos_urls

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
    if not os.path.exists('./log/pcat_log'):
        os.mkdir('./log/pcat_log')

    _, sid = os.path.split(url)
    logger = get_logger(sid, './log/pcat_log')
    try:
        objects_urls = make_cos_urls(id, type, url)
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

def draw_pcat(id, type, url):
    os.system('python ./utils/pcat_draw.py {} {} {}'.format(id, type, url))