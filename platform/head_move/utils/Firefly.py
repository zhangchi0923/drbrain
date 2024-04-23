"""
 author: zhangchi
 email: zcsjtu0923@gmail.com
 create date: 2023-03-29 14:44:32
"""

import datetime
import logging
from fastapi import Request
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from pydantic import BaseModel

from utils.auth import auth_validate
from utils.response_template import GeneralResponseModel
warnings.filterwarnings("ignore")

class FireflyRequestModel(BaseModel):
    url: str
    savePath: str

    
def plot(df, save_path):
    x = df['x']
    y = df['y']
    # plot two figures separately and save them without white spaces
    with plt.rc_context():
        # print(os.getcwd())
        img = plt.imread('./assets/src_firefly/fire_fly_bg.png')
        plt.imshow(img, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
        plt.plot(x, y, color="red", linewidth=0.5)
        plt.axis('off')
        # save figure without white spaces
        plt.savefig(save_path+'/traj.png', bbox_inches='tight')
        plt.close()

        from scipy.stats.kde import gaussian_kde
        k = gaussian_kde(np.vstack([x, y]))
        xi, yi = np.mgrid[x.min():x.max():x.size**0.5*1j,y.min():y.max():y.size**0.5*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        im = plt.imread('./assets/src_firefly/fire_fly_bg.png')
        plt.imshow(im, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto')
        plt.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.4, cmap='jet')
        plt.xlim(x.min(), x.max())
        plt.ylim(y.min(), y.max())
        plt.axis('off')
        # save figure without white spaces
        plt.savefig(save_path+'/heatmap.png', bbox_inches='tight')
        plt.close()

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

def firefly(model: FireflyRequestModel, request: Request):

    auth_resp = auth_validate(model, request)
    if auth_resp:
        return auth_resp
    url = model.url
    save_path = model.savePath
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    _, sid = os.path.split(url)
    if not os.path.exists('./log/firefly_log'):
        os.mkdir('./log/firefly_log')
    logger = get_logger(sid, './log/firefly_log')
    try:
        df = pd.read_csv(url, index_col=0)
        # 加了012字段，匹配新老版本机器
        if df.shape[1] == 4:
            df = df.loc[df.iloc[:, -1] == 0, ['timestamp', 'state', 'x']]
            df.columns = ['state', 'x', 'y']

        logger.info('Url data read successfully.')
        plot(df, save_path)
        logger.info('Firefly plot succeed.')
        return GeneralResponseModel(code=200, msg='Firefly plot succeed.', body=None)

    except Exception as e:
        logger.exception(e)
        return GeneralResponseModel(code=500, msg=str(e), body=None)


    