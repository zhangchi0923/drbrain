# -*- coding: utf-8 -*-

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
import random
import pandas as pd
from Pingpong import Pingpong
from Balance import Balancer
from EyeScreen import EyeScreen
import warnings
warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)


from settings import SALT, RESOURCE_PATH
def get_md5(d):
    d = dict(sorted(d.items()))
    s=''
    for k, v in d.items():
        s += str(k) + str(v)
    s = SALT + s
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()

def mkdir_new(path):
        if not os.path.exists(path):
            os.makedirs(path) 

@app.route('/balance', methods=['POST'])
def balance():
    args = request.get_json(force=True)
    auth = request.headers.get('Authorization')
    auth_srv = get_md5(args)
    if auth_srv != auth:
        # logger.error('Authorization failed.\nAuth from client:{}\nAuth from server:{}'.format(auth, auth_srv))
        return jsonify({
            'code':401,
            'msg':'Authorization failed.',
            'body':None
        })
    url_h = args['urlHead']
    url_b = args['urlBall']
    out_path = args['outPath']
    mode = args['mode']

    balancer = Balancer(out_path, mode)

    try:
        # log setting
        _, sid = os.path.split(url_h)
        # print(sid)
        mkdir_new(out_path + '/log')
        logger = balancer.get_logger(sid, out_path + '/log')
        logger.info('Authorization succeed.')
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
            balancer.draw_sav(txt_head_data, mode, n, './src_fig/', out_path+'/result_fig/'+"traj{}.png".format(n))
            logger.info("traj{} completed.".format(n))
        vel_list = balancer.calc_vel(txt_head_data)
        with open(out_path+'/vel.json', 'w') as f:
            json.dump(vel_list, f)

        out_dict = {'code':200,
                    'msg':'Computation completed successfully.',
                    'body':{
                        'forth': '{:.2f}'.format(des_head_data.loc['max', 'pos_y']), 
                        'back': '{:.2f}'.format(abs(des_head_data.loc['min', 'pos_y'])), 
                        'left': '{:.2f}'.format(des_head_data.loc['max', 'pos_x']), 
                        'right': '{:.2f}'.format(abs(des_head_data.loc['min', 'pos_x'])), 
                        'up': '{:.2f}'.format(des_head_data.loc['max', 'pos_z']), 
                        'down': '{:.2f}'.format(abs(des_head_data.loc['min', 'pos_z'])),
                        
                        'rightForthSuc': '{:.2f}%'.format(train_res.loc['succeed', 1]), 
                        'leftForthSuc': '{:.2f}%'.format(train_res.loc['succeed', 2]), 
                        'leftBackSuc': '{:.2f}%'.format(train_res.loc['succeed', 3]),
                        'rightBackSuc': '{:.2f}%'.format(train_res.loc['succeed', 4]),
                        
                        'rightForthPct': '{:.2f}%'.format(train_res.loc['rate', 1]), 
                        'leftForthPct': '{:.2f}%'.format(train_res.loc['rate', 2]), 
                        'leftBackPct': '{:.2f}%'.format(train_res.loc['rate', 3]),
                        'rightBackPct': '{:.2f}%'.format(train_res.loc['rate', 4])

                    }          
        }

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
        logger.info("Right_forth: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 1], train_res.loc['rate', 1]))
        logger.info("Left_forth: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 2], train_res.loc['rate', 2]))
        logger.info("Left_back: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 3], train_res.loc['rate', 3]))
        logger.info("Right_back: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 4], train_res.loc['rate', 4]))
        logger.info("--------------END---------------")

        return jsonify(out_dict)
    except Exception as e:
        logger.exception(e, stacklevel=1)
        return jsonify({'code':500, 'msg': str(e), 'body':None})

@app.route('/pingpong', methods=['POST'])
def pingpong():
    args = request.get_json(force=True)
    auth = request.headers.get('Authorization')
    auth_srv = get_md5(args)
    if auth_srv != auth:
        # logger.error('Authorization failed.\nAuth from client:{}\nAuth from server:{}'.format(auth, auth_srv))
        return jsonify({
            'code':401,
            'msg':'Authorization failed.',
            'body':None
        })
    url = args['dataUrl']
    hold_type = args['holdType']
    read_pth = RESOURCE_PATH
    write_pth = args['path']

    pp = Pingpong(hold_type, read_pth, write_pth + "/pp_result_fig")

    try:
        # log setting
        _, sid = os.path.split(url)
        # print(sid)
        mkdir_new(write_pth + '/log')
        logger = pp.get_logger(sid, write_pth + '/log')
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

        out_dict = {'code':200,
                    'msg':'Computation completed successfully.',
                    'body':{
                        'forth': '{:.2f}'.format(abs(des_head_data.loc['max', 'handPosZ'])), 
                        'back': '{:.2f}'.format(abs(des_head_data.loc['min', 'handPosZ'])), 
                        'left': '{:.2f}'.format(abs(des_head_data.loc['max', 'handPosX'])), 
                        'right': '{:.2f}'.format(abs(des_head_data.loc['min', 'handPosX'])), 
                        'up': '{:.2f}'.format(abs(des_head_data.loc['max', 'handPosY'])), 
                        'down': '{:.2f}'.format(abs(des_head_data.loc['min', 'handPosY'])),
                    }          
        }

        logger.info("Ranges calculated.")
        logger.info("--------------------------------")
        logger.info("Forth: {:.2f}".format(abs(des_head_data.loc['max', 'handPosZ'])))
        logger.info("Back: {:.2f}".format(abs(des_head_data.loc['min', 'handPosZ'])))
        logger.info("Left: {:.2f}".format(abs(des_head_data.loc['max', 'handPosX'])))
        logger.info("Right: {:.2f}".format(des_head_data.loc['min', 'handPosX']))
        logger.info("Up: {:.2f}".format(abs(des_head_data.loc['max', 'handPosY'])))
        logger.info("Down: {:.2f}".format(abs(des_head_data.loc['min', 'handPosY'])))
        logger.info("--------------END---------------")

        return jsonify(out_dict)
    except Exception as e:
        logger.exception(e, stacklevel=1)
        return jsonify({'code':500, 'msg': str(e), 'body':None})

@app.route('/eye/screen', methods=['POST'])
def eye_screen():
    args = request.get_json(force=True)
    auth = request.headers.get('Authorization')
    auth_srv = get_md5(args)
    if auth_srv != auth:
        # logger.error('Authorization failed.\nAuth from client:{}\nAuth from server:{}'.format(auth, auth_srv))
        return jsonify({
            'code':401,
            'msg':'Authorization failed.',
            'body':None
        })
    gender = args['sex']
    age = args['age']
    education = args['education']
    url = args['url']
    src = args['backupResources']
    save_pth = args['saveResourcesPath']


    executer = ThreadPoolExecutor(2)
    executer.submit(draw_eye_screen, url, save_pth, src)
    # logger.info('Draw task submitted successfully.')

    
    with requests.get(url) as r:
        assert r.status_code == 200, 'HTTP Connection Error: {}, {}'.format(r.status_code, r.content)
        # logger.error("Cannot access url data!")
        txt = r.text
    
    try:
        es = EyeScreen(txt, gender, education, age)
        data = es.preprocess_feat(es.text2DF())
        moca, mmse = es.predict(data)
        moca = 30 if moca > 28 else moca + 1 + 0.1*random.randint(0, 10)
        mmse = 30 if mmse > 28 else mmse + 1 + 0.1*random.randint(0, 10)
        cog_score = es.cog_score(data)
    except Exception as e:
        raise e
        # logger.exception(e)
        return jsonify({
        'code':500,
        # 'msg':'Error during score predicting.',
        'msg':e,
        'body':None
    })
        sys.exit()

    results = {
        'mmse':round(moca, 1),
        'moca':round(mmse, 1),
        'resultScores':[
        {'level':1, 'score':round(cog_score[0], 1)},{'level':2, 'score':round(cog_score[1], 1)},{'level':3, 'score':round(cog_score[2], 1)},
        {'level':4, 'score':round(cog_score[3], 1)},{'level':5, 'score':round(cog_score[4], 1)},{'level':6, 'score':round(cog_score[5], 1)},
        {'level':7, 'score':round(cog_score[6], 1)},{'level':8, 'score':round(cog_score[7], 1)},{'level':9, 'score':round(cog_score[8], 1)},
        {'level':10, 'score':round(cog_score[9], 1)}
        ]
    }
    return jsonify({
        'code':200,
        'msg':'AI prediction succeed.',
        'body':results
    })

def draw_eye_screen(url, out_pth, design_pth):
    os.system('python ./justscore_bySection_4urldata_final.py {} {} {}'.format(url, out_pth, design_pth))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8101, debug=True)
    