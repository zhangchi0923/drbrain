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
import pandas as pd
from Pingpong import Pingpong
from Balance import Balancer
from flask import Flask, request, jsonify

app = Flask(__name__)


from settings import SALT, RESOURCE_PATH
def get_md5(d):
    d = dict(sorted(d.items()))
    s=''
    for k, v in d.items():
        s += k+v
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
            balancer.draw_sav(txt_head_data, mode, n, './src_fig/', out_path+'/result_fig/'+"traj_{}_{}.png".format(mode, n))
            logger.info("traj_{}_{} completed.".format(mode, n))
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
                        
                        'rightForthSuc': '{:.2f}%'.format(train_res.loc['succeed', 1]*100), 
                        'leftForthSuc': '{:.2f}%'.format(train_res.loc['succeed', 2]*100), 
                        'leftBackSuc': '{:.2f}%'.format(train_res.loc['succeed', 3]*100),
                        'rightBackSuc': '{:.2f}%'.format(train_res.loc['succeed', 4]*100),
                        
                        'rightForthPct': '{:.2f}%'.format(train_res.loc['rate', 1]*100), 
                        'leftForthPct': '{:.2f}%'.format(train_res.loc['rate', 2]*100), 
                        'leftBackPct': '{:.2f}%'.format(train_res.loc['rate', 3]*100),
                        'rightBackPct': '{:.2f}%'.format(train_res.loc['rate', 4]*100)

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
        logger.info("Right_forth: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 1]*100, train_res.loc['rate', 1]*100))
        logger.info("Left_forth: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 2]*100, train_res.loc['rate', 2]*100))
        logger.info("Left_back: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 3]*100, train_res.loc['rate', 3]*100))
        logger.info("Right_back: suc--{:.2f}% pct--{:.2f}%".format(train_res.loc['succeed', 4]*100, train_res.loc['rate', 4]*100))
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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8101, debug=True)
    