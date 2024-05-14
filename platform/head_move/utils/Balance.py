"""
 author: zhangchi
 email: zcsjtu0923@gmail.com
 create date: 2023-03-13 11:27:43
"""

import json
import os
import sys
from fastapi import Request
import matplotlib.pyplot as plt
import math
import matplotlib.patches as patches
from matplotlib.patches import Wedge
import matplotlib
from pydantic import BaseModel
import requests

from utils.auth import auth_validate
from utils.logger import get_logger
from utils.response_template import GeneralResponseModel
matplotlib.use('Agg')
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from numpy import linalg as LA
from config.settings import settings

class BalanceRequestModel(BaseModel):
    urlHead: str
    urlBall: str
    outPath: str
    mode: str

def text2Df(myStr: str):
    lines=myStr.split('\n')
    head=lines[0].split(',')
    arr = [x.split(',') for x in lines[1:] if not not x ]
    df = pd.DataFrame(arr,columns = head)
    df['timestamp'] = df['timestamp'].astype('uint64', errors='ignore')
    df['timestamp'] = df['timestamp'] - df.loc[0, 'timestamp']
    df['difficulty'] = df['difficulty'].astype(int, errors='ignore')
    df['pos_x'] = 1*df['pos_x'].astype(float, errors='ignore')
    df['pos_y'] = 1*df['pos_y'].astype(float, errors='ignore')
    df['pos_z'] = 1*df['pos_z'].astype(float, errors='ignore')
    df['pos_x'] = df['pos_x'] - df.loc[0, 'pos_x']
    df['pos_y'] = df['pos_y'] - df.loc[0, 'pos_y']
    df['pos_z' ] = df['pos_z' ] - df.loc[0, 'pos_z']

    df['angle_yaw'] = df['angle_yaw'].astype(float, errors='ignore')
    df['angle_pitch'] = df['angle_pitch'].astype(float, errors='ignore')
    df['angle_roll'] = df['angle_roll'].astype(float, errors='ignore')
    df['angle_yaw'] = df['angle_yaw'] - df.loc[0, 'angle_yaw']
    df['angle_pitch'] = df['angle_pitch'] - df.loc[0, 'angle_pitch']
    df['angle_roll' ] = df['angle_roll' ] - df.loc[0, 'angle_roll']

    df['angle_yaw'] = df['angle_yaw'].apply(lambda x: abs(x) - 360 if abs(x) > 300 else x)
    df['angle_pitch'] = df['angle_pitch'].apply(lambda x: abs(x) - 360 if abs(x) > 300 else x)
    df['angle_roll'] = df['angle_roll'].apply(lambda x: abs(x) - 360 if abs(x) > 300 else x)
    return df

def draw_sav(h_data: pd.DataFrame, mode: str, fig_num: int, read_path:str, sav_path: str):
    assert mode in ['sit', 'stand', 'head'], "invalid mode!"
    assert fig_num == 1 or fig_num == 2 or fig_num == 3, "input fig_num not among [1, 2, 3]!"

    def draw_1(mode):
        points = np.array(h_data[['pos_x', 'pos_z']])
        hull = ConvexHull(points, incremental=True)
        try:
            im_xz = plt.imread(read_path + '{}-{}@2x.png'.format(mode, 1))
        except Exception as e:
            raise e
        
        if mode == 'stand':
            extent = [-100, 100, -175, 25]
            plt.imshow(im_xz, extent=extent)
            x_center, y_center = 0, -67
            r=40
            x_offset, y_offset = 27, 0
        elif mode == 'sit':
            extent = [-60, 60, -100, 20]
            plt.imshow(im_xz, extent=extent)
            x_center, y_center = 0, -35
            r=20
            x_offset, y_offset = 20, 0
        elif mode == 'head':
            extent=[-30, 30, -47, 13]
            plt.imshow(im_xz, extent=extent)
            x_center, y_center = 0, -15
            r=10
            x_offset, y_offset = 8, 2
        # plt.plot(0, 0, 'ro')
        plt.plot(h_data['pos_x'], h_data['pos_z'], color='tomato', lw=0.2, alpha=1)
        new_vertices = np.concatenate((hull.vertices, [hull.vertices[0]]))
        plt.plot(points[new_vertices,0], points[new_vertices,1], 'blue', lw=0.5, alpha=0.4)

        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color='grey', alpha=0.2)

        
        # plt.plot(x_center, y_center, 'ro')

        left_idx = np.argmin(points[:, 0])
        x_left, y_left = points[left_idx, 0], points[left_idx, 1]
        alpha_left = np.arctan(abs(x_left - x_center)/abs(y_left - y_center))
        # plt.plot(x_left, y_left, 'go')

        right_idx = np.argmax(points[:,0])
        x_right, y_right = points[right_idx,0], points[right_idx, 1]
        alpha_right = np.arctan(abs(x_right - x_center)/abs(y_right - y_center))
        # plt.plot(x_right, y_right, 'bo')

        x_start, y_start = x_center, y_center+r
        plt.plot([x_left, x_center, x_right], [y_left, y_center, y_right], 'k--', lw=0.8)


        _x_left, _y_left = x_center - r*math.sin(alpha_left), y_center + r*math.cos(alpha_left)
        _x_right, _y_right = x_center + r*math.sin(alpha_right), y_center + r*math.cos(alpha_right)

        wedge = Wedge([x_center, y_center], r=r, theta1=90-alpha_right*180/math.pi, theta2=90+alpha_left*180/math.pi, color='blue', alpha=0.2)
        style = "Simple, tail_width=0.7, head_width=2, head_length=3"
        kw = dict(arrowstyle=style, color="b", lw=0.5, alpha=0.8)
        arc1 = patches.FancyArrowPatch((x_start, y_start), (_x_left, _y_left), connectionstyle="arc3,rad=0.2", **kw)
        arc2 = patches.FancyArrowPatch((x_start, y_start), (_x_right, _y_right), connectionstyle="arc3,rad=-0.2", **kw)
        plt.gca().add_patch(wedge)
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)

        plt.text(
                x=_x_left-x_offset, y=_y_left,
                s='{:.1f}°'.format(180*alpha_left/math.pi),
                size=9
            )
        plt.text(
                x=_x_right+5, y=_y_right-y_offset,
                s='{:.1f}°'.format(180*alpha_right/math.pi),
                size=9,
            )
        plt.axis('off')
        plt.xlim(extent[0:2]); plt.ylim(extent[2:])
        plt.savefig(sav_path, dpi=400, bbox_inches='tight')
        plt.close()
        return
        

    def draw_2(mode):
        points = np.array(h_data[['pos_y', 'pos_z']])
        hull = ConvexHull(points, incremental=True)
        try:
            im_xz = plt.imread(read_path + '{}-{}@2x.png'.format(mode, 2))
        except Exception as e:
            raise e
        
        if mode == 'stand':
            extent = [-100, 100, -175, 25]
            plt.imshow(im_xz, extent=extent)
            x_center, y_center = 0, -67
            r=40
            x_offset, y_offset = 27, 8
        elif mode == 'sit':
            extent = [-60, 60, -100, 20]
            plt.imshow(im_xz, extent=extent)
            x_center, y_center = 0, -35
            r=20
            x_offset, y_offset = 20, 8
        elif mode == 'head':
            extent = [-30, 30, -47, 13]
            plt.imshow(im_xz, extent=extent)
            x_center, y_center = 0, -15
            r=10
            x_offset, y_offset = 8, 0
        
        # plt.plot(0, 0, 'ro')
        plt.plot(h_data['pos_y'], h_data['pos_z'], color='tomato', lw=0.2, alpha=1)
        new_vertices = np.concatenate((hull.vertices, [hull.vertices[0]]))
        plt.plot(points[new_vertices,0], points[new_vertices,1], 'blue', lw=0.5, alpha=0.4)
        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color='grey', alpha=0.2)
        # plt.plot(x_center, y_center, 'ro')

        left_idx = np.argmin(points[:, 0])
        x_left, y_left = points[left_idx, 0], points[left_idx, 1]
        alpha_left = np.arctan(abs(x_left - x_center)/abs(y_left - y_center))
        # plt.plot(x_left, y_left, 'go')

        right_idx = np.argmax(points[:,0])
        x_right, y_right = points[right_idx,0], points[right_idx, 1]
        alpha_right = np.arctan(abs(x_right - x_center)/abs(y_right - y_center))
        # plt.plot(x_right, y_right, 'bo')

        x_start, y_start = x_center, y_center+r
        plt.plot([x_left, x_center, x_right], [y_left, y_center, y_right], 'k--', lw=0.8)

        _x_left, _y_left = x_center - r*math.sin(alpha_left), y_center + r*math.cos(alpha_left)
        _x_right, _y_right = x_center + r*math.sin(alpha_right), y_center + r*math.cos(alpha_right)

        wedge = Wedge([x_center, y_center], r=r, theta1=90-alpha_right*180/math.pi, theta2=90+alpha_left*180/math.pi, color='blue', alpha=0.2)
        style = "Simple, tail_width=0.7, head_width=2, head_length=3"
        kw = dict(arrowstyle=style, color="b", lw=0.5, alpha=0.8)
        arc1 = patches.FancyArrowPatch((x_start, y_start), (_x_left, _y_left), connectionstyle="arc3,rad=0.2", **kw)
        arc2 = patches.FancyArrowPatch((x_start, y_start), (_x_right, _y_right), connectionstyle="arc3,rad=-0.2", **kw)
        plt.gca().add_patch(wedge)
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)

        plt.text(
                x=_x_left-x_offset, y=_y_left,
                s='{:.1f}°'.format(180*alpha_left/math.pi),
                size=9
            )
        plt.text(
                x=_x_right+5, y=_y_right-y_offset,
                s='{:.1f}°'.format(180*alpha_right/math.pi),
                size=9,
            )
        plt.axis('off')
        plt.xlim(extent[0:2]); plt.ylim(extent[2:])
        plt.savefig(sav_path, dpi=400, bbox_inches='tight')
        plt.close()
        return

    def draw_3(mode):
        points = np.array(h_data[['pos_y', 'pos_x']])
        left_ang = h_data['angle_yaw'].max()
        right_ang = h_data['angle_yaw'].min()
        r=40
        hull = ConvexHull(points, incremental=True)
        try:
            im_xz = plt.imread(read_path + '/top@2x.png')
        except Exception as e:
            raise e

        extent = [-75, 75, -81, 69]
        plt.imshow(im_xz, extent=extent)
        # plt.plot(0, 0, 'ro')
        plt.plot(h_data['pos_y'], h_data['pos_x'], color='tomato', lw=0.2, alpha=1)
        new_vertices = np.concatenate((hull.vertices, [hull.vertices[0]]))
        plt.plot(points[new_vertices,0], points[new_vertices,1], 'blue', lw=0.5, alpha=0.4)

        plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color='grey', alpha=0.2)

        x_center, y_center = 0, 0
        # plt.plot(x_center, y_center, 'ro')

        left_idx = np.argmin(points[:, 0])
        x_left, y_left = x_center + r*math.cos(abs(left_ang)*math.pi/180), y_center + r*math.sin(abs(left_ang)*math.pi/180)
        x_right, y_right = x_center + r*math.cos(abs(right_ang)*math.pi/180), y_center - r*math.sin(abs(right_ang)*math.pi/180)
        # plt.plot(x_left, y_left, 'go')
        # plt.plot(x_right, y_right, 'bo')

        x_start, y_start = x_center+r, y_center
        plt.plot([x_left, x_center, x_right], [y_left, y_center, y_right], 'k--', lw=0.8)

        wedge = Wedge([x_center, y_center], r=r, theta1=right_ang, theta2=left_ang, color='blue', alpha=0.2)
        style = "Simple, tail_width=0.7, head_width=3, head_length=5"
        kw = dict(arrowstyle=style, color="b", lw=0.5, alpha=0.8)
        arc1 = patches.FancyArrowPatch((x_start, y_start), (x_left, y_left), connectionstyle="arc3,rad=0.2", **kw)
        arc2 = patches.FancyArrowPatch((x_start, y_start), (x_right, y_right), connectionstyle="arc3,rad=-0.2", **kw)
        plt.gca().add_patch(wedge)
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)

        plt.text(
                x=x_left, y=y_left+5,
                s='{:.1f}°'.format(abs(left_ang)),
                size=9
            )
        plt.text(
                x=x_right, y=y_right-10,
                s='{:.1f}°'.format(abs(right_ang)),
                size=9,
            )
        plt.axis('off')
        plt.xlim(extent[0:2]); plt.ylim(extent[2:])
        plt.savefig(sav_path, dpi=400, bbox_inches='tight')
        plt.close()
        return

    if fig_num == 1:
        draw_1(mode)
        return
    elif fig_num == 2:
        draw_2(mode)
        return
    else:
        draw_3(mode)
        return

def calc_vel(h_data: pd.DataFrame):
    pos_mat = np.array(h_data[['timestamp', 'pos_x', 'pos_y', 'pos_z']])
    dlt_pos_mat = np.abs(np.diff(pos_mat, axis=0))
    shp = dlt_pos_mat.shape
    new_dlt_pos_mat = dlt_pos_mat[shp[0] % 10 :].reshape((
        shp[0] // 10, 10, shp[-1]
    )).sum(axis=1)
    def div_by_time(x):
        return np.array([x[1]/x[0], x[2]/x[0], x[3]/x[0]])*10
    vel_mat = np.apply_along_axis(div_by_time, 1, new_dlt_pos_mat)
    vel_norm = LA.norm(vel_mat, ord=2, axis=1)

    return list(vel_norm)

def train_time(b_data: pd.DataFrame):
    def label_area(row):
        if row['pos_x'] > 0 and row['pos_y'] > 0:
            return 1
        if row['pos_x'] < 0 and row['pos_y'] > 0:
            return 2
        if row['pos_x'] < 0 and row['pos_y'] < 0:
            return 3
        if row['pos_x'] > 0 and row['pos_y'] < 0:
            return 4

    area = b_data.apply(lambda row: label_area(row), axis=1)
    b_data['area'] = area
    res_df = pd.DataFrame([b_data.groupby('area')['status'].count(), b_data.groupby('area')['status'].sum()], index=['count', 'sum'], columns=[1, 2, 3, 4])
    res_df.loc['succeed'] = 1 - res_df.loc['sum'] / res_df.loc['count']
    res_df.loc['rate'] = res_df.loc['count'] / res_df.loc['count'].sum()
    res_df = res_df.fillna(0)
    return res_df

def balance(model: BalanceRequestModel, request: Request):
    auth_resp = auth_validate(model, request)
    if auth_resp:
        return auth_resp
    # params from request
    url_h = model.urlHead
    url_b = model.urlBall
    out_path = model.outPath
    mode = model.mode

    # log setting
    _, sid = os.path.split(url_h)
    if not os.path.exists(out_path + '/log'):
        os.mkdir(out_path + '/log')
    logger = get_logger(sid, out_path)
    logger.info('Authorization succeed.')

    try:
        # requests to get data
        if settings.deploy_mode == 'offline':
            with open(url_h) as f:
                head_data = f.read()
        else:
            with requests.get(url_h) as url_data:
                if url_data.status_code != 200:
                    logger.error("Cannot access url data!")
                    sys.exit()
                head_data = url_data.text
        
        txt_head_data = text2Df(head_data)
        des_head_data = txt_head_data.describe()

        ball_data = pd.read_csv(url_b)
        train_res = train_time(ball_data)

        if not os.path.exists(out_path + '/result_fig'):
            os.mkdir(out_path + '/result_fig')
        for n in [1, 2, 3]:
            draw_sav(txt_head_data, mode, n, './assets/src_fig/', out_path+'/result_fig/'+"traj{}.png".format(n))
            logger.info("traj{} completed.".format(n))
        vel_list = calc_vel(txt_head_data)
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