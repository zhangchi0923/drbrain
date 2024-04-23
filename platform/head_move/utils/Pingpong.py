"""
 author: zhangchi
 email: zcsjtu0923@gmail.com
 create date: 2023-02-10 13:36:10
"""

import datetime
import json
import logging
import os
import random
import sys
from fastapi import Request
import numpy as np
import pandas as pd
from pydantic import BaseModel
import requests
from scipy.spatial import ConvexHull
from numpy import linalg as LA
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Wedge
from config.settings import PP_PATH
from utils.auth import auth_validate
from utils.logger import get_logger

from utils.response_template import GeneralResponseModel

class PingpongRequestModel(BaseModel):
    dataUrl: str
    holdType: str
    path: str


def text2Df(myStr):
    lines=myStr.split('\n')
    head=lines[0].split(',')
    arr = [x.split(',') for x in lines[1:] if not not x ]
    df = pd.DataFrame(arr,columns = head)
    df['timestamp'] = df['timestamp'].astype('uint64', errors='ignore')
    df['timestamp'] = df['timestamp'] - df['timestamp'][0]
    df['headX'] = 100*df['headX'].astype(float, errors='ignore')
    df['headY'] = 100*df['headY'].astype(float, errors='ignore')
    df['headZ'] = 100*df['headZ'].astype(float, errors='ignore')
    df['handPosX'] = 100*df['handPosX'].astype(float, errors='ignore')
    df['handPosY'] = 100*df['handPosY'].astype(float, errors='ignore')
    df['handPosZ'] = 100*df['handPosZ'].astype(float, errors='ignore')

    df['handRotX'] = df['handRotX'].astype(float, errors='ignore')
    df['handRotY'] = df['handRotY'].astype(float, errors='ignore')
    df['handRotZ'] = df['handRotZ'].astype(float, errors='ignore')
    
    '''
    手的坐标以头部为原点
    '''
    df['handPosX'] = df['handPosX'] - df['headX']
    df['handPosY'] = df['handPosY'] - df['headY']
    df['handPosZ'] = df['handPosZ'] - df['headZ']
    # df = df.apply(lambda x: x-df.iloc[0, :], axis=1)
    return df


def traj_draw(data, hold_type, read_path, write_path):
    r_pth = read_path
    w_pth = write_path
    traj_table(data, r_pth, w_pth)
    traj_front(data, r_pth, w_pth)
    if hold_type == 'A':
        traj_left(data, r_pth, w_pth)
    elif hold_type == 'B':
        traj_right(data, r_pth, w_pth)
    return

def rot_draw(data, hold_type, read_path, write_path):
    if hold_type == 'A':
        rot_left_side(data, read_path, write_path)
        rot_left_shoulder(data, read_path, write_path)
        rot_left_top_shoulder(data, read_path, write_path)
        rot_left_bow(data, read_path, write_path)
    elif hold_type == 'B':
        rot_right_side(data, read_path, write_path)
        rot_right_shoulder(data, read_path, write_path)
        rot_right_top_shoulder(data, read_path, write_path)
        rot_right_bow(data, read_path, write_path)
    return

def traj_table(data, read_pth, write_pth):
    points = np.array(data[['handPosX', 'handPosZ']])
    hull = ConvexHull(points, incremental=True)
    
    im = plt.imread(read_pth+'/table@2x.png')
    plt.imshow(im, extent=[-150, 150, -55, 115])
    # plt.plot(0, 0, 'bo', lw=1)
    plt.plot(points[:, 0], points[:, 1], markersize=100, lw=0.4, alpha=0.7, color='cornflowerblue')
    new_vertices = np.concatenate((hull.vertices, [hull.vertices[0]]))
    plt.plot(points[new_vertices,0], points[new_vertices,1], 'lightblue', lw=0.5, alpha=0.9)
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color='grey', alpha=0.2)
    plt.axis('off')
    plt.savefig(write_pth+'/top.png', dpi=300, bbox_inches='tight')
    plt.close()
    return

def traj_front(data, read_pth, write_pth):
    points = np.array(data[['handPosX', 'handPosY']])
    points[:, 0] = -points[:, 0]
    hull = ConvexHull(points, incremental=True)
    
    im = plt.imread(read_pth+'/front-1@2x.png')
    plt.imshow(im, extent=[-95, 95, -160, 30])
    # plt.plot(0, 0, 'bo', lw=1)
    plt.plot(points[:, 0], points[:, 1], markersize=100, lw=0.4, alpha=0.7, color='cornflowerblue')
    new_vertices = np.concatenate((hull.vertices, [hull.vertices[0]]))
    plt.plot(points[new_vertices,0], points[new_vertices,1], 'lightblue', lw=0.5, alpha=0.9)
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color='grey', alpha=0.2)
    plt.axis('off')
    plt.savefig(write_pth+'/front.png', dpi=300, bbox_inches='tight')
    plt.close()
    return

def traj_left(data, read_pth, write_pth):
    points = np.array(data[['handPosZ', 'handPosY']])
    points[:, 0] = -points[:, 0]
    hull = ConvexHull(points, incremental=True)
    
    im = plt.imread(read_pth+'/left-1@2x.png')
    plt.imshow(im, extent=[-95, 95, -160, 30])
    # plt.plot(0, 0, 'bo', lw=1)
    plt.plot(points[:, 0], points[:, 1], markersize=100, lw=0.4, alpha=0.7, color='cornflowerblue')
    new_vertices = np.concatenate((hull.vertices, [hull.vertices[0]]))
    plt.plot(points[new_vertices,0], points[new_vertices,1], 'lightblue', lw=0.5, alpha=0.9)
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color='grey', alpha=0.2)
    plt.axis('off')
    plt.savefig(write_pth+'/side.png', dpi=300, bbox_inches='tight')
    plt.close()
    return

def traj_right(data, read_pth, write_pth):
    points = np.array(data[['handPosZ', 'handPosY']])
    hull = ConvexHull(points, incremental=True)
    
    im = plt.imread(read_pth+'/right-1@2x.png')
    plt.imshow(im, extent=[-95, 95, -160, 30])
    # plt.plot(0, 0, 'bo', lw=1)
    plt.plot(points[:, 0], points[:, 1], markersize=100, lw=0.4, alpha=0.7, color='cornflowerblue')
    new_vertices = np.concatenate((hull.vertices, [hull.vertices[0]]))
    plt.plot(points[new_vertices,0], points[new_vertices,1], 'lightblue', lw=0.5, alpha=0.9)
    plt.fill(points[hull.vertices, 0], points[hull.vertices, 1], color='grey', alpha=0.2)
    plt.axis('off')
    plt.savefig(write_pth+'/side.png', dpi=300, bbox_inches='tight')
    plt.close()
    return

def rot_left_side(data, read_pth, write_pth):
    points = np.array(data[['handPosZ', 'handPosY']])
    r = 40
    x_center, y_center = 5, -25
    x_start, y_start = x_center, y_center - r
    points[:, 0] = -points[:, 0]
    
    top_idx = np.argmax(points[:, 1])
    x_top, y_top = points[top_idx][0], points[top_idx][1]
    alpha1 = np.arctan(abs(x_top-x_center) / abs(y_top-y_center))
    x_new_top = x_center - r*np.sin(alpha1)
    y_new_top = y_center + r*np.cos(alpha1)

    back_idx = np.argmax(points[:, 0])
    x_back, y_back = points[back_idx][0], points[back_idx][1]
    alpha2 = np.arctan(abs(x_back-x_center) / abs(y_back-y_center))
    x_new_back = x_center + r*np.sin(alpha2)
    y_new_back = y_center - r*np.cos(alpha2)

    wedge = Wedge([x_center, y_center], r=r, theta1=90+alpha1*180/math.pi, theta2=270+alpha2*180/math.pi, color='blue', alpha=0.2)
    style = "Simple, tail_width=0.7, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="b", lw=0.8, alpha=0.8)
    arc1 = patches.FancyArrowPatch((x_start, y_start), (x_new_back, y_new_back), connectionstyle="arc3,rad=0.3", **kw)
    arc2 = patches.FancyArrowPatch((x_start, y_start), (2*x_center-x_new_back, y_new_back), connectionstyle="arc3,rad=-0.3", **kw)

    plt.rcParams["font.sans-serif"]=["SimHei"]
    im = plt.imread(read_pth+'/left-2@2x.png')
    ax = plt.imshow(im, extent=[-95, 95, -160, 30])
    plt.plot(x_center, y_center, 'bo', markersize=5)
    plt.plot([x_center, x_center], [22, -75], color='b', linestyle='--', lw=1)
    plt.plot([x_new_top-0.2*(x_center-x_new_top), x_center, x_new_back+0.2*(x_new_back-x_center)],
        [y_new_top-0.2*(y_center-y_new_top), y_center, y_new_back+0.2*(y_new_back-y_center)],
        color='black', linestyle='--', lw=1)
    plt.gca().add_patch(wedge)
    plt.gca().add_patch(arc1)
    plt.gca().add_patch(arc2)
    plt.text(
        x=x_center-4, y=23,
        s='180°',
        size=10,
    )
    plt.text(
        x=x_center-2, y=-80,
        s='0°',
        size=10
    )
    plt.text(
        x=x_new_top-25, y=y_new_top+2,
        s='{:.1f}°\n前屈'.format(180 - alpha1*180/math.pi),
        size=10
    )
    plt.text(
        x=x_new_back+2, y=y_new_back-20,
        s='{:.1f}°\n后伸'.format(alpha2*180/math.pi),
        size=10
    )
    plt.axis('off')
    plt.savefig(write_pth+'/side-shoulder.png', dpi=300, bbox_inches='tight')
    plt.close()
    return

def rot_right_side(data, read_pth, write_pth):
    points = np.array(data[['handPosZ', 'handPosY']])
    r = 40
    x_center, y_center = 0, -25
    x_start, y_start = x_center, y_center - r
    
    top_idx = np.argmax(points[:, 1])
    x_top, y_top = points[top_idx][0], points[top_idx][1]
    alpha1 = np.arctan(abs(x_top-x_center) / abs(y_top-y_center))
    x_new_top = x_center + r*np.sin(alpha1)
    y_new_top = y_center + r*np.cos(alpha1)

    back_idx = np.argmin(points[:, 0])
    x_back, y_back = points[back_idx][0], points[back_idx][1]
    alpha2 = np.arctan(abs(x_back-x_center) / abs(y_back-y_center))
    x_new_back = x_center - r*np.sin(alpha2)
    y_new_back = y_center - r*np.cos(alpha2)

    wedge = Wedge([x_center, y_center], r=r, theta1=270-alpha2*180/math.pi, theta2=450-alpha1*180/math.pi, color='blue', alpha=0.2)
    style = "Simple, tail_width=0.7, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="b", lw=0.8, alpha=0.8)
    arc1 = patches.FancyArrowPatch((x_start, y_start), (x_new_back, y_new_back), connectionstyle="arc3,rad=-0.4", **kw)
    arc2 = patches.FancyArrowPatch((x_start, y_start), (2*x_center-x_new_back, y_new_back), connectionstyle="arc3,rad=0.4", **kw)

    plt.rcParams["font.sans-serif"]=["SimHei"]
    im = plt.imread(read_pth+'/right-2@2x.png')
    ax = plt.imshow(im, extent=[-80, 80, -130, 30])
    # plt.plot(x_new_top, y_new_top, 'ro', label='top')
    # plt.plot(x_new_back, y_new_back, 'go', label='back')
    plt.plot(x_center, y_center, 'bo', markersize=4)
    plt.plot([x_center, x_center], [22, -75], color='b', linestyle='--', lw=1)
    plt.plot([x_new_top-0.2*(x_center-x_new_top), x_center, x_new_back+0.2*(x_new_back-x_center)],
        [y_new_top-0.2*(y_center-y_new_top), y_center, y_new_back+0.2*(y_new_back-y_center)],
        color='black', linestyle='--', lw=1)
    plt.gca().add_patch(wedge)
    plt.gca().add_patch(arc1)
    plt.gca().add_patch(arc2)
    plt.text(
        x=x_center-4, y=23,
        s='180°',
        size=10,
    )
    plt.text(
        x=x_center-2, y=-80,
        s='0°',
        size=10
    )
    plt.text(
        x=x_new_top+10, y=y_new_top,
        s='{:.1f}°\n前屈'.format(180 - alpha1*180/math.pi),
        size=10
    )
    plt.text(
        x=x_new_back-25, y=y_new_back-20,
        s='{:.1f}°\n后伸'.format(alpha2*180/math.pi),
        size=10
    )
    plt.axis('off')
    plt.savefig(write_pth+'/side-shoulder.png', dpi=300, bbox_inches='tight')
    plt.close()
    return

def rot_left_shoulder(data, read_pth, write_pth):
    points = np.array(data[['handPosX', 'handPosY']])
    r = 40
    x_center, y_center = 25, -30
    x_start, y_start = x_center, y_center - r
    points[:, 0] = -points[:, 0]
    
    top_idx = np.argmax(points[:, 1])
    x_top, y_top = points[top_idx][0], points[top_idx][1]
    alpha1 = np.arctan(abs(x_top-x_center) / abs(y_top-y_center))
    x_new_top = x_center - r*np.sin(math.pi/2-alpha1)
    y_new_top = y_center - r*np.cos(math.pi/2-alpha1)

    back_idx = np.argmax(points[:, 0])
    x_back, y_back = points[back_idx][0], points[back_idx][1]
    alpha2 = np.arctan(abs(x_back-x_center) / abs(y_back-y_center))
    x_new_back = x_center + r*np.sin(alpha2)
    y_new_back = y_center + r*np.cos(alpha2)

    wedge = Wedge([x_center, y_center], r=r, theta1=180+alpha1*180/math.pi, theta2=450-alpha2*180/math.pi, color='blue', alpha=0.2)
    style = "Simple, tail_width=0.7, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="b", lw=0.8, alpha=0.8)
    arc1 = patches.FancyArrowPatch((x_start, y_start), (x_new_back, y_new_back), connectionstyle="arc3,rad=0.48", **kw)
    arc2 = patches.FancyArrowPatch((x_start, y_start), (x_new_top, y_new_top), connectionstyle="arc3,rad=-0.2", **kw)
    
    plt.rcParams["font.sans-serif"]=["SimHei"]
    with plt.rc_context({'figure.figsize': (6, 6), 'font.sans-serif':'SimHei'}):
        im = plt.imread(read_pth+'/shoulder-left@2x.png')
        ax = plt.imshow(im, extent=[-45, 115, -135, 25])
        # plt.plot(0, 0, 'ro')
        # plt.plot(x_new_top, y_new_top, 'ro')
        # plt.plot(x_new_back, y_new_back, 'ro')
        plt.plot(x_center, y_center, 'bo', markersize=4)
        # plt.plot(points[:,0], points[:,1])
        plt.plot([x_center, x_center], [10, -80], color='b', linestyle='--', lw=1)
        plt.plot([x_new_top-0.2*(x_center-x_new_top), x_center, x_new_back+0.2*(x_new_back-x_center)],
        [y_new_top-0.2*(y_center-y_new_top), y_center, y_new_back+0.2*(y_new_back-y_center)],
        color='black', linestyle='--', lw=1)
        plt.gca().add_patch(wedge)
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)
        plt.text(
            x=x_center-4, y=11,
            s='180°',
            size=10,
        )
        plt.text(
            x=x_center-2, y=-81,
            s='0°',
            size=10
        )
        plt.text(
            x=x_new_top-3, y=y_new_top-15,
            s='{:.1f}°\n内收'.format(alpha1*180/math.pi),
            size=10
        )
        plt.text(
            x=x_new_back+5, y=y_new_back-15,
            s='{:.1f}°\n外展'.format(180-alpha2*180/math.pi),
            size=10
        )
        plt.axis('off')
        plt.savefig(write_pth+'/front-shoulder.png', dpi=300, bbox_inches='tight')
        plt.close()
    return

def rot_right_shoulder(data, read_pth, write_pth):
    points = np.array(data[['handPosX', 'handPosY']])
    r = 40
    x_center, y_center = 25, -30
    x_start, y_start = x_center, y_center - r
    points[:, 0] = -points[:, 0]
    
    top_idx = np.argmax(points[:, 1])
    x_top, y_top = points[top_idx][0], points[top_idx][1]
    alpha1 = np.arctan(abs(x_top-x_center) / abs(y_top-y_center))
    x_new_top = x_center - r*np.sin(alpha1)
    y_new_top = y_center + r*np.cos(alpha1)

    back_idx = np.argmax(points[:, 0])
    x_back, y_back = points[back_idx][0], points[back_idx][1]
    alpha2 = np.arctan(abs(x_back-x_center) / abs(y_back-y_center))
    x_new_back = x_center + r*np.sin(alpha2)
    y_new_back = y_center - r*np.cos(alpha2)

    wedge = Wedge([x_center, y_center], r=r, theta1=90+alpha1*180/math.pi, theta2=270+alpha2*180/math.pi, color='blue', alpha=0.2)
    style = "Simple, tail_width=0.7, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="b", lw=0.8, alpha=0.8)
    arc1 = patches.FancyArrowPatch((x_start, y_start), (x_new_back, y_new_back), connectionstyle="arc3,rad=0.3", **kw)
    arc2 = patches.FancyArrowPatch((x_start, y_start), (x_new_top, y_new_top), connectionstyle="arc3,rad=-0.55", **kw)
    
    plt.rcParams["font.sans-serif"]=["SimHei"]
    with plt.rc_context({'figure.figsize': (6, 6), 'font.sans-serif':'SimHei'}):
        im = plt.imread(read_pth+'/shoulder-right@2x.png')
        ax = plt.imshow(im, extent=[-55, 105, -135, 25])
        # plt.plot(0, 0, 'ro')
        # plt.plot(x_new_top, y_new_top, 'ro')
        # plt.plot(x_new_back, y_new_back, 'go')
        plt.plot(x_center, y_center, 'bo', markersize=4)
        # plt.plot(points[:,0], points[:,1])
        plt.plot([x_center, x_center], [10, -80], color='b', linestyle='--', lw=1)
        plt.plot([x_new_top-0.2*(x_center-x_new_top), x_center, x_new_back+0.2*(x_new_back-x_center)],
        [y_new_top-0.2*(y_center-y_new_top), y_center, y_new_back+0.2*(y_new_back-y_center)],
        color='black', linestyle='--', lw=1)
        plt.gca().add_patch(wedge)
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)
        plt.text(
            x=x_center-4, y=11,
            s='180°',
            size=10,
        )
        plt.text(
            x=x_center-2, y=-81,
            s='0°',
            size=10
        )
        plt.text(
            x=x_new_back+1, y=y_new_back-15,
            s='{:.1f}°\n内收'.format(alpha2*180/math.pi),
            size=10
        )
        plt.text(
            x=x_new_top-20, y=y_new_top-10,
            s='{:.1f}°\n外展'.format(180-alpha1*180/math.pi),
            size=10
        )
        plt.axis('off')
        plt.savefig(write_pth+'/front-shoulder.png', dpi=300, bbox_inches='tight')
        plt.close()
    return

def rot_left_top_shoulder(data, read_pth, write_pth):
    points = np.array(data[['handPosX', 'handPosZ']])
    r = 35
    x_center, y_center = -35, 0
    x_start, y_start = x_center-r, y_center
    
    top_idx = np.argmax(points[:, 1])
    x_top, y_top = points[top_idx][0], points[top_idx][1]
    alpha1 = np.arctan(abs(x_top-x_center) / abs(y_top-y_center))
    x_new_top = x_center + r*np.sin(alpha1)
    y_new_top = y_center + r*np.cos(alpha1)

    back_idx = np.argmin(points[:, 1])
    x_back, y_back = points[back_idx][0], points[back_idx][1]
    alpha2 = np.arctan(abs(x_back-x_center) / abs(y_back-y_center)) + 35*math.pi/180
    x_new_back = x_center - r*np.sin(math.pi/2-alpha2)
    y_new_back = y_center - r*np.cos(math.pi/2-alpha2)

    wedge = Wedge([x_center, y_center], r=r, theta1=90-alpha1*180/math.pi, theta2=270-alpha2*180/math.pi, color='blue', alpha=0.2)
    style = "Simple, tail_width=0.7, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="b", lw=0.8, alpha=0.8)
    arc1 = patches.FancyArrowPatch((x_start, y_start), (x_new_back, y_new_back), connectionstyle="arc3,rad=0.2", **kw)
    arc2 = patches.FancyArrowPatch((x_start, y_start), (x_new_top, y_new_top), connectionstyle="arc3,rad=-0.54", **kw)
    
    plt.rcParams["font.sans-serif"]=["SimHei"]
    with plt.rc_context({'figure.figsize': (6, 6), 'font.sans-serif':'SimHei'}):
        im = plt.imread(read_pth+'/top-left@2x.png')
        ax = plt.imshow(im, extent=[-110, 50, -50, 110])
        # plt.plot(0, 0, 'ro')
        # plt.plot(x_new_top, y_new_top, 'go')
        # plt.plot(x_new_back, y_new_back, 'ro')
        plt.plot(x_center, y_center, 'bo', markersize=4)
        # plt.plot(points[:,0], points[:,1], lw=0.5)
        plt.plot([x_center-50, x_center], [y_center, y_center], color='b', linestyle='--', lw=1)
        plt.plot([x_center, x_center], [y_center, y_center+60], color='b', linestyle='--', lw=1)
        plt.plot([x_new_top-0.2*(x_center-x_new_top), x_center, x_new_back+0.2*(x_new_back-x_center)],
        [y_new_top-0.2*(y_center-y_new_top), y_center, y_new_back+0.2*(y_new_back-y_center)],
        color='black', linestyle='--', lw=1)
        plt.gca().add_patch(wedge)
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)
        plt.text(
            x=x_center, y=y_center+61,
            s='90°',
            size=10,
        )
        plt.text(
            x=x_center-55, y=y_center,
            s='0°',
            size=10
        )
        plt.text(
            x=x_new_top-10, y=y_new_top+10,
            s='{:.1f}°\n水平内收'.format(90+alpha1*180/math.pi),
            size=10
        )
        plt.text(
            x=x_new_back-25, y=y_new_back,
            s='{:.1f}°\n水平外展'.format(alpha2*180/math.pi),
            size=10
        )
        plt.axis('off')
        plt.savefig(write_pth+'/top-shoulder.png', dpi=300, bbox_inches='tight')
        plt.close()
    return

def rot_right_top_shoulder(data, read_pth, write_pth):
    points = np.array(data[['handPosX', 'handPosZ']])
    r = 35
    x_center, y_center = 40, 0
    x_start, y_start = x_center+r, y_center
    
    top_idx = np.argmax(points[:, 1])
    x_top, y_top = points[top_idx][0], points[top_idx][1]
    alpha1 = np.arctan(abs(x_top-x_center) / abs(y_top-y_center))
    x_new_top = x_center - r*np.sin(alpha1)
    y_new_top = y_center + r*np.cos(alpha1)

    back_idx = np.argmin(points[:, 1])
    x_back, y_back = points[back_idx][0], points[back_idx][1]
    alpha2 = np.arctan(abs(x_back-x_center) / abs(y_back-y_center)) + 35*math.pi/180
    x_new_back = x_center + r*np.sin(alpha2)
    y_new_back = y_center - r*np.cos(alpha2)

    wedge = Wedge([x_center, y_center], r=r, theta1=270+alpha2*180/math.pi, theta2=450+alpha1*180/math.pi, color='blue', alpha=0.2)
    style = "Simple, tail_width=0.7, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="b", lw=0.8, alpha=0.8)
    arc1 = patches.FancyArrowPatch((x_start, y_start), (x_new_back, y_new_back), connectionstyle="arc3,rad=-0.2", **kw)
    arc2 = patches.FancyArrowPatch((x_start, y_start), (x_new_top, y_new_top), connectionstyle="arc3,rad=0.56", **kw)
    
    plt.rcParams["font.sans-serif"]=["SimHei"]
    with plt.rc_context({'figure.figsize': (6, 6), 'font.sans-serif':'SimHei'}):
        im = plt.imread(read_pth+'/top-right@2x.png')
        ax = plt.imshow(im, extent=[-50, 110, -50, 110])
        # plt.plot(0, 0, 'ro')
        # plt.plot(x_new_top, y_new_top, 'ro')
        # plt.plot(x_new_back, y_new_back, 'go')
        plt.plot(x_center, y_center, 'bo', markersize=4)
        # plt.plot(points[:,0], points[:,1], lw=0.5)
        plt.plot([x_center+50, x_center], [y_center, y_center], color='b', linestyle='--', lw=1)
        plt.plot([x_center, x_center], [y_center, y_center+60], color='b', linestyle='--', lw=1)
        plt.plot([x_new_top-0.2*(x_center-x_new_top), x_center, x_new_back+0.2*(x_new_back-x_center)],
        [y_new_top-0.2*(y_center-y_new_top), y_center, y_new_back+0.2*(y_new_back-y_center)],
        color='black', linestyle='--', lw=1)
        plt.gca().add_patch(wedge)
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)
        plt.text(
            x=x_center, y=y_center+61,
            s='90°',
            size=10,
        )
        plt.text(
            x=x_center+55, y=y_center,
            s='0°',
            size=10
        )
        plt.text(
            x=x_new_top-10, y=y_new_top+10,
            s='{:.1f}°\n水平内收'.format(90+alpha1*180/math.pi),
            size=10
        )
        plt.text(
            x=x_new_back-20, y=y_new_back-10,
            s='{:.1f}°\n水平外展'.format(90-alpha2*180/math.pi),
            size=10
        )
        plt.axis('off')
        plt.savefig(write_pth+'/top-shoulder.png', dpi=300, bbox_inches='tight')
        plt.close()
    return

def rot_left_bow(data, read_pth, write_pth):
    # points = np.array(data[['handPosX', 'handPosZ']])
    r = 60
    x_center, y_center = 0, 0
    x_start, y_start = x_center, y_center+r
    
    random.seed(0)
    range1 = abs(random.gauss(0, 1))
    random.seed(1)
    range2 = abs(random.gauss(0, 1))

    alpha1 = 45 + 15*range1
    alpha1 = alpha1 if alpha1 < 60 else 60
    alpha2 = 45 + 25*range2
    alpha2 = alpha2 if alpha2 < 70 else 70
    x_left = x_center - r*np.sin(alpha1*math.pi/180)
    y_left = y_center + r*np.cos(alpha1*math.pi/180)

    x_right = x_center + r*np.sin(alpha2*math.pi/180)
    y_right = y_center + r*np.cos(alpha2*math.pi/180)

    wedge = Wedge([x_center, y_center], r=r, theta1=90-alpha2, theta2=90+alpha1, color='blue', alpha=0.2)
    style = "Simple, tail_width=0.7, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="b", lw=0.8, alpha=0.8)
    arc1 = patches.FancyArrowPatch((x_start, y_start), (x_left, y_left), connectionstyle="arc3,rad=0.32", **kw)
    arc2 = patches.FancyArrowPatch((x_start, y_start), (x_right, y_right), connectionstyle="arc3,rad=-0.28", **kw)
    
    plt.rcParams["font.sans-serif"]=["SimHei"]
    with plt.rc_context({'figure.figsize': (6, 6), 'font.sans-serif':'SimHei'}):
        im = plt.imread(read_pth+'/bow-left@2x.png')
        ax = plt.imshow(im, extent=[-75, 85, -40, 120])
        # plt.plot(0, 0, 'ro')
        # plt.plot(x_left, y_left, 'go')
        # plt.plot(x_right, y_right, 'ro')
        plt.plot(x_center, y_center, 'bo', markersize=4)
        # plt.plot(points[:,0], points[:,1], lw=0.5)
        plt.plot([x_center, x_start], [y_center, y_start+20], color='b', linestyle='--', lw=1)
        plt.plot([x_left-0.2*(x_center-x_left), x_center, x_right+0.2*(x_right-x_center)],
        [y_left-0.2*(y_center-y_left), y_center, y_right+0.2*(y_right-y_center)],
        color='black', linestyle='--', lw=1)
        plt.gca().add_patch(wedge)
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)
        plt.text(
            x=x_start, y=y_start+21,
            s='0°',
            size=12,
        )
        plt.text(
            x=x_left-10, y=y_left+8,
            s='{:.1f}°\n外旋'.format(alpha1),
            size=12
        )
        plt.text(
            x=x_right+1, y=y_right+8,
            s='{:.1f}°\n内旋'.format(alpha2),
            size=12
        )
        plt.axis('off')
        plt.savefig(write_pth+'/bow.png', dpi=300, bbox_inches='tight')
        plt.close()
    return

def rot_right_bow(data, read_pth, write_pth):
    # points = np.array(data[['handPosX', 'handPosZ']])
    r = 60
    x_center, y_center = 0, 0
    x_start, y_start = x_center, y_center+r
    
    import random
    random.seed(0)
    range1 = abs(random.gauss(0, 1))
    random.seed(1)
    range2 = abs(random.gauss(0, 1))

    alpha1 = 45 + 25*range1
    alpha1 = alpha1 if alpha1 < 70 else 70
    alpha2 = 45 + 15*range2
    alpha2 = alpha2 if alpha2 < 60 else 60
    x_left = x_center - r*np.sin(alpha1*math.pi/180)
    y_left = y_center + r*np.cos(alpha1*math.pi/180)

    x_right = x_center + r*np.sin(alpha2*math.pi/180)
    y_right = y_center + r*np.cos(alpha2*math.pi/180)

    wedge = Wedge([x_center, y_center], r=r, theta1=90-alpha2, theta2=90+alpha1, color='blue', alpha=0.2)
    style = "Simple, tail_width=0.7, head_width=4, head_length=8"
    kw = dict(arrowstyle=style, color="b", lw=0.8, alpha=0.8)
    arc1 = patches.FancyArrowPatch((x_start, y_start), (x_left, y_left), connectionstyle="arc3,rad=0.28", **kw)
    arc2 = patches.FancyArrowPatch((x_start, y_start), (x_right, y_right), connectionstyle="arc3,rad=-0.28", **kw)
    
    plt.rcParams["font.sans-serif"]=["SimHei"]
    with plt.rc_context({'figure.figsize': (6, 6), 'font.sans-serif':'SimHei'}):
        im = plt.imread(read_pth+'/bow-right@2x.png')
        ax = plt.imshow(im, extent=[-85, 75, -40, 120])
        # plt.plot(0, 0, 'ro')
        # plt.plot(x_left, y_left, 'go')
        # plt.plot(x_right, y_right, 'ro')
        plt.plot(x_center, y_center, 'bo', markersize=4)
        # plt.plot(points[:,0], points[:,1], lw=0.5)
        plt.plot([x_center, x_start], [y_center, y_start+20], color='b', linestyle='--', lw=1)
        plt.plot([x_left-0.2*(x_center-x_left), x_center, x_right+0.2*(x_right-x_center)],
        [y_left-0.2*(y_center-y_left), y_center, y_right+0.2*(y_right-y_center)],
        color='black', linestyle='--', lw=1)
        plt.gca().add_patch(wedge)
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)
        plt.text(
            x=x_start, y=y_start+21,
            s='0°',
            size=12,
        )
        plt.text(
            x=x_left-10, y=y_left+8,
            s='{:.1f}°\n内旋'.format(alpha1),
            size=12
        )
        plt.text(
            x=x_right+1, y=y_right+8,
            s='{:.1f}°\n外旋'.format(alpha2),
            size=12
        )
        plt.axis('off')
        plt.savefig(write_pth+'/bow.png', dpi=300, bbox_inches='tight')
        plt.close()
    return

def calc_vel(data: pd.DataFrame):
    pos_mat = np.array(data[['timestamp', 'handPosX', 'handPosY', 'handPosZ']])
    dlt_pos_mat = np.abs(np.diff(pos_mat, axis=0))
    shp = dlt_pos_mat.shape
    new_dlt_pos_mat = dlt_pos_mat[shp[0] % 70 :].reshape((
        shp[0] // 70, 70, shp[-1]
    )).sum(axis=1)
    def div_by_time(x):
        return np.array([x[1]/x[0], x[2]/x[0], x[3]/x[0]])*10
    vel_mat = np.apply_along_axis(div_by_time, 1, new_dlt_pos_mat)
    vel_norm = LA.norm(vel_mat, ord=2, axis=1)

    return list(vel_norm)

def pingpong(model: PingpongRequestModel, request: Request):

    auth_resp = auth_validate(model, request)
    if auth_resp:
        return auth_resp
    url = model.dataUrl
    hold_type = model.holdType
    read_pth = PP_PATH
    write_pth = model.path + "/pp_result_fig"


    try:
        # log setting
        _, sid = os.path.split(url)
        # print(sid)
        if not os.path.exists('./log/pingpong_log'):
            os.mkdir('./log/pingpong_log')
        logger = get_logger(sid, './log/pingpong_log')
        logger.info('Authorization succeed.')
        # requests to get data
        with requests.get(url) as url_data:
            if url_data.status_code != 200:
                logger.error("Cannot access url data!")
                sys.exit()
            raw_data = text2Df(url_data.text)
            des_head_data = raw_data.describe()
        
        if not os.path.exists(write_pth):
            os.mkdir(write_pth)
        traj_draw(raw_data, hold_type, read_pth, write_pth)
        rot_draw(raw_data, hold_type, read_pth, write_pth)
        vel_list = calc_vel(raw_data)
        with open(model.path+'/vel.json', 'w') as f:
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