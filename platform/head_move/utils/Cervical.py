import io, math, datetime
import os
from fastapi import Request

import pandas as pd
from pydantic import BaseModel
from config.settings import settings
from qcloud_cos import CosConfig, CosS3Client
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Wedge

from utils.auth import auth_validate
from utils.logger import get_logger
from utils.response_template import GeneralResponseModel

class CervicalReuqestModel(BaseModel):
    id: int
    url: str

    
def get_url_data(url):
    try:
        df = pd.read_csv(url)
        df['timestamp'] = df['timestamp'] - df['timestamp'].min()
        df['pitch'] = df['pitch'].apply(lambda x: x-360 if x > 270 else x)
        df['yaw'] = df['yaw'].apply(lambda x: x-360 if x > 270 else x)
        df['roll'] = df['roll'].apply(lambda x: x-360 if x > 270 else x)
        return df
    except Exception as e:
        raise e

def get_magnitude(df) -> dict:
    try:
        df_action_1 = df.loc[(df['level'] == 1) | (df['level'] == 2)].describe()
        ang_1 = int(df_action_1.loc['max', 'pitch'])
        df_action_2 = df.loc[(df['level'] == 3) | (df['level'] == 4)].describe()
        ang_2 = int(abs(df_action_2.loc['min', 'pitch']))
        df_action_3 = df.loc[(df['level'] == 5) | (df['level'] == 6)].describe()
        ang_3 = int(df_action_3.loc['max', 'roll'])
        df_action_4 = df.loc[(df['level'] == 7) | (df['level'] == 8)].describe()
        ang_4 = int(abs(df_action_4.loc['min', 'roll']))
        df_action_5 = df.loc[(df['level'] == 9) | (df['level'] == 10)].describe()
        ang_5 = int(abs(df_action_5.loc['min', 'yaw']))
        df_action_6 = df.loc[(df['level'] == 11) | (df['level'] == 12)].describe()
        ang_6 = int(df_action_6.loc['max', 'yaw'])

        angles = [ang_1, ang_2, ang_3, ang_4, ang_5, ang_6]
        tags = ['forward', 'back', 'left', 'right', 'leftRot', 'rightRot']

        return dict(zip(tags, angles))
    except Exception as e:
        raise e

def get_vel_ang(id, df):
    buffer_pos = []
    bio = io.BytesIO()

    if settings.deploy_mode == 'offline':
        now = datetime.datetime.now()
        year, month, day = str(now.year), str(now.month), str(now.day)
        save_key = '/'.join([settings.url_prefix_offline, settings.sd_prefix_offline, year, month, day, str(id)])
        base_key = '/'.join([settings.sd_prefix_offline, year, month, day, str(id)])

    def vel(row):
        dist = math.sqrt(row['x'] ** 2 + row['y'] ** 2 + row['z'] ** 2)
        vel = dist / row['timestamp']
        return vel
    # 1
    df_action = df.loc[(df['level'] == 1) | (df['level'] == 2)]
    df_action['timestamp'] = (df_action['timestamp'] - df_action['timestamp'].min())/1000
    diff = df_action[['timestamp', 'x', 'y', 'z']].diff()
    df_action['vel'] = diff.apply(lambda x: vel(x), axis=1)
    df_action['vel'] = df_action['vel'].rolling(35, center=True).mean()
    df_action['vel'].fillna(0, inplace=True)

    if settings.deploy_mode == 'offline':
        df_action[['timestamp', 'vel']].to_json(save_key + '/1.json', orient='values')
        df_action[['timestamp', 'pitch']].to_json(save_key + '/2.json', orient='values')
        json_keys['forward'] = {'vel': base_key + '/1.json', 'angle': base_key + '/2.json'}
    else:
        df_action[['timestamp', 'vel']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        df_action[['timestamp', 'pitch']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
    # 2
    df_action = df.loc[(df['level'] == 3) | (df['level'] == 4)]
    df_action['pitch'] = -df_action['pitch']
    df_action['timestamp'] = (df_action['timestamp'] - df_action['timestamp'].min())/1000
    diff = df_action[['timestamp', 'x', 'y', 'z']].diff()
    df_action['vel'] = diff.apply(lambda x: vel(x), axis=1)
    df_action['vel'] = df_action['vel'].rolling(35, center=True).mean()
    df_action['vel'].fillna(0, inplace=True)

    if settings.deploy_mode == 'offline':
        df_action[['timestamp', 'vel']].to_json(save_key + '/3.json', orient='values')
        df_action[['timestamp', 'pitch']].to_json(save_key + '/4.json', orient='values')
        json_keys['back'] = {'vel': base_key + '/3.json', 'angle': base_key + '/4.json'}
    else:
        df_action[['timestamp', 'vel']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        df_action[['timestamp', 'pitch']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
    # 3
    df_action = df.loc[(df['level'] == 5) | (df['level'] == 6)]
    df_action['timestamp'] = (df_action['timestamp'] - df_action['timestamp'].min())/1000
    diff = df_action[['timestamp', 'x', 'y', 'z']].diff()
    df_action['vel'] = diff.apply(lambda x: vel(x), axis=1)
    df_action['vel'] = df_action['vel'].rolling(35, center=True).mean()
    df_action['vel'].fillna(0, inplace=True)


    if settings.deploy_mode == 'offline':
        df_action[['timestamp', 'vel']].to_json(save_key + '/5.json', orient='values')
        df_action[['timestamp', 'roll']].to_json(save_key + '/6.json', orient='values')
        json_keys['left'] = {'vel': base_key + '/5.json', 'angle': base_key + '/6.json'}
    else:
        df_action[['timestamp', 'vel']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        df_action[['timestamp', 'roll']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
    # 4
    df_action = df.loc[(df['level'] == 7) | (df['level'] == 8)]
    df_action['roll'] = -df_action['roll']
    df_action['timestamp'] = (df_action['timestamp'] - df_action['timestamp'].min())/1000
    diff = df_action[['timestamp', 'x', 'y', 'z']].diff()
    df_action['vel'] = diff.apply(lambda x: vel(x), axis=1)
    df_action['vel'] = df_action['vel'].rolling(35, center=True).mean()
    df_action['vel'].fillna(0, inplace=True)

    if settings.deploy_mode == 'offline':
        df_action[['timestamp', 'vel']].to_json(save_key + '/7.json', orient='values')
        df_action[['timestamp', 'roll']].to_json(save_key + '/8.json', orient='values')
        json_keys['right'] = {'vel': base_key + '/7.json', 'angle': base_key + '/8.json'}
    else:
        df_action[['timestamp', 'vel']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        df_action[['timestamp', 'roll']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
    # 5
    df_action = df.loc[(df['level'] == 9) | (df['level'] == 10)]
    df_action['yaw'] = -df_action['yaw']
    df_action['timestamp'] = (df_action['timestamp'] - df_action['timestamp'].min())/1000
    diff = df_action[['timestamp', 'x', 'y', 'z']].diff()
    df_action['vel'] = diff.apply(lambda x: vel(x), axis=1)
    df_action['vel'] = df_action['vel'].rolling(35, center=True).mean()
    df_action['vel'].fillna(0, inplace=True)

    if settings.deploy_mode == 'offline':
        df_action[['timestamp', 'vel']].to_json(save_key + '/9.json', orient='values')
        df_action[['timestamp', 'yaw']].to_json(save_key + '/10.json', orient='values')
        json_keys['leftRot'] = {'vel': base_key + '/9.json', 'angle': base_key + '/10.json'}
    else:
        df_action[['timestamp', 'vel']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        df_action[['timestamp', 'yaw']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
    # 6
    df_action = df.loc[(df['level'] == 11) | (df['level'] == 12)]
    df_action['timestamp'] = (df_action['timestamp'] - df_action['timestamp'].min())/1000
    diff = df_action[['timestamp', 'x', 'y', 'z']].diff()
    df_action['vel'] = diff.apply(lambda x: vel(x), axis=1)
    df_action['vel'] = df_action['vel'].rolling(35, center=True).mean()
    df_action['vel'].fillna(0, inplace=True)

    if settings.deploy_mode == 'offline':
        df_action[['timestamp', 'vel']].to_json(save_key + '/11.json', orient='values')
        df_action[['timestamp', 'yaw']].to_json(save_key + '/12.json', orient='values')
        json_keys['rightRot'] = {'vel': base_key + '/11.json', 'angle': base_key + '/12.json'}
    else:
        df_action[['timestamp', 'vel']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        df_action[['timestamp', 'yaw']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())

    if settings.deploy_mode == 'offline':
        return json_keys
    else:
        json_keys = save_data2cos(id, bio, buffer_pos)
        return json_keys


def draw(id, df):
    try:
        config = CosConfig(Region=settings.region, SecretId=settings.secret_id, SecretKey=settings.secret_key, Token=None, Scheme="https")
        client = CosS3Client(config)
        now = datetime.datetime.now()
        year, month, day = str(now.year), str(now.month), str(now.day)
        base_key = '/'.join([settings.sd_prefix, year, month, day, str(id)])
        save_key = '/'.join([settings.url_prefix_offline, settings.sd_prefix_offline, year, month, day, str(id)])

        if settings.deploy_mode == 'offline':
            draw1(save_key)
            key1 = base_key + '/1.jpg'
            draw2(save_key)
            key2 = base_key + '/2.jpg'
            draw3(save_key)
            key3 = base_key + '/3.jpg'
        else:
            bio1 = draw1(df)
            key1 = settings.url_prefix + _save_img2cos(bio1, client, base_key, 1)
            bio2 = draw2(df)
            key2 = settings.url_prefix + _save_img2cos(bio2, client, base_key, 2)
            bio3 = draw3(df)
            key3 = settings.url_prefix + _save_img2cos(bio3, client, base_key, 3)

        tags = ['side', 'front', 'up']
        keys = [key1, key2, key3]
        return dict(zip(tags, keys))
    except Exception as e:
        raise e


def save_data2cos(id, bio, buffer_pos=None):

    try:
        config = CosConfig(Region=settings.region, SecretId=settings.secret_id, SecretKey=settings.secret_key, Token=None, Scheme="https")
        client = CosS3Client(config)
        now = datetime.datetime.now()
        year, month, day = str(now.year), str(now.month), str(now.day)
        base_key = '/'.join([settings.sd_prefix, year, month, day, str(id)])

        keys = _save_data2cos(bio, client, base_key, buffer_pos)
        return keys
    except Exception as e:
        raise e

def _save_data2cos(bio, client, base_key, buffer_pos):
    tmp_keys = []
    result_keys = []
    try:
        for i in range((len(buffer_pos))):
            if i == 0:
                bio.seek(0)
                key = base_key + '/{}.json'.format(i+1)
                client.put_object(
                    Bucket=settings.bucket_name,
                    Body=bio.read1(buffer_pos[i]),
                    Key=key,
                )
            else:
                bio.seek(buffer_pos[i-1])
                key = base_key + '/{}.json'.format(i+1)
                client.put_object(
                    Bucket=settings.bucket_name,
                    Body=bio.read1(buffer_pos[i] - buffer_pos[i-1]),
                    Key=key,
                )
            
            tmp_keys.append(settings.url_prefix + key)
            if len(tmp_keys) == 2:
                result_keys.append(dict(zip(['vel', 'angle'], tmp_keys)))
                tmp_keys = []

        tags = ['forward', 'back', 'left', 'right', 'leftRot', 'rightRot']
        return dict(zip(tags, result_keys))
    except Exception as e:
        raise e

def  _save_img2cos(bio, client, base_key, ord):
    try:
        key = base_key + '/{}.jpg'.format(ord)
        bio.seek(0)
        client.put_object(
            Bucket=settings.bucket_name,
            Body=bio.read1(),
            Key=key
        )
        return key
    except Exception as e:
        raise e

def draw1(df, save_path=None):
    try:
        im = plt.imread('./assets/src_fig/head-2@2x.png')
        extent=[-30, 30, -47.5, 12.5]
        plt.imshow(im, extent=extent)
        x_center, y_center = 0, -15
        r=18
        x_offset, y_offset = 8, 2

        df_head = df.loc[df['level'].isin([1,2,3,4])]

        alpha_left = abs(df_head.describe().loc['min', 'pitch']) / 180 * math.pi
        alpha_right = abs(df_head.describe().loc['max', 'pitch']) / 180 * math.pi

        x_start, y_start = x_center, y_center+r

        _x_left, _y_left = x_center - r*math.sin(alpha_left), y_center + r*math.cos(alpha_left)
        _x_right, _y_right = x_center + r*math.sin(alpha_right), y_center + r*math.cos(alpha_right)

        wedge = Wedge([x_center, y_center], r=r, theta1=90-alpha_right*180/math.pi, theta2=90+alpha_left*180/math.pi, color='blue', alpha=0.2)
        style = "Simple, tail_width=0.7, head_width=2, head_length=3"
        kw = dict(arrowstyle=style, color="b", lw=0.5, alpha=0.8)
        arc1 = patches.FancyArrowPatch((x_start, y_start), (_x_left, _y_left), connectionstyle="arc3,rad=0.25", **kw)
        arc2 = patches.FancyArrowPatch((x_start, y_start), (_x_right, _y_right), connectionstyle="arc3,rad=-0.25", **kw)
        plt.gca().add_patch(wedge)
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)

        plt.text(
            x=_x_left-x_offset, y=_y_left,
            s='{:.1f}°'.format(180*alpha_left/math.pi),
            size=9
        )
        plt.text(
            x=_x_right+2, y=_y_right-y_offset,
            s='{:.1f}°'.format(180*alpha_right/math.pi),
            size=9,
        )
        plt.axis('off')
        plt.xlim(extent[0:2]); plt.ylim(extent[2:])

        if settings.deploy_mode == 'offline':
            plt.savefig(save_path + '/1.jpg', format='jpg', dpi=200, bbox_inches='tight')
            plt.close()
            return
        else:
            bio = io.BytesIO()
            plt.savefig(bio, format='jpg', dpi=200, bbox_inches='tight')
            plt.close()
            return bio
    except Exception as e:
        raise e

def draw2(df, save_path=None):
    try:
        im = plt.imread('./assets/src_fig/head-1@2x.png')
        extent=[-30, 30, -47.5, 12.5]
        plt.imshow(im, extent=extent)
        x_center, y_center = 0, -15
        r=18
        x_offset, y_offset = 8, 2

        df_head = df.loc[df['level'].isin([5,6,7,8])]


        alpha_left = abs(df_head.describe().loc['min', 'roll']) / 180 * math.pi
        alpha_right = abs(df_head.describe().loc['max', 'roll']) / 180 * math.pi

        x_start, y_start = x_center, y_center+r
        _x_left, _y_left = x_center - r*math.sin(alpha_left), y_center + r*math.cos(alpha_left)
        _x_right, _y_right = x_center + r*math.sin(alpha_right), y_center + r*math.cos(alpha_right)

        wedge = Wedge([x_center, y_center], r=r, theta1=90-alpha_right*180/math.pi, theta2=90+alpha_left*180/math.pi, color='blue', alpha=0.2)
        style = "Simple, tail_width=0.7, head_width=2, head_length=3"
        kw = dict(arrowstyle=style, color="b", lw=0.5, alpha=0.8)
        arc1 = patches.FancyArrowPatch((x_start, y_start), (_x_left, _y_left), connectionstyle="arc3,rad=0.23", **kw)
        arc2 = patches.FancyArrowPatch((x_start, y_start), (_x_right, _y_right), connectionstyle="arc3,rad=-0.23", **kw)
        plt.gca().add_patch(wedge)
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)

        plt.text(
            x=_x_left-x_offset, y=_y_left,
            s='{:.1f}°'.format(180*alpha_left/math.pi),
            size=9
        )
        plt.text(
            x=_x_right+2, y=_y_right-y_offset,
            s='{:.1f}°'.format(180*alpha_right/math.pi),
            size=9,
        )
        plt.axis('off')
        plt.xlim(extent[0:2]); plt.ylim(extent[2:])

        if settings.deploy_mode == 'offline':
            plt.savefig(save_path + '/2.jpg', format='jpg', dpi=200, bbox_inches='tight')
            plt.close()
            return 
        else:
            bio = io.BytesIO()
            plt.savefig(bio, format='jpg', dpi=200, bbox_inches='tight')
            plt.close()
            return bio
    except Exception as e:
        raise e

def draw3(df, save_path=None):
    try:
        im = plt.imread('./assets/src_fig/head-3@2x.png')
        extent = [-75, 75, -81, 69]
        plt.imshow(im, extent=extent)
        x_center, y_center = 0, 0
        r=40
        x_offset, y_offset = 8, 2

        df_head = df.loc[df['level'].isin([9,10,11,12])]

        alpha_left = abs(df_head.describe().loc['min', 'yaw']) / 180 * math.pi
        left_ang = abs(df_head.describe().loc['min', 'yaw'])

        alpha_right = abs(df_head.describe().loc['max', 'yaw']) / 180 * math.pi
        right_ang = abs(df_head.describe().loc['max', 'yaw'])

        x_start, y_start = x_center+r, y_center
        _x_left, _y_left = x_center + r*math.cos(alpha_left), y_center + r*math.sin(alpha_left)
        _x_right, _y_right = x_center + r*math.cos(alpha_right), y_center - r*math.sin(alpha_right)

        wedge = Wedge([x_center, y_center], r=r, theta1=-right_ang, theta2=left_ang, color='blue', alpha=0.2)
        style = "Simple, tail_width=0.7, head_width=2, head_length=3"
        kw = dict(arrowstyle=style, color="b", lw=0.5, alpha=0.8)
        arc1 = patches.FancyArrowPatch((x_start, y_start), (_x_left, _y_left), connectionstyle="arc3,rad=0.38", **kw)
        arc2 = patches.FancyArrowPatch((x_start, y_start), (_x_right, _y_right), connectionstyle="arc3,rad=-0.38", **kw)
        plt.gca().add_patch(wedge)
        plt.gca().add_patch(arc1)
        plt.gca().add_patch(arc2)

        plt.text(
            x=_x_left, y=_y_left+y_offset,
            s='{:.1f}°'.format(left_ang),
            size=9
        )
        plt.text(
            x=_x_right+2, y=_y_right-y_offset*3,
            s='{:.1f}°'.format(right_ang),
            size=9,
        )
        plt.axis('off')
        plt.xlim(extent[0:2]); plt.ylim(extent[2:])
        if settings.deploy_mode == 'offline':
            plt.savefig(save_path + '/3.jpg', format='jpg', dpi=200, bbox_inches='tight')
            plt.close()
            return 
        else:
            bio = io.BytesIO()
            plt.savefig(bio, format='jpg', dpi=200, bbox_inches='tight')
            plt.close()
            return bio
    except Exception as e:
        raise e

def sd_cervical(model: CervicalReuqestModel, request: Request):

    auth_resp = auth_validate(model, request)
    if auth_resp:
        return auth_resp
    id, url = model.id, model.url
    if not os.path.exists('./log/sd_cervical_log'):
        os.mkdir('./log/sd_cervical_log')
    _, sid = os.path.split(url)
    logger = get_logger(sid, './log/sd_cervical_log')
    try:
        df = get_url_data(url)
        magnitudes = get_magnitude(df)
        json_keys = get_vel_ang(id, df)
        img_keys = draw(id, df)

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