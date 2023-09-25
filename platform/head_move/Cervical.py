import io, math, datetime

import pandas as pd
from settings import *
from qcloud_cos import CosConfig, CosS3Client
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Wedge

class Cervical(object):
    def __init__(self, url, id):
        self.url = url
        self.id = id
    
    def get_url_data(self):
        try:
            df = pd.read_csv(self.url)
            df['timestamp'] = df['timestamp'] - df['timestamp'].min()
            df['pitch'] = df['pitch'].apply(lambda x: x-360 if x > 270 else x)
            df['yaw'] = df['yaw'].apply(lambda x: x-360 if x > 270 else x)
            df['roll'] = df['roll'].apply(lambda x: x-360 if x > 270 else x)
            self.df = df
            return df
        except Exception as e:
            raise e
    
    def get_magnitude(self) -> dict:
        try:
            df_action_1 = self.df.loc[(self.df['level'] == 1) | (self.df['level'] == 2)].describe()
            ang_1 = int(df_action_1.loc['max', 'pitch'])
            df_action_2 = self.df.loc[(self.df['level'] == 3) | (self.df['level'] == 4)].describe()
            ang_2 = int(abs(df_action_2.loc['min', 'pitch']))
            df_action_3 = self.df.loc[(self.df['level'] == 5) | (self.df['level'] == 6)].describe()
            ang_3 = int(df_action_3.loc['max', 'roll'])
            df_action_4 = self.df.loc[(self.df['level'] == 7) | (self.df['level'] == 8)].describe()
            ang_4 = int(abs(df_action_4.loc['min', 'roll']))
            df_action_5 = self.df.loc[(self.df['level'] == 9) | (self.df['level'] == 10)].describe()
            ang_5 = int(abs(df_action_5.loc['min', 'yaw']))
            df_action_6 = self.df.loc[(self.df['level'] == 11) | (self.df['level'] == 12)].describe()
            ang_6 = int(df_action_6.loc['max', 'yaw'])

            angles = [ang_1, ang_2, ang_3, ang_4, ang_5, ang_6]
            tags = ['forward', 'back', 'left', 'right', 'leftRot', 'rightRot']

            return dict(zip(tags, angles))
        except Exception as e:
            raise e
    
    def get_vel_ang(self):
        buffer_pos = []
        bio = io.BytesIO()
        def vel(row):
            dist = math.sqrt(row['x'] ** 2 + row['y'] ** 2 + row['z'] ** 2)
            vel = dist / row['timestamp']
            return vel
        # 1
        df_action = self.df.loc[(self.df['level'] == 1) | (self.df['level'] == 2)]
        df_action['timestamp'] = (df_action['timestamp'] - df_action['timestamp'].min())/1000
        diff = df_action[['timestamp', 'x', 'y', 'z']].diff()
        df_action['vel'] = diff.apply(lambda x: vel(x), axis=1)
        df_action['vel'].fillna(0, inplace=True)
        df_action[['timestamp', 'vel']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        df_action[['timestamp', 'pitch']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        # 2
        df_action = self.df.loc[(self.df['level'] == 3) | (self.df['level'] == 4)]
        df_action['pitch'] = -df_action['pitch']
        df_action['timestamp'] = (df_action['timestamp'] - df_action['timestamp'].min())/1000
        diff = df_action[['timestamp', 'x', 'y', 'z']].diff()
        df_action['vel'] = diff.apply(lambda x: vel(x), axis=1)
        df_action['vel'].fillna(0, inplace=True)
        df_action[['timestamp', 'vel']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        df_action[['timestamp', 'pitch']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        # 3
        df_action = self.df.loc[(self.df['level'] == 5) | (self.df['level'] == 6)]
        df_action['timestamp'] = (df_action['timestamp'] - df_action['timestamp'].min())/1000
        diff = df_action[['timestamp', 'x', 'y', 'z']].diff()
        df_action['vel'] = diff.apply(lambda x: vel(x), axis=1)
        df_action['vel'].fillna(0, inplace=True)
        df_action[['timestamp', 'vel']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        df_action[['timestamp', 'roll']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        # 4
        df_action = self.df.loc[(self.df['level'] == 7) | (self.df['level'] == 8)]
        df_action['roll'] = -df_action['roll']
        df_action['timestamp'] = (df_action['timestamp'] - df_action['timestamp'].min())/1000
        diff = df_action[['timestamp', 'x', 'y', 'z']].diff()
        df_action['vel'] = diff.apply(lambda x: vel(x), axis=1)
        df_action['vel'].fillna(0, inplace=True)
        df_action[['timestamp', 'vel']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        df_action[['timestamp', 'roll']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        # 5
        df_action = self.df.loc[(self.df['level'] == 9) | (self.df['level'] == 10)]
        df_action['yaw'] = -df_action['yaw']
        df_action['timestamp'] = (df_action['timestamp'] - df_action['timestamp'].min())/1000
        diff = df_action[['timestamp', 'x', 'y', 'z']].diff()
        df_action['vel'] = diff.apply(lambda x: vel(x), axis=1)
        df_action['vel'].fillna(0, inplace=True)
        df_action[['timestamp', 'vel']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        df_action[['timestamp', 'yaw']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        # 6
        df_action = self.df.loc[(self.df['level'] == 11) | (self.df['level'] == 12)]
        df_action['timestamp'] = (df_action['timestamp'] - df_action['timestamp'].min())/1000
        diff = df_action[['timestamp', 'x', 'y', 'z']].diff()
        df_action['vel'] = diff.apply(lambda x: vel(x), axis=1)
        df_action['vel'].fillna(0, inplace=True)
        df_action[['timestamp', 'vel']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())
        df_action[['timestamp', 'yaw']].to_json(bio, orient='values')
        buffer_pos.append(bio.tell())

        json_keys = self.save_data2cos(bio, buffer_pos)

        return json_keys
    
    def draw(self):
        try:
            config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY, Token=None, Scheme="https")
            client = CosS3Client(config)
            now = datetime.datetime.now()
            year, month, day = str(now.year), str(now.month), str(now.day)
            base_key = '/'.join([SD_PREFIX, year, month, day, str(self.id)])

            bio1 = self.draw1()
            key1 = URL_PREFIX + self._save_img2cos(bio1, client, base_key, 1)
            bio2 = self.draw2()
            key2 = URL_PREFIX + self._save_img2cos(bio2, client, base_key, 2)
            bio3 = self.draw3()
            key3 = URL_PREFIX + self._save_img2cos(bio3, client, base_key, 3)

            tags = ['side', 'front', 'up']
            keys = [key1, key2, key3]
            return dict(zip(tags, keys))
        except Exception as e:
            raise e


    def save_data2cos(self, bio, buffer_pos=None):

        try:
            config = CosConfig(Region=REGION, SecretId=SECRET_ID, SecretKey=SECRET_KEY, Token=None, Scheme="https")
            client = CosS3Client(config)
            now = datetime.datetime.now()
            year, month, day = str(now.year), str(now.month), str(now.day)
            base_key = '/'.join([SD_PREFIX, year, month, day, str(self.id)])

            keys = self._save_data2cos(bio, client, base_key, buffer_pos)
            return keys
        except Exception as e:
            raise e
    
    def _save_data2cos(self, bio, client, base_key, buffer_pos):
        tmp_keys = []
        result_keys = []
        try:
            for i in range((len(buffer_pos))):
                if i == 0:
                    bio.seek(0)
                    key = base_key + '/{}.json'.format(i+1)
                    client.put_object(
                        Bucket=BUCKET_NAME,
                        Body=bio.read1(buffer_pos[i]),
                        Key=key,
                    )
                else:
                    bio.seek(buffer_pos[i-1])
                    key = base_key + '/{}.json'.format(i+1)
                    client.put_object(
                        Bucket=BUCKET_NAME,
                        Body=bio.read1(buffer_pos[i] - buffer_pos[i-1]),
                        Key=key,
                    )
                
                tmp_keys.append(URL_PREFIX + key)
                if len(tmp_keys) == 2:
                    result_keys.append(dict(zip(['vel', 'angle'], tmp_keys)))
                    tmp_keys = []

            tags = ['forward', 'back', 'left', 'right', 'leftRot', 'rightRot']
            return dict(zip(tags, result_keys))
        except Exception as e:
            raise e
    
    def  _save_img2cos(self, bio, client, base_key, ord):
        try:
            key = base_key + '/{}.jpg'.format(ord)
            bio.seek(0)
            client.put_object(
                Bucket=BUCKET_NAME,
                Body=bio.read1(),
                Key=key
            )
            return key
        except Exception as e:
            raise e

        

    def draw1(self):
        try:
            im = plt.imread('./src_fig/head-2@2x.png')
            extent=[-30, 30, -47.5, 12.5]
            plt.imshow(im, extent=extent)
            x_center, y_center = 0, -15
            r=18
            x_offset, y_offset = 8, 2

            df_head = self.df.loc[self.df['level'].isin([1,2,3,4])]

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
            bio = io.BytesIO()
            plt.savefig(bio, format='jpg', dpi=200, bbox_inches='tight')
            plt.close()
            return bio
        except Exception as e:
            raise e
    
    def draw2(self):
        try:
            im = plt.imread('./src_fig/head-1@2x.png')
            extent=[-30, 30, -47.5, 12.5]
            plt.imshow(im, extent=extent)
            x_center, y_center = 0, -15
            r=18
            x_offset, y_offset = 8, 2

            df_head = self.df.loc[self.df['level'].isin([5,6,7,8])]


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
            bio = io.BytesIO()
            plt.savefig(bio, format='jpg', dpi=200, bbox_inches='tight')
            plt.close()
            return bio
        except Exception as e:
            raise e
    
    def draw3(self):
        try:
            im = plt.imread('./src_fig/head-3@2x.png')
            extent = [-75, 75, -81, 69]
            plt.imshow(im, extent=extent)
            x_center, y_center = 0, 0
            r=40
            x_offset, y_offset = 8, 2

            df_head = self.df.loc[self.df['level'].isin([9,10,11,12])]

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
            bio = io.BytesIO()
            plt.savefig(bio, format='jpg', dpi=200, bbox_inches='tight')
            plt.close()
            return bio
        except Exception as e:
            raise e