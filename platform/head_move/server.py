"""
 author: zhangchi
 email: zcsjtu0923@gmail.com
 create date: 2022-12-15 11:36:19
"""

import uvicorn
from fastapi import FastAPI
import warnings
warnings.filterwarnings('ignore')

from utils.Pingpong import pingpong
from utils.Balance import balance
from utils.EyeScreen import eye_screen, eye_screen_v1
from utils.Firefly import firefly
from utils.PCAT import eye_pcat
from utils.Cervical import sd_cervical


app = FastAPI()

app.post("/balance")(balance)
app.post("/pingpong")(pingpong)
app.post("/eye/screen")(eye_screen)
app.post("/eye/screen/v1")(eye_screen_v1)
app.post("/eye/train/firefly")(firefly)
app.post("/eye/pcat")(eye_pcat)
app.post("/rehab/sd/cervical")(sd_cervical)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8101)
    