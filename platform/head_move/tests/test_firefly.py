import requests
import json
import hashlib

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from config.settings import settings


def get_md5(d):
    d = dict(sorted(d.items()))
    s=''
    for k, v in d.items():
        s += str(k) + str(v)
    s = settings.salt + s
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()


if __name__ == '__main__':
    data2post = {
        # "url":  "https://cos.drbrain.net/profile/tj/2023/3/29/0626dfee-6bdd-4b48-af3e-d6545094d1d6.txt",
        # "url":  "https://cos.drbrain.net/profile/tj/2023/7/6/e0c39e46-4c4f-4eaa-afb6-d02a82362600.txt",
        "url":  "E:/zc/GitProjects/algorithm-platform/platform/head_move/local_data/ff/ff_data.txt",
        "savePath": './log/firefly_log/',
    }
    crypt2post = get_md5(data2post)
    with requests.post(
        url='http://127.0.0.1:8101/eye/train/firefly', 
        data=json.dumps(data2post), 
        headers={'Authorization':''.join(crypt2post)}
    ) as r:
        print(r.text)
        print(r.status_code)