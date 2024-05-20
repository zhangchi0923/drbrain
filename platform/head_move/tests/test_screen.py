import requests
import json
import hashlib
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))

from config.settings import settings


def get_md5(d):
    d = dict(sorted(d.items()))
    s = ''
    for k, v in d.items():
        s += str(k) + str(v)
    s = settings.salt + s
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()


if __name__ == '__main__':
    data2post = {
        "sex":  "女",
        "age": 52,
        "education": "专科",
        # "url": "https://cos.drbrain.net/profile/tj/2023/5/16/3c6b49df-3d93-40ec-8bce-70390174becf.txt",
        "url": "https://cos.drbrain.net/profile/tj/2023/5/18/e80eea4d-0ef6-4675-86ff-e73593d25121.txt",
        # "url": "E:/zc/GitProjects/algorithm-platform/platform/head_move/local_data/eye/eye_data.txt",
        # "url": "/usr/local/project/api/algorithm/algorithm-platform-test/platform/head_move/local_data/eye/eye_data.txt",
        # "backupResources": "/usr/local/project/algorithm/backup/utiles/design/",
        "questionVersion": 'A',
        "saveResourcesPath": "./log/eyescreen_log/"
    }
    crypt2post = get_md5(data2post)
    with requests.post(
        url='http://127.0.0.1:8101/eye/screen',
        data=json.dumps(data2post),
        headers={'Authorization': ''.join(crypt2post)}
    ) as r:
        print(r.text)
        print(r.status_code)
