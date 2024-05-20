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
        s += k+v
    s = settings.salt + s
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()


if __name__ == "__main__":
    data2post = {
        "urlHead": "https://cos.drbrain.net/profile/tj/2023/2/19/73c54bbf-579b-4d16-9b4c-d57af6d4c057.txt",
        # "urlHead": "E:/zc/GitProjects/algorithm-platform/platform/head_move/local_data/balance/head_data.txt",
        # "urlHead": "/usr/local/project/api/algorithm/algorithm-platform-test/platform/head_move/local_data/balance/head_data.txt",
        "urlBall": "https://cos.drbrain.net/profile/tj/2023/2/19/fd86650e-58e3-4890-8f06-8ebce6cd4ddc.txt",
        # "urlBall": "E:/zc/GitProjects/algorithm-platform/platform/head_move/local_data/balance/sphere_data.txt",
        # "urlBall": "/usr/local/project/api/algorithm/algorithm-platform-test/platform/head_move/local_data/balance/sphere_data.txt",
        "outPath": "./log/balance_log",
        "mode": "sit"
    }
    crypt2post = get_md5(data2post)
    # print(crypt2post)

    with requests.post(
        url="http://127.0.0.1:8101/balance",
        data=json.dumps(data2post),
        headers={"Authorization": "".join(crypt2post)}
    ) as r:
        print(r.text)
