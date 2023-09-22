import requests
import json
import hashlib

from settings import SALT
def get_md5(d):
    d = dict(sorted(d.items()))
    s=''
    for k, v in d.items():
        s += k+v
    s = SALT + s
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()


if __name__ == '__main__':
    data2post = {
        # 'dataUrl':'https://cos.drbrain.net/profile/tj/2023/2/13/98eef05c-5100-42ba-a3f6-9cc629d7327f.txt', # left
        # 'dataUrl':'https://cos.drbrain.net/profile/tj/2023/2/13/466d4dd2-f81b-4d51-b478-090c75f6ba98.txt', # right
        'dataUrl':'https://cos.drbrain.net/profile/tj/2023/9/3/495a8013-38e4-479d-98ae-9d22e0718b91.txt', # right
        'holdType':'A',
        # 'holdType':'B',
        'path':'./',
    }
    crypt2post = get_md5(data2post)
    with requests.post(
        url='http://127.0.0.1:8101/pingpong', 
        data=json.dumps(data2post), 
        headers={'Authorization':''.join(crypt2post)}
    ) as r:
        print(r.text)
        print(r.status_code)