import requests
import json
import hashlib

from settings import SALT
def get_md5(d):
    d = dict(sorted(d.items()))
    d = SALT + str(d)
    md5 = hashlib.md5()
    md5.update(d.encode('utf-8'))
    return md5.hexdigest()


if __name__ == '__main__':
    data2post = {
        'urlHead':'https://cos.drbrain.net/profile/tj/2022/11/25/888d6ccc-b137-414c-9a8c-a56f751cfc5d.txt',
        'urlBall':'https://cos.drbrain.net/profile/tj/2022/11/25/16d2aed6-86ab-4ce8-92d6-096341abd520.txt',
        'inPath':'./',
        'outPath':'./',
        'mode':'stand'
    }
    crypt2post = get_md5(data2post)
    r = requests.post(
        url='http://127.0.0.1:6666/head_move', 
        data=json.dumps(data2post), 
        headers={'Authorization':''.join(crypt2post)}
    )
    print(r.text)
    print(r.status_code)
