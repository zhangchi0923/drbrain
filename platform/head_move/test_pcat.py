import requests
import json
import hashlib
import os

from settings import SALT


def get_md5(d):
    d = dict(sorted(d.items()))
    s = ''
    for k, v in d.items():
        s += str(k) + str(v)
    s = SALT + s
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()


if __name__ == '__main__':
    data2post = {
        "id":  4,
        "type": 'SYMBOL_SEARCH',
        # "type": "PORTRAIT_MEMORY",
        "url": "https://cos.drbrain.net/profile/tj/2023/8/25/49321be2-ae59-4d25-8372-95c4527da32d.txt", # symbol
        # "url": "https://cos.drbrain.net/profile/tj/2023/8/30/32bfd502-1cfe-41bf-b6d3-4b19e729899d.txt", # portrait
    }
    crypt2post = get_md5(data2post)
    with requests.post(
        url='http://127.0.0.1:8101/eye/pcat',
        data=json.dumps(data2post),
        headers={'Authorization': ''.join(crypt2post)}
    ) as r:
        print(r.text)
