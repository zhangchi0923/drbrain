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
        "id":  601,
        "type": 'SYMBOL_SEARCH',
        # "type": "PORTRAIT_MEMORY",
        # "type": "VOCABULARY_TEST",
        # "type": "ORIGAMI_TEST",
        "url": "https://cos.drbrain.net/profile/tj/2023/8/25/49321be2-ae59-4d25-8372-95c4527da32d.txt", # symbol
        # "url": "https://cos.drbrain.net/profile/tj/2023/8/30/32bfd502-1cfe-41bf-b6d3-4b19e729899d.txt", # portrait
        # "url": "https://cos.drbrain.net/profile/tj/2023/9/2/0811ae6f-08b8-4311-a4b1-97e23db4cf8d.txt", # origami
        # "url": "https://cos.drbrain.net/profile/tj/2023/9/2/1b2aacf4-5e42-4a7a-870e-2b48e2f8b06a.txt", # vocab
        # "url": "https://cos.drbrain.net/profile/tj/2023/4/9/3d560422-bf7b-483d-9283-87ad0115d992.txt", # eye track data for KeyError test
    }
    crypt2post = get_md5(data2post)
    with requests.post(
        url='http://127.0.0.1:8101/eye/pcat',
        data=json.dumps(data2post),
        headers={'Authorization': ''.join(crypt2post)}
    ) as r:
        print(r.text)
