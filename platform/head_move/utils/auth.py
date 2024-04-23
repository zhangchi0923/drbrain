import hashlib
from fastapi import Request
from pydantic import BaseModel 
from utils.response_template import GeneralResponseModel
from config.settings import SALT


def get_md5(d):
    d = dict(sorted(d.dict().items()))
    s=''
    for k, v in d.items():
        s += str(k) + str(v)
    s = SALT + s
    md5 = hashlib.md5()
    md5.update(s.encode("utf-8"))
    return md5.hexdigest()

def auth_validate(model: BaseModel, request: Request):
    auth = request.headers.get('Authorization')
    auth_srv = get_md5(model)
    if auth_srv != auth:
        return GeneralResponseModel(
            code=401,
            msg='Authorization failed.',
            body=None
        )
    else:
        return None