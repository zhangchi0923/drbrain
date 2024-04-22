from pydantic import BaseModel
from typing import Union

class GeneralResponseModel(BaseModel):
    code: int
    msg: str
    body: Union[str, dict, None] = None