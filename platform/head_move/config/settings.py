from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # crypt parameters
    salt: str = "drbrain&*("

    # pingpong parameters
    pp_path: str = './assets/src_fig_pp'

    # screen eye movement parameters
    max_vel: int = 100
    min_dur: int = 60
    max_ang_diff: float = 0.5
    max_time_interval: int = 75
    aois: dict = {
        3:[(1.644000, -1.792000),(2.844000, -0.592000)],
        4:[(-0.444000, -0.190000),(0.444000, 0.698000)],
        5:[(-0.444000, -1.638000),(0.444000, -0.750000)],
        6:[(-2.532000, -1.638000),(-1.644000, -0.750000)],
        7:[(-2.700000, -1.800000),(-1.300000, -0.600000)],
        8:[(0.678000, -0.398000),(2.570000, 0.686000)],
        9:[(-2.570400, -1.806000),(-0.678400, -0.722000)],
        10:[(-2.570400, -0.398000),(-0.678400, 0.686000)],
        11:[(-2.570400, -1.802000),(-0.678400, -0.718000)]
        }
    bezier_points: list = [[0.000, -0.040],
        [0.376, -1.440],
        [2.284, -1.440],
        [2.620, -0.040],
        [2.620, -0.040],
        [2.284,  1.360],
        [0.376,  1.360],
        [0.000, -0.040],
        [0.000, -0.040],
        [-0.376,-1.440],
        [-2.284,-1.440],
        [-2.620,-0.040],
        [-2.620,-0.040],
        [-2.284, 1.360],
        [-0.376, 1.360],
        [0.000, -0.040]]

    # MOCA_MODEL_PATH = './moca_et_pipeline.pkl'
    # MMSE_MODEL_PATH = './mmse_et_pipeline.pkl'

    secret_id: str = "AKID31oRvG5YTDmANPrwQj4WuEQTNBoGmDRf"
    secret_key: str = "HOK5f5KWaTCj5GG5jCXjldKyY4PZYN1T"
    region: str = "ap-guangzhou"
    url_prefix: str = "https://cos.drbrain.net/"
    prefix: str = "profile/pcat/prod"
    bucket_name: str = "product-c-formal-1254083048"

    sd_prefix: str = "profile/sd-cervical/prod"


settings = Settings()