import json
import os


# 读取环境配置
def read_config():
    config = {}
    # 配置文件路径
    json_config = os.path.join(os.getcwd(), "dabi", "config", "config.json")

    # 如果json_config文件不存在，则创建一个空的json文件
    if os.path.exists(json_config):
        with open(json_config, encoding='utf-8') as f:
            config = json.load(f)

    return {} if not config else config

config = read_config()

def get_config_by_env(env):
    return config[env] if env in config else {}

def get_oss_config(env):
    env_config = get_config_by_env(env)
    return env_config['oss_config']

def get_dabi_config(env):
    env_config = get_config_by_env(env)
    return env_config['dabi_config']