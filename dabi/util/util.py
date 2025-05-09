import json
import os.path

import requests

from dabi.util.config_manager import get_oss_config, get_dabi_config
from mikazuki.log import log
import oss2
from datetime import datetime
def get_url(env, path, expire_seconds):
    oss_config = get_oss_config(env)
    url = oss_config['oss_host'] + '/getTempUrl'
    data = {
        'bucketName': oss_config['bucket_name'],
        'path': path,
        'appToken': oss_config['app_token'],
        'appId': oss_config['app_id'],
        'expireSeconds': expire_seconds
    }
    try:
        # 只要服务通的就能获取到地址，不会校验文件是否存在
        response = requests.post(url, data=data, timeout=30)
        res = response.json()
        if res['code'] == 0:
            return res['data']['url']
        raise Exception(response.text)
    except Exception as ex:
        raise ex

def upload_oss(env, files, expire_seconds):
    oss_config = get_oss_config(env)
    data = {
        'bucketName': oss_config['bucket_name'],
        'dir': oss_config['lora_train_dir'],
        'appToken': oss_config['app_token'],
        'appId': oss_config['app_id']
    }
    try:
        log.info(f"Start upload to oss")
        response = requests.post(f"{oss_config['oss_host']}/upload/multipart",
                                 data=data,
                                 files=files,
                                 timeout=expire_seconds)
        response.raise_for_status()  # 如果请求失败，抛出异常
        json = response.json()
        path = json['data']['path']
        return path  # 返回响应内容，这里只是作为示例，实际使用中可能不需要返回
    except Exception as e:
        log.error(f"Upload to oss failed, Error fetching : {e}")
        return None

def upload_local_file_to_oss_direct(env, oss_file_name, local_file):
    oss_config = get_oss_config(env)
    oss_dir = oss_config['lora_train_dir']
    access_key_id = oss_config['access_key_id']
    access_key_secret = oss_config['access_key_secret']
    bucket_name = oss_config['bucket_name']
    endpoint = oss_config['oss_endpoint']
    region = oss_config['region']
    # 使用获取的RAM用户的访问密钥配置访问凭证
    auth = oss2.AuthV4(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name, region=region)
    try:
        with open(local_file, 'rb') as file:
            oss_file_path = f"{oss_dir}{datetime.now().strftime('%Y%m%d')}/{oss_file_name}"
            log.info(f"Start uploading to oss success, local file: {local_file}, oss file path: {oss_file_path}")
            bucket.put_object(oss_file_path, file)
            log.info(f"Upload to oss success, local file: {local_file}, oss file path: {oss_file_path}")
            return oss_file_path
    except Exception as e:
        log.error(f"Upload to oss failed, Error fetching : {e}")
        return None

def upload_sample_image(env, local_file_name, local_file_dir):
    config = get_dabi_config(env)
    url = config['host'] + '/lora/train/uploadSampleImage'
    try:
        with open(local_file_dir, 'rb') as f:
            files = {'file': (local_file_name, f.read(), 'image/png')}
            # 只要服务通的就能获取到地址，不会校验文件是否存在
            response = requests.post(url, files=files, timeout=30)
            response.raise_for_status()  # 如果请求失败，抛出异常
            json = response.json()
            if "data" not in json or json['data'] is None:
                log.error(f"Upload to oss failed, local file: {local_file_name}, result={json}")
                return None
            res_data = json['data']
            if "path" not in res_data or res_data['path'] is None:
                log.error(f"Upload to oss failed, local file: {local_file_name}, result={json}")
                return None
            oss_file_path = res_data['path']
            log.info(f"Upload to oss success, local file: {local_file_name}, oss file path: {oss_file_path}")
            return oss_file_path
    except Exception as ex:
        log.error(f"Upload to oss failed, local file: {local_file_name}, oss file path: {oss_file_path}")

def sync_result_to_dabi(env, data, code):
    config = get_dabi_config(env)
    url = config['host'] + '/lora/train/callbackAfterTrainDone'
    param = {
        'code': code,
        'taskId': data.task_id,
        'models': data.models,
        'sample_images': data.sample_images,
    }
    try:
        # 只要服务通的就能获取到地址，不会校验文件是否存在
        response = requests.post(url, json=param, timeout=30)
        response.raise_for_status()  # 如果请求失败，抛出异常
        json = response.json()
        log.info(f"Sync result to dabi success, param = {data}, result={json}")
    except Exception as ex:
        log.error(f"Sync result to dabi failed, param = {data}")

def copy_local_file(source_file, source_file_dir, target_file, target_file_dir):
    #目标文件夹不存在，创建目标文件夹
    if not os.path.exists(target_file_dir):
        os.mkdir(target_file_dir)

    with open(os.path.join(source_file_dir, source_file), 'rb') as s_file:
        with open(os.path.join(target_file_dir, target_file), 'wb') as t_file:
            t_file.write(s_file.read())
            log.info(f"Copy file success, source file: {os.path.join(source_file_dir, source_file)}, target file: {os.path.join(target_file_dir, target_file)}")