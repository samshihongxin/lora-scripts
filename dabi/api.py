import concurrent.futures
import json
import os
from datetime import datetime

import requests
import toml
from fastapi import APIRouter
from starlette.requests import Request

import mikazuki.process as process
from dabi.util import tasks
from dabi.util.util import get_url
from mikazuki.app.api import trainer_mapping, get_sample_prompts
from mikazuki.app.models import (APIResponseFail, APIResponseSuccess, APIResponse)
from mikazuki.log import log
from mikazuki.tasks import TaskStatus, tm
from mikazuki.utils import train_utils

router = APIRouter()

@router.post("/run")
async def create_toml_file(request: Request):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    toml_file = os.path.join(os.getcwd(), f"config", "autosave", f"{timestamp}.toml")
    json_data = await request.body()
    log.info(f"Submit train task, param: {json_data}")
    config: dict = json.loads(json_data.decode("utf-8"))

    if 'dabi_task_id' not in config:
        return APIResponseFail(message="dabi_task_id is required")
    task_id = config["dabi_task_id"]
    dabi_task = tasks.DabiTask()
    dabi_task.task_id = task_id
    tasks.dabi_task_manager.tasks[task_id] = dabi_task
    if 'dabi_env' not in config:
        return APIResponseFail(message="dabi_env is required")
    dabi_task.config = config

    train_utils.fix_config_types(config)
    gpu_ids = config.pop("gpu_ids", None)
    suggest_cpu_threads = 8 if len(train_utils.get_total_images(config["train_data_dir"])) > 200 else 2
    model_train_type = config.pop("model_train_type", "sd-lora")
    trainer_file = trainer_mapping[model_train_type]

    if model_train_type != "sdxl-finetune":
        if not train_utils.validate_data_dir(config["train_data_dir"]):
            return APIResponseFail(message="训练数据集路径不存在或没有图片，请检查目录。")

    validated, message = train_utils.validate_model(config["pretrained_model_name_or_path"], model_train_type)
    if not validated:
        return APIResponseFail(message=message)

    try:
        positive_prompt, sample_prompts_arg = get_sample_prompts(config=config)

        if positive_prompt is not None and train_utils.is_promopt_like(sample_prompts_arg):
            sample_prompts_file = os.path.join(os.getcwd(), f"config", "autosave", f"{timestamp}-promopt.txt")
            with open(sample_prompts_file, "w", encoding="utf-8") as f:
                f.write(sample_prompts_arg)

            config["sample_prompts"] = sample_prompts_file
            log.info(f"Wrote prompts to file {sample_prompts_file}")
    except ValueError as e:
        log.error(f"Error while processing prompts: {e}")
        return APIResponseFail(message=str(e))

    with open(toml_file, "w", encoding="utf-8") as f:
        f.write(toml.dumps(config))

    result = process.dabi_run_train(task_id, toml_file, trainer_file, gpu_ids, suggest_cpu_threads)

    return result

@router.post("/upload/train/data")
async def upload_train_data(request: Request):
    json_data = await request.body()
    log.info(f"Upload train data, param: {json_data}")
    req_data: dict = json.loads(json_data.decode("utf-8"))
    if 'env' not in req_data:
        return APIResponseFail(message="env is required")
    env = req_data['env']

    if 'trainDataId' not in req_data:
        return APIResponseFail(message="trainDataId is required")

    base_train_data_dir = os.path.join(os.getcwd(), "train")
    if not os.path.exists(base_train_data_dir):
        os.makedirs(base_train_data_dir)
    train_data_id = req_data['trainDataId']
    #创建训练数据文件夹
    if 'subTrainDir' not in req_data:
        return APIResponseFail(message="subTrainDir is required")
    sub_train_dir = req_data['subTrainDir']
    train_data_dir = os.path.join(base_train_data_dir, sub_train_dir, train_data_id)

    if not os.path.exists(train_data_dir):
        os.makedirs(train_data_dir)
    if 'trainTimesPerImage' not in req_data:
        return APIResponseFail(message="trainTimesPerImage is required")
    train_times_per_image = req_data['trainTimesPerImage']

    #创建训练数据文件夹下的子文件夹{train_times_per_image}_{train_data_id}
    full_train_data_dir = os.path.join(train_data_dir, f"{train_times_per_image}_{train_data_id}")
    if not os.path.exists(full_train_data_dir):
        os.makedirs(full_train_data_dir)

    def save_image_to_local(image):
        try:
            file_url = get_url(env, image['path'], 86400)
            response = requests.get(file_url, stream=True)
            with open(os.path.join(full_train_data_dir, image['displayName']), 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        except Exception as e:
            log.error(f"Save image to local failed, param: {image}, Error fetching : {e}")
            return None
    def save_tags_to_local(tag):
        try:
            with open(os.path.join(full_train_data_dir, tag['displayName']), 'w') as f:
                f.write(tag['content'])
        except Exception as e:
            log.error(f"Save tag to local failed, param: {tag}, Error fetching : {e}")
            return None
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=50)
    if 'images' in req_data:
        images = req_data['images']
        if images is not None and len(images) > 0:
            executor.map(save_image_to_local, images)
    if 'tags' in req_data:
        tags = req_data['tags']
        if tags is not None and len(tags) > 0:
            executor.map(save_tags_to_local, tags)
    # 效率不行
    # loop = asyncio.get_event_loop()
    # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    #     futures = []
    #     for image in train_data['images']:
    #         futures.append(loop.run_in_executor(executor, save_image_to_local, image))
    #
    #     for tag in train_data['tags']:
    #         futures.append(loop.run_in_executor(executor, save_tags_to_local, tag))
    #
    #     await asyncio.gather(*futures)
    return APIResponseSuccess()

@router.get("/tasks/terminate/{task_id}", response_model_exclude_none=True)
async def terminate_task(task_id: str):
    log.info(f"Terminate task {task_id}")
    try:
        tm.terminate_task(task_id)
    finally:
        if task_id in tm.tasks:
            task = tm.tasks[task_id]
            if not task.status == TaskStatus.TERMINATED:
                task.status = TaskStatus.TERMINATED
        if task_id in tasks.dabi_task_manager.tasks:
            dabi_task = tasks.dabi_task_manager.tasks[task_id]
            if not dabi_task.status == TaskStatus.TERMINATED:
                dabi_task.status = TaskStatus.TERMINATED
    return APIResponseSuccess()

@router.get("/tasks", response_model_exclude_none=True)
async def get_tasks() -> APIResponse:
    return APIResponseSuccess(data={
        "tasks": tasks.dabi_task_manager.dump()
    })

@router.get("/task/status/{task_id}", response_model_exclude_none=True)
async def task_status(task_id: str):
    if task_id not in tasks.dabi_task_manager.tasks:
        return APIResponseFail(message="Task not found")

    dabi_task = tasks.dabi_task_manager.tasks[task_id]

    return APIResponseSuccess(data = {
        "status_name": dabi_task.status.name,
        "status_value": dabi_task.status.value,
        "models": dabi_task.models,
        "sample_images": dabi_task.sample_images
    })

@router.get("/free", response_model_exclude_none=True)
async def task_status():
    #0空闲，1非空闲
    free = 0
    dabi_tasks = tasks.dabi_task_manager.tasks
    for task in dabi_tasks.values():
        if [TaskStatus.CREATED,
            TaskStatus.RUNNING,
            TaskStatus.FINISHED].__contains__(task.status):
            free = 1
            break

    return APIResponseSuccess(data = {
        "free": free
    })