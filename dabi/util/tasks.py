import os
from typing import Dict, List

from dabi.util.util import upload_local_file_to_oss_direct, sync_result_to_dabi, copy_local_file, upload_sample_image
from mikazuki.log import log
from mikazuki.tasks import tm, TaskStatus
from enum import Enum
from datetime import datetime

class DabiTaskStatus(Enum):
    TRAIN_FAILED = 4
    UPLOADED = 5
    UPLOAD_FAILED = 6

class DabiTask:
    def __init__(self):
        self.task_id : str = ''
        self.config : Dict = {}
        self.models : List[dict] = []
        self.sample_images : List[dict] = []

class DabiTaskManager:
    def __init__(self):
        self.tasks: Dict[DabiTask] = {}

    def upload_model_to_oss(self, task_id):
        try:
            dabi_task = self.tasks[task_id]
            env = dabi_task.config['dabi_env']
            max_train_epochs = dabi_task.config['max_train_epochs']
            sub_output_dir = dabi_task.config['output_dir']
            output_model_name = dabi_task.config['output_name']
            output_model_extension = dabi_task.config['save_model_as']
            local_model_path = os.path.join(os.getcwd(), sub_output_dir)
            output_model_size_match_epoch = False
            if os.path.exists(local_model_path):
                model_files = [model_file for model_file in os.listdir(local_model_path)
                                if os.path.isfile(os.path.join(local_model_path, model_file))
                                    and model_file.startswith(output_model_name)
                                    and model_file.endswith(output_model_extension)]
                if max_train_epochs == len(model_files):
                    output_model_size_match_epoch = True
                    model_files.sort()
                    timestamp = datetime.now().strftime("%Y%m%d")
                    for item in model_files:
                        # 上传模型文件时判断任务是否处于FINISHED状态，如果期间任务被中止，就不在上传
                        if self.tasks[task_id].status == TaskStatus.FINISHED:
                            oss_file_path = upload_local_file_to_oss_direct(env, item, os.path.join(local_model_path, item), timestamp)
                            dabi_task.models.append({
                                "name": item,
                                "oss_path": oss_file_path
                            })
                            try:
                                target_file_dir = f"/mnt/train_loras/{timestamp}"
                                copy_local_file(item, local_model_path, item, target_file_dir)
                            except  Exception as e:
                                log.error(f"Copy model file to /mnt/train_loras failed, Error fetching : {e}")
                        else:
                            log.info(f"Task {task_id} may be terminated, skip upload model file: {item}")

            local_sample_image_path = os.path.join(local_model_path, "sample")
            if os.path.exists(local_sample_image_path):
                sample_images = [sample_image for sample_image in os.listdir(local_sample_image_path)
                                    if os.path.isfile(os.path.join(local_sample_image_path, sample_image))
                                        and sample_image.startswith(output_model_name)]
                if max_train_epochs == len(sample_images):
                    output_model_size_match_epoch = True
                    sample_images.sort()
                    for item in sample_images:
                        # 上传模型文件时判断任务是否处于FINISHED状态，如果期间任务被中止，就不在上传
                        if self.tasks[task_id].status == TaskStatus.FINISHED:
                            oss_file_path = upload_sample_image(env, item, os.path.join(local_sample_image_path, item))
                            dabi_task.sample_images.append({
                                "name": item,
                                "oss_path": oss_file_path
                            })
                        else:
                            log.info(f"Task {task_id} may be terminated, skip upload sample image: {item}")
            if output_model_size_match_epoch:
                self.set_task_status(task_id, DabiTaskStatus.UPLOADED)
                sync_result_to_dabi(env, dabi_task, 0)
            else:
                sync_result_to_dabi(env, dabi_task, 1)
                self.set_task_status(task_id, DabiTaskStatus.UPLOAD_FAILED)
        except Exception as e:
            sync_result_to_dabi(env, dabi_task, 1)
            self.set_task_status(task_id, DabiTaskStatus.UPLOAD_FAILED)
            log.error(f"Upload model to oss throw exception: {e}")

    def sync_result_to_dabi(self, task_id, code):
        if task_id in self.tasks:
            dabi_task = self.tasks[task_id]
            env = dabi_task.config['dabi_env']
            sync_result_to_dabi(env, dabi_task, code)


    def dump(self) -> List[Dict]:
        return [
            {
                "id": task.task_id,
                "status_name": task.status.name,
                "status_value": task.status.value
            }
            for task in self.tasks.values()
        ]

    def set_task_status(self, task_id, status):
        if task_id not in self.tasks:
            return
        dabi_task = self.tasks[task_id]
        dabi_task.status = status

dabi_task_manager = DabiTaskManager()