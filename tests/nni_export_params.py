import os
import nni
from nni.experiment import Experiment
import requests
experiment = Experiment('local')
experiment.id = 'gemm_3bit_16x3072x768'
experiment.config.experiment_working_directory = '.nnidatabase'
experiment.config.experiment_name = f'_autogptq_search_{experiment.id}'
experiment.config.tuner.name = 'Evolution'
experiment.config.tuner.class_args['optimize_mode'] = 'minimize'
experiment.config.tuner.class_args['population_size'] = 2048
experiment.config.max_trial_number = 10
experiment.config.trial_concurrency = 2
experiment.config.trial_gpu_number = 1
experiment.config.tuner_gpu_indices = [0]
experiment.config.use_annotation = False
experiment.config.training_service.use_active_gpu = True
experiment.config.training_service.platform = 'local'
experiment.config.training_service.max_trial_number_per_gpu = 1
experiment.config.training_service.gpu_indices = [0]


# experiment.view('gemm_3bit_16x3072x768', non_blocking=True)
import sqlite3

db_path = "/workspace/v-leiwang3/lowbit_workspace/GPTQ-tvm/.nnidatabase/gemm_4bit_16x6656x6656/db/nni.sqlite"  # 替换为您的数据库文件路径

# 连接到数据库
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# 查询 MetricData 表中具有最低 data 值的记录
cursor.execute("SELECT * FROM MetricData")
all_data_record = cursor.fetchall()

# find the minimum data record (sort by the last column)
all_data_record.sort(key=lambda x: float(x[-1].replace('"','')))
lowest_metric_data_record = all_data_record[0]

if lowest_metric_data_record:
    print(f"Lowest metric data record: {lowest_metric_data_record}")

    # 获取 trialJobId
    trial_job_id = lowest_metric_data_record[1]  # trialJobId 是记录中的第二个字段
    print(f"Trial job ID: {trial_job_id}")

    # 使用 trialJobId 查询 TrialJobEvent 表中的 data
    cursor.execute("SELECT * FROM TrialJobEvent WHERE trialJobId = ?", (trial_job_id,))
    trial_job_event_records = cursor.fetchall()

    is_trail_success = False
    for record in trial_job_event_records:
        if record[2] == "SUCCEEDED":
            is_trail_success = True
            break
    for record in trial_job_event_records:
        if record[2] == "WAITING":
            print(record[3], type(record[3]), eval(record[3])['parameters'], type(eval(record[3])))
else:
    print("No metric data records found.")

# 关闭数据库连接
conn.close()


# experiment.stop()