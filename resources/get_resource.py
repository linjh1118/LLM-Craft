import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download

ignore_patterns = ['*.h5', '*.msgpack', '*.onnx', '*.ot']
import sys
print(sys.argv)
if len(sys.argv) == 3:
    assert sys.argv[2] == 'dataset'
    dataset_id = sys.argv[1]
    datset_name = dataset_id.split('/')[-1]
    snapshot_download(dataset_id, local_dir=datset_name)
elif len(sys.argv) == 2:
    model_id = sys.argv[1]
    model_name = model_id.split('/')[-1]
    snapshot_download(model_id, local_dir=model_name, ignore_patterns=ignore_patterns)


""" huggingface-cli 命令行方式，适用于没有安装镜像
if len(sys.argv) == 2:
    # 下载数据集, 如果传入两个参数
    # python get_resource.py dataset_id remain_data
    # dataset_name = 'silk-road/ChatHaruhi-Expand-118K'
    dataset_name = sys.argv[0]
    os.system(f'huggingface-cli download --repo-type dataset \
              --resume-download {dataset_name} --local-dir {local_path}')
elif len(sys.argv) == 1:
    # get_resource.py model_id
    # model_naem = 'internlm/internlm-chat-7b'
    model_name = sys.argv[0]
    os.system(f'huggingface-cli download \
              --resume-download {model_name} --local-dir {local_path}')
else:
    raise ValueError('Invalid number of arguments')
"""


