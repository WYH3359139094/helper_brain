# import torch
# import torchvision
#
# print(f"当前GPU是否可用：{torch.cuda.is_available()}")
# print(f"设备名：{torch.cuda.get_device_name()}")
#
# print(torch.__version__, torchvision.__version__)
import os
from modelscope import snapshot_download

ROOT = r"D:\agentpro_model\modelsco"
os.makedirs(ROOT, exist_ok=True)

MODELS = [
    ("BAAI/bge-m3", "bge-m3"),
    ("BAAI/bge-reranker-v2-m3", "bge-reranker-v2-m3"),
]

EXCLUDE = ["onnx/*", "imgs/*", "*.md", ".gitattributes"]

for repo_id, folder in MODELS:
    target = os.path.join(ROOT, folder)
    print(f"下载 {repo_id} -> {target}")
    try:
        path = snapshot_download(model_id=repo_id, local_dir=target, ignore_patterns=EXCLUDE)
    except TypeError:
        path = snapshot_download(model_id=repo_id, local_dir=target, ignore_file_pattern=EXCLUDE)
    print(f"完成：{os.path.abspath(path)}\n")
