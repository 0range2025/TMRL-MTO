import torch 


LORA_ALPHA=1    # lora的a权重
LORA_R=8    # lora的秩
DEVICE = "cuda" if torch.cuda.is_available() else "cpu" # 训练设备