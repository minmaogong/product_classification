# import sys
# print(sys.path)
from dataclasses import dataclass

import torch
from torch.optim import Adam

from config.config import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification # 带任务头的预训练模型，自带损失函数
from process.dataset import get_dataloader

# 定义训练配置类
@dataclass
class TrainConfig:
    epochs: int = EPOCHS
    output_dir: str = MODEL_DIR

# 定义训练器
class Trainer:
    # 初始化
    def __init__(self, model, train_loader, optimizer, device, config=TrainConfig()):
        self.model = model
        self.train_loader = train_loader
        self.optimizer = optimizer
        self.device = device
        self.config = config

    # 核心方法
    def train(self):
        pass



def train():
    print("开始训练 ...")
    # 1. 定义设备
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available()
                          else "cpu")
    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # 3. 加载预训练模型
    # 加载标签列表
    with open(MODEL_DIR / LABELS_FILE, 'r', encoding="utf-8") as f:
        all_labels = f.read().splitlines()
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels = len(all_labels),
        id2label= { id:label for id, label in enumerate(all_labels) },
        label2id= { label:id for id, label in enumerate(all_labels) }
    ).to(device)

    # 4. 定义优化器
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. 数据加载器
    train_loader = get_dataloader(tokenizer, ds_type="train")

    # 6. 定义训练器
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        device=device,
        config=TrainConfig(epochs=EPOCHS, output_dir=MODEL_DIR)
    )
    # 7. 训练
    trainer.train()

if __name__ == "__main__":
    train()

