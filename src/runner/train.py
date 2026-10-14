# import sys
# print(sys.path)
from dataclasses import dataclass

import torch
from torch.optim import Adam

from config.config import *
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification # DataCollatorWithPadding 带padding填充的数据整理器; AutoModelForSequenceClassification 带任务头的预训练模型，自带损失函数
from process.dataset import get_dataset
from torch.utils.data import DataLoader

# 定义训练配置类
@dataclass
class TrainConfig:
    epochs: int = EPOCHS
    output_dir: str = MODEL_DIR
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE

# 定义训练器
class Trainer:
    # 初始化
    def __init__(self, model, train_dataset, collator, device, config=TrainConfig()):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.collator = collator
        self.device = device
        self.config = config
        # 优化器
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)

    # 定义内部方法：获取数据加载器
    def _get_dataloader(self):
        # 设置Tensor格式
        self.train_dataset.set_format(type='torch')
        # 定义数据加载器
        dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator
        )
        return dataloader

    # 核心方法
    def train(self):
        dataloader = self._get_dataloader()



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

    # 4. 定义训练器
    # 获取训练数据集
    train_dataset = get_dataset(ds_type='train')
    # 定义数据整理器
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True, # 等价于 padding='longest'：按批次内最长序列动态填充
    )
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        collator=collator,
        device=device,
        config=TrainConfig(epochs=EPOCHS, output_dir=MODEL_DIR)
    )
    # 5. 训练
    trainer.train()

if __name__ == "__main__":
    train()

