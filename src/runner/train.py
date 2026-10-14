# import sys
# print(sys.path)
import time
from dataclasses import dataclass

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from config.config import *
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModelForSequenceClassification # DataCollatorWithPadding 带padding填充的数据整理器; AutoModelForSequenceClassification 带任务头的预训练模型，自带损失函数
from process.dataset import get_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

# 定义训练配置类
@dataclass
class TrainConfig:
    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    save_steps: int = SAVE_STEPS

    output_dir: str = MODEL_DIR
    logs_dir: str = LOGS_DIR

    # 定义早停参数
    early_stop_patience: int = EARLY_STOP_PATIENCE # 早停的耐心值，连续多少次验证指标没有提升就停止训练
    early_stop_metric: str = 'loss' # 早停的指标，默认使用验证集损失作为早停指标

    # 定义是否开启AMP功能（自动混合精度）
    use_amp: bool = True

# 定义训练器
class Trainer:
    # 初始化
    def __init__(self, model, train_dataset, valid_dataset, collator, compute_metrics, device, config=TrainConfig()):
        self.model = model.to(device)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.collator = collator
        self.compute_metrics = compute_metrics
        self.device = device
        self.config = config

        # 优化器
        self.optimizer = Adam(model.parameters(), lr=config.learning_rate)
        # Tensorboard写入器
        self.writer = SummaryWriter(log_dir=str(Path(config.logs_dir / time.strftime("%Y%m%d-%H%M%S"))))

        # 定义全局变量
        self.steps = 0
        self.early_stop_best_score = -float('inf') # 如果评估指标是loss损失，那么应该越小越好，如果是准确率、精确率、召回率和f1评分，那么应该越大越好，所以这里初始化为负无穷
        self.early_stop_counter = 0

        # AMP梯度缩放器
        self.gradS_scaler = torch.amp.GradScaler(device=device, enabled=config.use_amp)

    # 定义内部方法：获取数据加载器
    def _get_dataloader(self, dataset):
        # 设置Tensor格式
        dataset.set_format(type='torch')
        # 定义数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self.collator
        )
        return dataloader

    # 定义内部函数：训练一个批次(一步迭代)
    def _train_one_step(self, inputs):
        self.model.train()
        inputs = { k:v.to(self.device) for k, v in inputs.items()}
        # 在 autocast 上下文中执行前向传播. 作用是自动混合精度训练，减少显存占用和加速训练。dtype=torch.float16 表示使用半精度浮点数进行计算
        with torch.autocast(device_type=self.device.type, dtype=torch.float16, enabled=self.config.use_amp):
            outputs = self.model(**inputs)
        # 反向传播
        # 在调用 backward() 前退出 autocast 上下文
        # 不推荐在 autocast 环境中执行反向传播（backward）
        # 反向传播时，相关操作会使用与前向传播时 autocast 所选择相同的 dtype
        self.gradS_scaler.scale(outputs.loss).backward() # 放大 loss，调用 backward() 时对 loss 进行放大，生成放大后的梯度，防止梯度下溢
        # 更新参数
        # scaler.step() 会先对优化器中参数的梯度进行反缩放（恢复正常范围）
        # 如果梯度中不包含 inf 或 NaN，才会执行 optimizer.step() 更新参数
        # 如果包含 inf 或 NaN，optimizer.step() 会被跳过，避免错误更新
        self. gradS_scaler.step(optimizer=self.optimizer)
        # 更新 scaler 的缩放因子，供下一次迭代使用，自动动态调整 scale 保证稳定性
        self. gradS_scaler.update()
        self.optimizer.zero_grad()
        return outputs.loss.item()

    # 定义内部函数：判断是否早停，以及保存模型
    def _should_early_stop(self, metrics) -> bool:
        # 取出判定早停的指标
        metrics = metrics[self.config.early_stop_metric]
        # 转换成评分
        score = -metrics if self.config.early_stop_metric == 'loss' else metrics # 如果是loss指标，score取负值，因为loss越小越好；如果是其他指标，score取原值，因为指标越大越好
        # 如果评分超过历史最高，就保存模型
        if score > self.early_stop_best_score: # early_stop_best_score 默认是负无穷，所以loss指标取负值，就能与其他分数指标统一判断条件
            self.early_stop_best_score = score
            self.early_stop_counter = 0 # 计数器清零
            tqdm.write("保存模型...")
            self.model.save_pretrained(self.config.output_dir)
            return False
        else:
            # 如果评分降低，计数加1
            self.early_stop_counter += 1
            # 如果超过容忍度，就早停
            if self.early_stop_counter >= self.config.early_stop_patience:
                return True
            else:
                return False

    # 核心方法
    def train(self):
        # 获取数据加载器
        dataloader = self._get_dataloader(self.train_dataset)
        # 循环迭代
        for epoch in range(self.config.epochs):
            for inputs in tqdm(dataloader, desc=f"[训练：Epoch {epoch+1}]"):
                # 调用一批数据的训练过程，得到损失
                this_loss = self._train_one_step(inputs)
                self.steps += 1
                # 判断是否达到50次迭代，做验证，进行日志和模型保存
                if self.steps % self.config.save_steps == 0:
                    # 保存日志
                    tqdm.write(f"[Epoch {epoch+1} | Step: {self.steps}] Loss: {this_loss:.4f}")
                    self.writer.add_scalar('loss', this_loss, self.steps)
                    # 进行验证，得到评估指标
                    metrics = self.evaluate() # 每训练完成50个批次，进行一次验证，返回指标字典
                    metrics_str = '|'.join([f'{k}:{v:.4f}' for k, v in metrics.items()])
                    tqdm.write(f'[Evaluate] {metrics_str}')

                    # 早停逻辑
                    if self._should_early_stop(metrics):
                        tqdm.write("早停！")
                        return

                    # # 判断是否保存模型
                    # if metrics['loss'] < self.min_loss:
                    #     self.min_loss = metrics['loss']
                    #     tqdm.write("保存模型...")
                    #     self.model.save_pretrained(self.config.output_dir)

    # 验证评估方法，返回不同指标（字典）
    def evaluate(self) -> dict:
        # 获取验证数据加载器
        dataloader = self._get_dataloader(self.valid_dataset)
        self.model.eval()
        # 记录总损失，用于计算指标1 - 验证集平均损失
        total_loss = 0.0
        # 记录所有数据的预测标签和真实标签，用于计算其他指标（准确率，精确率，召回率，f1...）
        predictions = []
        targets = []
        # 遍历验证集
        with torch.no_grad():
            for inputs in tqdm(dataloader, desc=f"[验证]"):
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                # 前向传播
                outputs = self.model(**inputs)
                # 获取损失
                loss = outputs.loss.item()
                total_loss += loss
                # 添加预测结果到列表
                this_predictions = torch.argmax(outputs.logits, dim=-1) # outputs.logits返回的是num_labels个类别的概率分布组成的二维张量，形状为(batch_size, num_labels)，通过argmax获取概率最大的类别索引
                predictions.extend(this_predictions.tolist())
                # 添加真实标签到列表
                this_labels = inputs["labels"]
                targets.extend(this_labels.tolist())
        # 计算评估指标
        # 指标1: 平均损失
        loss = total_loss / len(dataloader)
        # 指标2: 其他指标，调用外部传入的计算函数，返回字典
        metrics = self.compute_metrics(predictions, targets)

        return {'loss': loss, **metrics} # **metrics 解包成字典


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
    # 获取验证数据集
    valid_dataset = get_dataset(ds_type='valid')
    # 定义数据整理器
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True, # 等价于 padding='longest'：按批次内最长序列动态填充
    )
    # 定义计算评估指标的函数
    def compute_metrics(predictions, targets) -> dict:
        accuracy = accuracy_score(targets, predictions) # 计算准确率
        precision = precision_score(targets, predictions, average='weighted') # 计算精确率，average='weighted'表示对每个类别的精确率进行加权平均，权重为每个类别的样本数
        recall = recall_score(targets, predictions, average='weighted') # 计算召回率
        f1 = f1_score(targets, predictions, average='weighted') # 计算F1分数
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        collator=collator,
        compute_metrics=compute_metrics,
        device=device,
        config=TrainConfig(
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            save_steps=SAVE_STEPS,
            output_dir=MODEL_DIR,
            logs_dir=LOGS_DIR,
            early_stop_patience=EARLY_STOP_PATIENCE,
            early_stop_metric='loss',
            use_amp=True
        )
    )
    # 5. 训练
    trainer.train()

if __name__ == "__main__":
    train()

