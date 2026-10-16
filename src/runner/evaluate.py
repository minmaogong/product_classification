import torch
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding

from config.config import *
from process.dataset import get_dataset
from runner.train import Trainer, TrainConfig


def evaluate():
    print("开始评估 ...")
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
        MODEL_DIR/'best',
        num_labels = len(all_labels),
        id2label= { id:label for id, label in enumerate(all_labels) },
        label2id= { label:id for id, label in enumerate(all_labels) }
    ).to(device)

    # 4. 定义训练器
    # 获取测试数据集
    test_dataset = get_dataset(ds_type='test')
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
        train_dataset=None,
        valid_dataset=test_dataset,
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
    metrics = trainer.evaluate()
    print("评估结果:", metrics)

if __name__ == '__main__':
    evaluate()
