from torch.utils.data import DataLoader
from datasets import load_from_disk # 加载箭头arrow格式文件
from transformers import AutoTokenizer, DataCollatorWithPadding # DataCollatorWithPadding 带padding填充的数据整理器
from config.config import *

def get_dataset(ds_type):
    # 加载数据集
    path = str(PROCESSED_DATA_DIR / ds_type)
    dataset = load_from_disk(path)

    # 设置Tensor格式
    dataset.set_format(type='torch')
    return dataset

def get_dataloader(tokenizer, ds_type='train'):
    dataset = get_dataset(ds_type)

    # 定义数据整理器
    collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True, # 等价于 padding='longest'：按批次内最长序列动态填充
    )

    # 定义数据加载器
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator)

    return dataloader

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataloader = get_dataloader(tokenizer, ds_type='train')
    for batch in dataloader:
        for k, v in batch.items():
            print(k, '->', v.shape)
        break
