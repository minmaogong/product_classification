# import sys
# print(sys.path)

from datasets import load_dataset, ClassLabel  # 加载非箭头arrow格式文件
from transformers import AutoTokenizer
from config.config import *

def preprocess():
    print("开始数据预处理 ...")
    # 1. 从文件加载数据集
    dataset_dict = load_dataset('csv', data_files={
        'train': str(RAW_DATA_DIR/TRAIN_DATA_FILE),
        'valid': str(RAW_DATA_DIR/VALID_DATA_FILE),
        'test': str(RAW_DATA_DIR/TEST_DATA_FILE)
    }, delimiter='\t') # delimiter='\t' 按\t 分割
    # print(dataset_dict)
    # 2. 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # 3. 编码标签，得到类别编码
    # 3.1 获取所有标签列表
    all_labels = sorted(set(dataset_dict['train']['label']))
    print(all_labels)
    # 3.2 将标签列表保存成文件
    with open(MODEL_DIR/LABELS_FILE, 'w', encoding='utf-8') as f:
        f.write("\n".join(all_labels))
    # 3.3 将label列转成ClassLabel
    class_label = ClassLabel(names=all_labels)
    dataset_dict = dataset_dict.cast_column('label', class_label) # 将label列转成ClassLabel

    # 4. 编码文本（商品标题），得到模型输入
    def batch_encode(batch):
        inputs = tokenizer(batch['text_a'], truncation=True) # 默认最大长度读取分词器config中的max_length 512
        inputs['labels'] = batch['label']
        return inputs
    dataset_dict = dataset_dict.map(batch_encode, batched=True, remove_columns=['text_a', 'label'])
    print(dataset_dict)
    print(dataset_dict['train'][:3])

    # 5. 保存数据集
    dataset_dict.save_to_disk(PROCESSED_DATA_DIR)


if __name__ == "__main__":
    preprocess()

