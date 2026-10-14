from pathlib import Path

# 目录
ROOT_DIR = Path(__file__).parent.parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# 文件
TRAIN_DATA_FILE = "train.txt"
TEST_DATA_FILE = "test.txt"
VALID_DATA_FILE = "valid.txt"

MODEL_NAME = "google-bert/bert-base-chinese"

LABELS_FILE = "labels.txt"

# 超参数
LEARNING_RATE = 1e-5
BATCH_SIZE = 64
EPOCHS = 10

SAVE_STEPS= 50 # 每50个批次查看一次损失，判断是否保存模型
EARLY_STOP_PATIENCE = 5 # 早停策略：如果验证集损失在连续5个评估周期内没有下降，则停止训练
