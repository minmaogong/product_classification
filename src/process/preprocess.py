import sys
print(sys.path)

from config.config import *
from transformers import AutoTokenizer, AutoModelForSequenceClassification



def preprocess():
    print("Preprocessing ...")
    print("项目根目录：", ROOT_DIR)


if __name__ == "__main__":
    preprocess()

