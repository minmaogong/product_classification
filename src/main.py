from sys import argv

if __name__ == "__main__":
    print(argv)

    if len(argv) != 2:
        print("参数错误")
        exit()
    else:
        script = argv[1]
        match script:
            case "preprocess":
                from process.preprocess import preprocess
                preprocess()

            case "train":
                from runner.train import train
                train()
