import os

class BaseConfig:
    # 路径设置
    data_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), "PlantDoc")
    train_dir = os.path.join(data_root, "TRAIN")
    test_dir = os.path.join(data_root, "TEST")
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")

    # 训练参数
    num_classes = 27
    image_size = 224
    batch_size = 32
    num_workers = 8
    epochs = 30
    learning_rate = 1e-3
    weight_decay = 1e-4
    optimizer = "adam"  # or "sgd"

    # 设备
    device = "cuda"

    # 其他
    seed = 42

    # 日志 & 模型保存
    experiment_name = "baseline_cnn"
    save_best_only = True

    @classmethod
    def get_experiment_dir(cls):
        exp_dir = os.path.join(cls.output_dir, cls.experiment_name)
        os.makedirs(exp_dir, exist_ok=True)
        return exp_dir