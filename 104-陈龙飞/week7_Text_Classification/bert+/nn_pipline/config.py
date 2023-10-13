# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "log_path": "log",
    "log_name": "main_bert系列模型_12层_15轮_训练和评估日志_2023_09_24_02_16.log",
    "log_format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "result_file": "bert+模型效果对比.csv",
    "data_path": "../data",
    "data_name": "文本分类练习.csv",
    "str_lens_path": "../data/str_lens.csv",
    "model_path": "bert+_layer12_epoch15_output_2023_09_24_02_16",
    "train_data_path": "../data/train_review_label.json",
    "valid_data_path": "../data/valid_review_label.json",
    "vocab_path": "chars.txt",
    "model_type": "fast_text",
    "max_length": 40,
    "hidden_size": 128,
    "kernel_size": 3,
    "num_layers": 2,
    "epoch": 15,
    "batch_size": 64,
    # "pooling_style": "max",
    "pooling_style": "avg",
    "optimizer": "adam",
    "learning_rate": 1e-3,
    # "learning_rate": 1e-5,
    "pretrain_model_path": r"C:\Users\95699\Documents\PycharmProjects\ClassProject\week6_Language_Model\huggingface_model\bert-base-Chinese",
    "seed": 987
}
