#!/usr/bin/env python3
# -*- coding:utf-8 -*--
from src.config_gpu import BASE_DIR
from src.framework import detect_device
from src.framework.cfgs.ner_config import NerConfig
from src.framework.dataset.ner.ner_dataloader import NERDataLoader
from src.framework.model.ner.bert_bi_lstm_crf import BertBiLstmCrf
from src.framework.predictor import Predictor4NerCrf
from src.framework.trainer_ner_crf import Trainer4NerCrf


def train():
    _corpus_path = (BASE_DIR.joinpath("data", "content_ner", "ner_annotations.train"),
                    BASE_DIR.joinpath("data", "content_ner", "ner_annotations.dev"),
                    BASE_DIR.joinpath("data", "content_ner", "ner_annotations.test"))
    cfg = NerConfig()
    dataloader = NERDataLoader(cfg, _corpus_path)
    train_dataloader = dataloader.train_dataloader()
    val_dataloader = dataloader.val_dataloader()

    model = BertBiLstmCrf(cfg)
    device = detect_device()
    # device = "cpu"

    n_epoch = 30
    scheduler_params = {"num_warmup_steps": int(len(train_dataloader) * n_epoch * 0.01),
                        "num_training_steps": len(train_dataloader) * n_epoch}
    trainer = Trainer4NerCrf(model, n_epoch=n_epoch, device=device, early_stop_patience=3,
                             scheduler_params=scheduler_params, model_path=cfg.model_file)
    trainer.fit(train_dataloader, val_dataloader)
    print(model)

    # 离线评估
    test_dataloader = dataloader.test_dataloader()
    predictor = Predictor4NerCrf(cfg)
    predictor.evaluate(val_dataloader)
    predictor.evaluate(test_dataloader)

    # 在线预测
    sentences = [
        "广东省广播电视网络股份有限公司佛山南海分公司2017-2018年日常办公用品供货资格采购项目招标公告",
        "中山市妇女儿童活动中心印刷服务定点采购合同",
        "广州市越秀区城市管理局2018年环卫设备购置项目（项目编号：QSHG201800098）修改通知",
    ]
    predicts = predictor.predict(sentences)
    for sentence, pred in zip(sentences, predicts):
        print(sentence)
        print(pred)
        print()


if __name__ == "__main__":
    train()
