import copy
import pickle
import socket,ssl
from threading import Thread
import json
import time

import numpy as np
import pandas as pd
# from pyparsing import nested_expr
import torch

from utils.options import args_parser
from utils.train_utils import get_data, get_model
from utils.send_recv_utils import *
from models.Update import LocalUpdate
from models.test import test_img
import os
import logging
# logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.basicConfig(level=logging.DEBUG)

CLOUD_ADDRESS_PORT = ('127.0.0.1', 8700)
BUF_SIZE = 1024

if "__main__" == __name__:
    # parse argsclient
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
    print("device: {}".format(args.device))  

    # 配置SSL
    sock = socket.create_connection(CLOUD_ADDRESS_PORT)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.load_verify_locations("cert/cert.pem")
    edge_client = context.wrap_socket(sock, server_hostname="Cloud")

    welcome_msg = recv_msg(edge_client).decode()
    logging.info(welcome_msg)

    # 接收训练数据，后续应该改为在另一个线程中开个socket接收
    from torchvision import datasets, transforms
    trans_mnist = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.1307,), (0.3081,))])
    dataset_train = datasets.MNIST('data/mnist/', train=True, download=True, transform=trans_mnist)
    dataset_test = datasets.MNIST('data/mnist/', train=False, download=True, transform=trans_mnist)


    dict_user = pickle.loads(recv_msg(edge_client))
    idxs_train_local = dict_user["train"]
    idxs_test_local = dict_user["test"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 建立局部模型
    net_local = get_model(args)
    net_local.train()
    logging.info("模型初始化已完成")

    while True:
        """
        每轮训练时，先从cloud接受全局参数，然后根据是否被选中选择是否进行训练并更新参数，然后上传局部参数和loss
        """
        # 接受全局参数
        round_and_state = pickle.loads(recv_msg(edge_client))
        local_round = round_and_state["round"]
        net_globa_state_dict = round_and_state["state"]
        logging.info(f"{local_round} 接收到来自cloud的全局模型参数")
        net_local.load_state_dict(net_globa_state_dict)

        if pickle.loads(recv_msg(edge_client))["choosen"] == True:
            # 训练，获得局部参数和loss, 并发送给cloud
            logging.info(f"Round = {local_round} 参与训练")
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=idxs_train_local)
            w_local, loss_local = local.train(net=net_local.to(device))
            send_msg(edge_client, pickle.dumps({
                "w_local": w_local,
                "loss_local": loss_local
            }))
            logging.info(f"Round = {local_round} 向server发送 weight&loss")
        else:
            logging.infp(f"Round = {local_round} 未被选择参与训练")
            pass