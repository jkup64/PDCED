
import copy
import pickle
import socket,ssl
from threading import Thread,Lock
import json
import time

import numpy as np
import pandas as pd
import torch

from utils.options import args_parser
from utils.send_recv_utils import recv_msg, send_msg
from utils.train_utils import get_data, get_model
from models.Update import LocalUpdate
from models.test import test_img
import os
import logging
# logging.basicConfig(format='%(asctime)s %(message)s', level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)

CLOUD_ADDRESS_PORT = ("127.0.0.1", 8700)
BUF_SIZE = 1024
g_socket_server = None
g_conn_pool = {}

# parse argsclient
args = args_parser()
args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
base_dir = './save/{}/{}_iid{}_num{}_C{}_le{}/shard{}/{}/'.format(
    args.dataset, args.model, args.iid, args.num_users, args.frac, args.local_ep, args.shard_per_user, args.results_save)
if not os.path.exists(os.path.join(base_dir, 'fed')):
    os.makedirs(os.path.join(base_dir, 'fed'), exist_ok=True)

dataset_train, dataset_test, dict_users_train, dict_users_test = get_data(args)
dict_save_path = os.path.join(base_dir, 'dict_users.pkl')
with open(dict_save_path, 'wb') as handle:
    pickle.dump((dict_users_train, dict_users_test), handle)

# build model
net_glob = get_model(args)  # 云端模型
net_glob.train()

# training
loss_train = []
net_best = None
best_loss = None
best_acc = None
best_epoch = None
lr = args.lr
results = []
results_save_path = os.path.join(base_dir, 'fed/results.csv')

grads_glob = []     # 参数，训练聚合时使用
loss_locals = []    # loss，训练聚合时使用
edge_count = 0
global_round = 0    # 当前全局模型所在轮数
choosen_this_round = [0]
m = 1

def init():
    """
    初始化服务端
    """
    global g_socket_server
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  
    sock.bind(CLOUD_ADDRESS_PORT)
    sock.listen(100)  # 最大等待数（有很多人理解为最大连接数，其实是错误的）
    # 配置TLS
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain("cert/cert.pem", "cert/key.pem")
    g_socket_server = context.wrap_socket(sock, server_side=True)
    print("server已启动，正在等待来自client的连接...")

def accept_client():
    global edge_count
    """
    接收新连接
    """
    while True:
        client, info = g_socket_server.accept()  # 阻塞
        # 注册client
        g_conn_pool[edge_count] = client
        # 给每个客户端创建一个独立的线程调用message_handle()进行管理
        thread = Thread(target=message_handle, args=(client, info, edge_count), daemon=True)
        thread.start()
        edge_count += 1

def message_handle(client, info, edge_id):
    """
    消息处理，由accept_client()创建的线程进行调用
    """
    global dict_users_train, dict_users_test, global_round, net_glob, grads_glob, loss_locals,lr
    send_msg(client, "连接成功".encode("utf-8"))
    local_round = global_round # 加入联邦学习的初始化
    """先用cloud代替edge模拟发数据吧"""
    dict_user = {
        "train": dict_users_train[edge_id],
        "test": dict_users_test[edge_id]
    }
    send_msg(client, pickle.dumps(dict_user))
    while True:
        try:
            """
            每轮client先检查上一轮聚合已经完成，然后发送全局参数，检查自己是否被选择，然后等待被选中的edge训练完成后发送局部参数
            """
            # 检查上一轮是否已经完成
            while local_round != global_round:
                time.sleep(0.1)
                pass

            # 发送全局参数
            send_msg(client, pickle.dumps({
                "round": local_round,
                "state": net_glob.state_dict(),
                "lr":lr
                }))
            logging.debug(f"Round = {local_round} 向 edge_id = {edge_id} 发送全局变量和学习率")
            local_round += 1

            # 发送client此轮是否被选中
            if edge_id in choosen_this_round:
                send_msg(client, pickle.dumps({"choosen": True}))
                logging.debug(f"edge_id = {edge_id} 被选择")
                # 接受来自边缘服务器的参数和loss
                data = pickle.loads(recv_msg(client))
                logging.debug(f"接收到来自 edge_id = {edge_id} 的 grads&loss")
                # 加入参数
                grads_local = data["grads_local"]
                if len(grads_glob) == 0:
                    grads_glob = copy.deepcopy(grads_local)
                else:
                    for level in range(len(grads_local)):
                        grads_glob[level] += grads_local[level]
                # 加入loss
                loss_local = data["loss_local"]
                loss_locals.append(loss_local)
                logging.debug(len(loss_locals))
            else:
                send_msg(client, pickle.dumps({"choosen": True}))  

        except Exception as e:
            print(e)
            remove_client(edge_id)
            break

def remove_client(edge_id):
    """
    移除client，由accept_client()创建的线程在进行message_handle()时进行调用
    """
    client = g_conn_pool[edge_id]
    if None != client:
        client.close()
        g_conn_pool.pop(edge_id)
        logging.info("client已离线：", edge_id)


if __name__ == '__main__':
    init()
    try:
        # 新创建一个线程，用于接收新连接
        thread = Thread(target=accept_client, daemon=True)
        thread.start()

        # 第一轮需要初始化
        m = max(1, min(args.num_users, edge_count) * args.frac)    # 选择client的数量
        choosen_this_round = np.random.choice(range(edge_count+1), m, replace=False)

        while global_round <= args.epochs:
            """
            自旋检查每轮是否完成，随机选择下一轮参与的client，在合适时间进行test和save
            """
            while len(loss_locals) != m:
                # logging.debug(len(loss_locals))
                time.sleep(0.1)

            # 只有当接受到所有被选择的client传来的参数，并完成聚合，选择下一轮参与的client才进入下一轮
            # learning rate 衰减
            lr *= args.lr_decay

            # 更新全局梯度
            """
            PEFL需要修改的部分——传入（累计梯度），输出模型参数
            """
            optimizer = torch.optim.SGD(net_glob.parameters(), lr=lr, momentum=args.momentum)
            optimizer.zero_grad()
            for level,para in enumerate(net_glob.parameters()):
                grads_glob[level] = torch.div(grads_glob[level], m)
                para.grad = grads_glob[level].to(args.device)
            logging.debug(f"Groud = {global_round} the grad of para[0] is {next(net_glob.parameters()).grad}")
            logging.debug(f"Before Step the para[0] is {next(net_glob.parameters())}")
            optimizer.step()
            logging.debug(f"After Step the para[0] is {next(net_glob.parameters())}")
            
            # 更新全局loss
            loss_avg = sum(loss_locals) / len(loss_locals)
            loss_train.append(loss_avg)
            
            # 选择下一轮参与的用户
            m = int(max(1, min(args.num_users, edge_count)*args.frac))    # 选择下一轮参与的用户
            choosen_this_round = np.random.choice(range(edge_count), m, replace=False)
            logging.info("Round = {:3d} 选择 {} 参与训练 lr = {:.3f}".format(global_round, choosen_this_round, lr))

            # test和save
            if (global_round + 1) % args.test_freq == 0:
                net_glob.eval()
                acc_test, loss_test = test_img(net_glob, dataset_test, args)
                logging.info('Round:{:3d}, Avg loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.3f}'.format(
                    global_round, loss_avg, loss_test, acc_test))

                if best_acc is None or acc_test > best_acc:
                    net_best = copy.deepcopy(net_glob)
                    best_acc = acc_test
                    best_epoch = global_round

                results.append(np.array([global_round, loss_avg, loss_test, acc_test, best_acc]))
                final_results = np.array(results)
                final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
                final_results.to_csv(results_save_path, index=False)

            if (global_round + 1) % 50 == 0:
                best_save_path = os.path.join(base_dir, 'fed/best_{}.pt'.format(global_round + 1))
                model_save_path = os.path.join(base_dir, 'fed/model_{}.pt'.format(global_round + 1))
                torch.save(net_best.state_dict(), best_save_path)
                torch.save(net_glob.state_dict(), model_save_path)

            # 进入下一轮
            grads_glob = []
            loss_locals = []
            logging.debug(f"进入下一轮： round = {global_round}")
            global_round += 1
            
        # 结束
        print('Best model, round: {}, acc: {}'.format(best_epoch, best_acc))

    finally:
        g_socket_server.close()