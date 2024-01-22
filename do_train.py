import torch
from pretrainers.pretrain_proto4 import pretrain_proto as pretrain_proto4
from trainers.train_proto2 import train_proto as train_proto2


if __name__ == '__main__':
    print(torch.cuda.is_available())
    pretrain_proto4('config/bace_proto_test.yaml')
    train_proto2('config/bace_proto_test.yaml')
