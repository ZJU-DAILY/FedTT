from model.pdformer_model import traffic_flow_prediction
from model.pdformer_model.dataset import PDFormerDataset
from model.pdformer_model.pdformer_config import ConfigParser
# from federatedscope.core.configs.config import CN

from torch import nn


class PDFormer(nn.Module):
    def __init__(self, args):
        super(PDFormer, self).__init__()
        self.model = traffic_flow_prediction.PDFormer(args, args.data_feature, args.pre_len)

    def forward(self, x):  # B x T x N x D
        return self.model(x)

# def load_my_net(local_data):
#     pdformer_model_config = ConfigParser(
#         task="traffic_state_pred", model="PDFormer", dataset="PeMS04", data_config=local_data[3],
#         config_file=local_data[3].type).config
#     dataset = PDFormerDataset(pdformer_model_config, local_data[1], local_data[4], local_data[2])
#     dataset.get_data()
#     data_feature = dataset.get_data_feature()
#     return PDFormer(pdformer_model_config, data_feature)


# def call_my_net(model_config, local_data):  # data, geo_ids, worker_id, data_config
#     if model_config.type == "pdformer":
#         return load_my_net(local_data)


# register_model("pdformer", call_my_net)
