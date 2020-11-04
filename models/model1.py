import torch
import torch.nn as nn
import utils.evaluation as evaluation
from time import time
from copy import deepcopy
import numpy as np
def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()

class user_preference_estimator(torch.nn.Module):
    def __init__(self, item_embeddings,args):
        super().__init__()
        self.lam_u = args.regs
        self.loss_func = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.user_embeddings = torch.nn.Parameter(torch.mul(torch.randn(1, 32),0.01).cuda())
        self.user_embeddings.requires_grad = True
        self.item_embeddings = torch.tensor(item_embeddings).cuda()
        self.item_embeddings.requires_grad = False
    def forward(self, item, matrix):
        predicted = torch.mm(self.user_embeddings, self.item_embeddings[item].t())
        prediction_error = self.loss_func(predicted[0], matrix)
        u_regularization = self.lam_u * self.user_embeddings.norm()
        return prediction_error + u_regularization
    def predict(self, item):
        predicted = torch.mm(self.user_embeddings, self.item_embeddings[item].t())
        return predicted.view(-1)


class model1(torch.nn.Module):
    def __init__(self,args):
        super(model1, self).__init__()
        self.args = args
        if args.dataset=='movielens':
            itm_embedding = np.load('./pretrain/movielens/item_pretrained.npy')
        else:
            itm_embedding = np.load('./pretrain/amazon/item_pretrained.npy')
        self.item_embeddings = torch.tensor(itm_embedding)
        self.item_embeddings.requires_grad = False
        self.model = user_preference_estimator(itm_embedding,args).cuda()
        self.local_optim = torch.optim.SGD(self.model.parameters(), lr=args.local_lr)
        self.meta_optim = torch.optim.SGD(self.model.parameters(), lr=args.global_lr)
    def predict(self,user_embedding,itm_lst):
        return torch.mm(user_embedding.cuda(), self.item_embeddings[itm_lst].t().cuda()).view(-1)
    def global_forward(self, support_x, support_y, query_x, query_y):
        t1 = time()
        keep_weight = deepcopy(self.model.state_dict())
        losses_query = []
        for idx in range(len(support_x)):
            self.model.load_state_dict(keep_weight)
            for _ in range(self.args.local_epoch):
                loss_support = self.model.forward(support_x[idx][1], torch.FloatTensor(support_y[idx]).cuda())
                self.local_optim.zero_grad()
                loss_support.backward()
                self.local_optim.step()
            loss_query = self.model.forward(query_x[idx][1], torch.FloatTensor(query_y[idx]).cuda())
            losses_query.append(loss_query)
            self.model.load_state_dict(keep_weight)
        losses_query = torch.stack(losses_query).sum()
        self.meta_optim.zero_grad()
        losses_query.backward()
        self.meta_optim.step()
        t2 = time()
        return losses_query

    def evaluate(self, train_support_fill, val_negative_dict, val_negative):
        keep_weight = deepcopy(self.model.state_dict())
        hr_list, ndcg_list = [], []
        tmp_train_loss = []
        for idx in train_support_fill:

            u, user_list, item_list, label_list = train_support_fill[idx]['user'], train_support_fill[idx]['user_list'], \
                                                  train_support_fill[idx]['item_list'], train_support_fill[idx][
                                                      'label_list']

            for _ in range(self.args.local_epoch):
                # first step optimization
                loss = self.model(item_list, torch.FloatTensor(label_list).cuda())
                self.local_optim.zero_grad()
                loss.backward()
                self.local_optim.step()
            tmp_train_loss.append(tensorToScalar(loss))
            hr, ndcg = evaluation.metrics_meta(self,self.model.user_embeddings,u, val_negative_dict, val_negative, self.args.topK)
            hr_list.extend(hr)
            ndcg_list.extend(ndcg)
            self.model.load_state_dict(keep_weight)
        return np.mean(hr_list), np.mean(ndcg_list)




