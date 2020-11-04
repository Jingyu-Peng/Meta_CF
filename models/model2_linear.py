
import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import utils.evaluation as evaluation
class model2_linear(torch.nn.Module):
    def __init__(self, args):
        super(model2_linear, self).__init__()
        self.args = args
        if args.dataset == 'movielens':
            item_embedding = np.load('./pretrain/movielens/item_pretrained.npy')
        else:
            item_embedding = np.load('./pretrain/amazon/item_pretrained.npy')
        self.lam_u = args.regs
        self.item_embeddings = torch.tensor(item_embedding)
        self.item_embeddings.requires_grad = False
        self.generate_layer = torch.nn.Linear(32, 32,bias=False).cuda()
        self.local_lr = self.args.local_lr
        self.global_lr = self.args.global_lr
        torch.nn.init.normal_(self.generate_layer.weight, 0,0.01)
        self.lossfunction = torch.nn.BCEWithLogitsLoss(reduction='sum')

    def loss_func(self, predicted, label_list, user_embedding):
        loss = self.lossfunction(predicted[0].cuda(), torch.tensor(label_list).cuda()) + self.lam_u * torch.sum(
            user_embedding.norm(dim=1))
        return loss
    def predict(self,user_embedding,itm_lst):
        return torch.mm(user_embedding.cuda(), self.item_embeddings[itm_lst].t().cuda()).view(-1)
    def generate_embedding(self,pos_embedding):
        itm_history_embedding = pos_embedding.cuda()
        user_embedding = self.generate_layer(itm_history_embedding)
        return user_embedding
    def global_forward(self, supp_x, supp_y, query_x, query_y):
        loss_sum = 0.0
        grad_w_sum = torch.zeros((32, 32))
        keep_weight = self.state_dict()
        for idx in range(len(supp_x)):
            step_loss = 0.0
            usrs = supp_x[idx][0]
            itms = supp_x[idx][1]
            label_list = supp_y[idx]
            pos_embedding = torch.zeros(1, 32).cuda()
            u = usrs[0]
            if self.args.dataset=='movielens':
                for i in range(1+u%10):
                    pos_embedding += self.item_embeddings[itms[i*(self.args.num_negative_support+1)]].cuda()
                pos_embedding = pos_embedding / (1+u%10)
                itm_history_embedding = pos_embedding.cuda()
                user_embedding = self.generate_layer(itm_history_embedding)
                grad1_w = torch.autograd.grad(user_embedding.sum(), self.generate_layer.weight, retain_graph=True)[0]
                for epoch in range(self.args.local_epoch):
                    user_embedding.detach_()
                    user_embedding.requires_grad = True
                    predicted = torch.mm(user_embedding.cuda(), self.item_embeddings[itms].t().cuda())
                    loss = self.loss_func(predicted, label_list, user_embedding)
                    loss.backward()
                    user_embedding = user_embedding - user_embedding.grad * self.local_lr
                    user_embedding.grad=None
                user_embedding.detach_()
                user_embedding.requires_grad = True
                query_itms = query_x[idx][1]
                query_labels = query_y[idx]
                q_predicted = torch.mm(user_embedding.cuda(), self.item_embeddings[query_itms].t().cuda())
                q_loss = self.loss_func(q_predicted, query_labels, user_embedding)
                loss_sum = loss_sum + q_loss
                q_loss.backward()
                tmp_grad = user_embedding.grad
                grad2_w = tmp_grad.repeat(32, 1)
                grad_w = torch.mm(grad2_w.t(), grad1_w).cuda()
                grad_w_sum = grad_w_sum.cuda() + grad_w
                self.load_state_dict(keep_weight)
                user_embedding.grad=None
            else:
                num_pos = len(label_list) // (self.args.num_negative_support + 1)
                for i in range(num_pos):
                    pos_embedding += self.item_embeddings[itms[i * (self.args.num_negative_support + 1)]].cuda()
                pos_embedding = pos_embedding / num_pos
                itm_history_embedding = pos_embedding.cuda()
                user_embedding = self.generate_layer(itm_history_embedding)
                grad1_w = torch.autograd.grad(user_embedding.sum(), self.generate_layer.weight, retain_graph=True)[0]
                for epoch in range(self.args.local_epoch):
                    user_embedding.detach_()
                    user_embedding.requires_grad = True
                    predicted = torch.mm(user_embedding.cuda(), self.item_embeddings[itms].t().cuda())
                    loss = self.loss_func(predicted, label_list, user_embedding)
                    loss.backward()
                    user_embedding = user_embedding - user_embedding.grad * self.local_lr
                    user_embedding.grad = None
                user_embedding.detach_()
                user_embedding.requires_grad = True
                query_itms = query_x[idx][1]
                query_labels = query_y[idx]
                q_predicted = torch.mm(user_embedding.cuda(), self.item_embeddings[query_itms].t().cuda())
                q_loss = self.loss_func(q_predicted, query_labels, user_embedding)
                loss_sum = loss_sum + q_loss
                q_loss.backward()
                tmp_grad = user_embedding.grad
                grad2_w = tmp_grad.repeat(32, 1)
                grad_w = torch.mm(grad2_w.t(), grad1_w).cuda()
                grad_w_sum = grad_w_sum.cuda() + grad_w
                self.load_state_dict(keep_weight)
                user_embedding.grad = None
        tmp_state_dict = self.state_dict()
        tmp_state_dict['generate_layer.weight'] = tmp_state_dict['generate_layer.weight'] - self.global_lr * grad_w
        self.load_state_dict(tmp_state_dict)
        return loss_sum

    def evaluate(self, train_support_fill, val_negative_dict, val_negative):
        keep_weight = deepcopy(self.state_dict())
        val_hr_list, val_ndcg_list, train_loss, val_loss = [], [], [], []
        train_support_loss = []
        for (idx, value) in train_support_fill.items():
            u, user_list, item_list, label_list = value['user'], value['user_list'], value['item_list'], value[
                'label_list']
            pos_embedding = torch.zeros(1, 32).cuda()
            if self.args.dataset=='movielens':
                for i in range(1+u%10):
                    pos_embedding += self.item_embeddings[item_list[i*(self.args.num_negative_support+1)]].cuda()
                pos_embedding = pos_embedding / (1+u%10)
                itm_history_embedding = pos_embedding.cuda()
                user_embedding = self.generate_layer(itm_history_embedding)
                for _ in range(self.args.local_epoch):
                    user_embedding.detach_()
                    user_embedding.requires_grad = True
                    predicted = torch.torch.mm(user_embedding.cuda(), self.item_embeddings[item_list].t().cuda())
                    loss = self.loss_func(predicted, label_list, user_embedding)
                    loss.backward()
                    user_embedding = user_embedding - user_embedding.grad * self.local_lr
                    user_embedding.grad=None
                hr, ndcg = evaluation.metrics_meta(self, user_embedding, u, val_negative_dict, val_negative,self.args.topK)
                val_hr_list.extend(hr)
                val_ndcg_list.extend(ndcg)
                self.load_state_dict(keep_weight)
            else:
                num_pos = len(label_list) // (self.args.num_negative_support + 1)
                for i in range(num_pos):
                    pos_embedding += self.item_embeddings[item_list[i * (self.args.num_negative_support + 1)]].cuda()
                pos_embedding = pos_embedding / num_pos
                itm_history_embedding = pos_embedding.cuda()
                user_embedding = self.generate_layer(itm_history_embedding)
                for _ in range(self.args.local_epoch):
                    user_embedding.detach_()
                    user_embedding.requires_grad = True
                    predicted = torch.torch.mm(user_embedding.cuda(), self.item_embeddings[item_list].t().cuda())
                    loss = self.loss_func(predicted, label_list, user_embedding)
                    loss.backward()
                    user_embedding = user_embedding - user_embedding.grad * self.local_lr
                    user_embedding.grad = None
                hr, ndcg = evaluation.metrics_meta(self, user_embedding, u, val_negative_dict, val_negative, self.args.topK)
                val_hr_list.extend(hr)
                val_ndcg_list.extend(ndcg)
                self.load_state_dict(keep_weight)

        return np.mean(val_hr_list), np.mean(val_ndcg_list)

