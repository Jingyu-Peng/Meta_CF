import torch
import torch.nn as nn
from copy import deepcopy
import numpy as np
from time import time
import utils.evaluation as evaluation
class model3_neural(torch.nn.Module):
    def __init__(self, args):
        super(model3_neural, self).__init__()
        self.args = args
        if args.dataset == 'movielens':
            self.item_embeddings = np.load('./pretrain/movielens/item_pretrained.npy')
        else:
            self.item_embeddings = np.load('./pretrain/amazon/item_pretrained.npy')
        self.lam_u = args.regs
        self.item_embeddings = torch.tensor(self.item_embeddings)
        self.item_embeddings.requires_grad = False
        self.global_lr_for_share_embedding = args.global_lr
        self.generate_layer1 = torch.nn.Linear(64, 32).cuda()
        self.relu1 = nn.ReLU ()
        self.generate_layer2 = torch.nn.Linear(32, 32).cuda()
        self.local_lr = args.local_lr
        self.global_lr = args.global_lr
        self.share_embedding =  torch.mul(torch.randn(1,32),0.01).cuda()
        self.share_embedding.requires_grad = True
        torch.nn.init.normal_(self.generate_layer1.weight, 0, 0.01)
        torch.nn.init.normal_(self.generate_layer2.weight, 0, 0.01)
        torch.nn.init.normal_(self.generate_layer1.bias, 0, 0.01)
        torch.nn.init.normal_(self.generate_layer2.bias, 0, 0.01)
        self.lossfunction = torch.nn.BCEWithLogitsLoss(reduction='sum')
    def predict(self,user_embedding,itm_lst):
        return torch.mm(user_embedding.cuda(), self.item_embeddings[itm_lst].t().cuda()).view(-1)
    def loss_func(self, predicted, label_list, user_embedding):
        loss = self.lossfunction(predicted[0].cuda(), torch.tensor(label_list).cuda()) + self.lam_u * torch.sum(
            user_embedding.norm(dim=1))
        return loss
    def generate_embedding(self,pos_embedding):
        itm_history_embedding = pos_embedding.cuda()
        user_embedding_init = torch.cat((itm_history_embedding, self.share_embedding.cuda()), 1)
        user_embedding = self.generate_layer1(user_embedding_init)
        user_embedding = self.relu1(user_embedding)
        user_embedding = self.generate_layer2(user_embedding)
        return user_embedding
    def global_forward(self, supp_x, supp_y, query_x, query_y):
        t1s=[]
        t2s=[]
        t3s = []
        loss_sum = 0.0
        grad_w1_sum = torch.zeros((32, 64)).cuda()
        grad_w2_sum = torch.zeros((32, 32)).cuda()
        grad_b1_sum = torch.zeros((1,32)).cuda()
        grad_b2_sum = torch.zeros((1,32)).cuda()
        grad_share_sum = torch.zeros((1,32)).cuda()
        keep_weight = self.state_dict()
        for idx in range(len(supp_x)):
            t1 = time()
            step_loss = 0.0
            usrs = supp_x[idx][0]
            itms = supp_x[idx][1]
            label_list = supp_y[idx]
            pos_embedding = torch.zeros(1, 32).cuda()
            u = usrs[0]
            if self.args.dataset=='movielens':
                for i in range(1 + u % 10):
                    pos_embedding += self.item_embeddings[itms[i * (self.args.num_negative_support + 1)]].cuda()
                pos_embedding = pos_embedding / (1 + u % 10)
                itm_history_embedding = pos_embedding.cuda()
                user_embedding_init = torch.cat((itm_history_embedding,self.share_embedding.cuda()),1)
                user_embedding = self.generate_layer1(user_embedding_init)
                user_embedding = self.relu1(user_embedding)
                user_embedding = self.generate_layer2(user_embedding)

                grad_w1s = []
                grad_b1s = []
                grad_w2s = []
                grad_b2s = []

                grad_share = []
                t2 = time()
                t1s.append(t2-t1)
                for i in range(32):
                    user_embedding[0][i].backward(retain_graph = True)
                    grad_w1s.append(deepcopy(self.generate_layer1.weight.grad))
                    grad_b1s.append(deepcopy(self.generate_layer1.bias.grad))
                    grad_w2s.append(deepcopy(self.generate_layer2.weight.grad))
                    grad_b2s.append(deepcopy(self.generate_layer2.bias.grad))
                    grad_share.append(deepcopy(self.share_embedding.grad))
                    self.generate_layer1.weight.grad.data.zero_()
                    self.generate_layer1.bias.grad.data.zero_()
                    self.generate_layer2.weight.grad.data.zero_()
                    self.generate_layer2.bias.grad.data.zero_()
                    self.share_embedding.grad.data.zero_()

                t3 = time()
                t2s.append(t3 - t2)
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
                for i in range(32):
                    grad_w1_sum = grad_w1_sum+ torch.mul(grad_w1s[i],tmp_grad[0][i])
                    grad_b1_sum = grad_b1_sum+ torch.mul(grad_b1s[i],tmp_grad[0][i])
                    grad_w2_sum = grad_w2_sum + torch.mul(grad_w2s[i],tmp_grad[0][i])
                    grad_b2_sum = grad_b2_sum + torch.mul(grad_b2s[i],tmp_grad[0][i])
                    grad_share_sum = grad_share_sum + torch.mul(grad_share[i],tmp_grad[0][i])
                t4 = time()
                t3s.append(t4 - t3)
                user_embedding.grad=None
            else:
                num_pos = len(label_list) // (self.args.num_negative_support + 1)
                for i in range(num_pos):
                    pos_embedding += self.item_embeddings[itms[i * (self.args.num_negative_support + 1)]].cuda()
                pos_embedding = pos_embedding / (num_pos)
                itm_history_embedding = pos_embedding.cuda()
                user_embedding_init = torch.cat((itm_history_embedding, self.share_embedding.cuda()), 1)
                user_embedding = self.generate_layer1(user_embedding_init)
                user_embedding = self.relu1(user_embedding)
                user_embedding = self.generate_layer2(user_embedding)

                grad_w1s = []
                grad_b1s = []
                grad_w2s = []
                grad_b2s = []

                grad_share = []
                t2 = time()
                t1s.append(t2 - t1)
                for i in range(32):
                    user_embedding[0][i].backward(retain_graph=True)
                    grad_w1s.append(deepcopy(self.generate_layer1.weight.grad))
                    grad_b1s.append(deepcopy(self.generate_layer1.bias.grad))
                    grad_w2s.append(deepcopy(self.generate_layer2.weight.grad))
                    grad_b2s.append(deepcopy(self.generate_layer2.bias.grad))
                    grad_share.append(deepcopy(self.share_embedding.grad))
                    self.generate_layer1.weight.grad.data.zero_()
                    self.generate_layer1.bias.grad.data.zero_()
                    self.generate_layer2.weight.grad.data.zero_()
                    self.generate_layer2.bias.grad.data.zero_()
                    self.share_embedding.grad.data.zero_()

                t3 = time()
                t2s.append(t3 - t2)
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
                for i in range(32):
                    grad_w1_sum = grad_w1_sum + torch.mul(grad_w1s[i], tmp_grad[0][i])
                    grad_b1_sum = grad_b1_sum + torch.mul(grad_b1s[i], tmp_grad[0][i])
                    grad_w2_sum = grad_w2_sum + torch.mul(grad_w2s[i], tmp_grad[0][i])
                    grad_b2_sum = grad_b2_sum + torch.mul(grad_b2s[i], tmp_grad[0][i])
                    grad_share_sum = grad_share_sum + torch.mul(grad_share[i], tmp_grad[0][i])
                t4 = time()
                t3s.append(t4 - t3)
                user_embedding.grad = None
        self.load_state_dict(keep_weight)
        tmp_state_dict = self.state_dict()
        tmp_state_dict['generate_layer1.weight'] = tmp_state_dict['generate_layer1.weight'] - self.global_lr *grad_w1_sum
        tmp_state_dict['generate_layer1.bias'] = tmp_state_dict['generate_layer1.bias'] - self.global_lr * grad_b1_sum.view(
            -1)
        tmp_state_dict['generate_layer2.weight'] = tmp_state_dict['generate_layer2.weight'] - self.global_lr * grad_w2_sum
        tmp_state_dict['generate_layer2.bias'] = tmp_state_dict['generate_layer2.bias'] - self.global_lr * grad_b2_sum.view(
            -1)

        self.load_state_dict(tmp_state_dict)
        self.share_embedding = self.share_embedding-self.global_lr_for_share_embedding * grad_share_sum
        self.share_embedding.detach_()
        self.share_embedding.requires_grad = True


        return loss_sum
    def evaluate(self, train_support_fill, val_negative_dict, val_negative):
        keep_weight = deepcopy(self.state_dict())
        val_hr_list, val_ndcg_list, train_loss, val_loss = [], [], [], []

        for (idx, value) in train_support_fill.items():
            u, user_list, item_list, label_list = value['user'], value['user_list'], value['item_list'], value[
                'label_list']
            pos_embedding = torch.zeros(1, 32).cuda()
            if self.args.dataset=='movielens':
                for i in range(1+u%10):
                    pos_embedding += self.item_embeddings[item_list[i*(self.args.num_negative_support+1)]].cuda()
                pos_embedding = pos_embedding / 3
                itm_history_embedding = pos_embedding.cuda()
                user_embedding =torch.cat((itm_history_embedding, self.share_embedding.cuda()),1)
                user_embedding = self.generate_layer1(user_embedding)
                user_embedding = self.relu1(user_embedding)
                user_embedding = self.generate_layer2(user_embedding)

                for _ in range(self.args.local_epoch):
                    user_embedding.detach_()
                    user_embedding.requires_grad = True
                    predicted = torch.mm(user_embedding.cuda(), self.item_embeddings[item_list].t().cuda())
                    loss = self.loss_func(predicted, label_list, user_embedding)
                    loss.backward()
                    user_embedding = user_embedding - user_embedding.grad * self.local_lr
                    user_embedding.grad=None
                hr, ndcg= evaluation.metrics_meta(self, user_embedding, u, val_negative_dict, val_negative, self.args.topK )
                val_hr_list.extend(hr)
                val_ndcg_list.extend(ndcg)
                self.load_state_dict(keep_weight)
            else:
                num_pos = len(label_list) // (self.args.num_negative_support + 1)
                for i in range(num_pos):
                    pos_embedding += self.item_embeddings[item_list[i * (self.args.num_negative_support + 1)]].cuda()
                pos_embedding = pos_embedding / (num_pos)
                itm_history_embedding = pos_embedding.cuda()
                user_embedding = torch.cat((itm_history_embedding, self.share_embedding.cuda()), 1)
                user_embedding = self.generate_layer1(user_embedding)
                user_embedding = self.relu1(user_embedding)
                user_embedding = self.generate_layer2(user_embedding)

                for _ in range(self.args.local_epoch):
                    user_embedding.detach_()
                    user_embedding.requires_grad = True
                    predicted = torch.mm(user_embedding.cuda(), self.item_embeddings[item_list].t().cuda())
                    loss = self.loss_func(predicted, label_list, user_embedding)
                    loss.backward()
                    user_embedding = user_embedding - user_embedding.grad * self.local_lr
                    user_embedding.grad = None
                hr, ndcg = evaluation.metrics_meta(self, user_embedding, u, val_negative_dict, val_negative, self.args.topK)
                val_hr_list.extend(hr)
                val_ndcg_list.extend(ndcg)
                self.load_state_dict(keep_weight)
        return np.mean(val_hr_list), np.mean(val_ndcg_list)
