from  models.model1 import model1
from  models.model2_linear import model2_linear
from  models.model2_neural import model2_neural
from  models.model3_linear import model3_linear
import torch
import utils.evaluation as evaluation
from utils.Logging import *
import torch.nn as nn
from copy import deepcopy
from models.model3_neural import model3_neural
import numpy as np
import utils.data_prepare as data_prepare
class PMFLoss(torch.nn.Module):
    def __init__(self,args):
        super().__init__()
        self.lam_u = args.regs
        if args.dataset=='movielens':
            self.user_num=6040
            self.item_num=3952
        else:
            self.user_num=27879
            self.item_num=9449
        self.user_embedding = torch.randn(self.user_num, 32).cuda()
        self.user_embedding = torch.mul(self.user_embedding, 0.1)
        self.lossfunc = torch.nn.BCEWithLogitsLoss(reduction='sum')
        self.item_embedding = torch.randn(self.item_num, 32).cuda()
        self.user_embedding.requires_grad = True
        self.item_embedding.requires_grad = False

    def forward(self, user, item, ratings):
        ratings = torch.tensor(ratings).cuda()
        predicted = torch.mm(self.user_embedding[user].cuda(), self.item_embedding[item].t().cuda())
        u_regularization = self.lam_u * torch.sum(self.user_embedding[user])
        return self.lossfunc(predicted[0].cuda(), torch.tensor(ratings).cuda()) + u_regularization

    def predict(self, user, item):
        predicted = nn.Sigmoid()(torch.mm(self.user_embedding[user].cuda(), self.item_embedding[item].t().cuda()))
        return predicted.cuda()
def test(args):
    user_num, item_num, test_support_data, test_negative_dict, test_negative, test_mat, test_support_mat = data_prepare.load_all(
        args)
    itm_emb_pretrained = torch.tensor(np.load('./pretrain/'+args.dataset+'/item_pretrained.npy'))
    test_data = data_prepare.Test_data(test_support_data, item_num, test_support_mat,args)
    test_data.ng_test_sample()
    log_str_path = './test_log/hr'+str(args.topK)+'/' + args.model + str(args.number) + 'test' + args.dataset +'GL'+str(args.global_lr)+'LL'+str(args.local_lr)+'LE'+str(args.local_epoch)+ '.log'
    mod_str_path = './saved_models/' + args.model + str(args.number) + 'train' + args.dataset+'GL'+str(args.global_lr)+'LL'+str(args.local_lr)+'LE'+str(args.local_epoch)
    log = Logging(log_str_path)
    if args.model=='baseline':
        log_str_path = './test_log/' +str(args.topK)+'/' + args.model + str(args.number) + 'test' + args.dataset +'LL'+str(args.local_lr)+'LE'+ '.log'
        pmf = PMFLoss(args).cuda()
        pmf.user_embedding.requires_grad = True
        pmf.item_embedding = torch.tensor(itm_emb_pretrained).cuda()
        pmf.item_embedding.requires_grad = False
        args.local_epoch=150
    elif args.model=='model1':
        usr_emb = np.load(mod_str_path+'.npy' )
        usr = []
        for i in range(user_num):
            usr.append(deepcopy(usr_emb[0]))
        pmf = PMFLoss(args).cuda()
        pmf.user_embedding = torch.tensor(usr).cuda()
        pmf.user_embedding.requires_grad = True
        pmf.item_embedding = torch.tensor(itm_emb_pretrained).cuda()
        pmf.item_embedding.requires_grad = False
    else:
        eval_ = args.model + "(args)"
        model = eval(eval_)
        mod = torch.load(mod_str_path+'.mod')
        model.load_state_dict(mod)
        if args.model=='model3_neural':
            model.share_embedding = torch.tensor(np.load(mod_str_path+'shared_emb.npy',allow_pickle=True))
        test_embeddings = torch.zeros(user_num, 32)
        for (idx, value) in test_data.test_support_fill.items():
            u, user_list, item_list, label_list = value['user_list'][0], value['user_list'], value['item_list'], \
                                                  value[
                                                      'label_list']
            pos_embedding = torch.zeros(1, 32).cuda()
            if args.dataset=='movielens':
                for i in range(1 + u % 10):
                    pos_embedding += itm_emb_pretrained[item_list[i * (args.num_negative_support + 1)]].cuda()
                pos_embedding = pos_embedding / (1 + u % 10)
            else:
                num_pos = len(label_list) // (args.num_negative_support + 1)
                for i in range(num_pos):
                    pos_embedding += itm_emb_pretrained[item_list[i * (args.num_negative_support + 1)]].cuda()
                pos_embedding = pos_embedding / (num_pos)
            test_embeddings[u] = model.generate_embedding(pos_embedding)
        pmf = PMFLoss(args).cuda()
        pmf.user_embedding = torch.tensor(test_embeddings).cuda()
        pmf.user_embedding.requires_grad = True
        pmf.item_embedding = torch.tensor(itm_emb_pretrained).cuda()
        pmf.item_embedding.requires_grad = False

    optim = torch.optim.SGD([pmf.user_embedding], lr=args.local_lr)
    if args.dataset=='amazon':
        hrs, ndcgs, losses = [], [], []
        hr, ndcg = evaluation.metrics_test(pmf, test_negative_dict, test_negative, test_data.sup_dict,args.topK)
        hr_tmp, ndcg_tmp = [], []
        hr_tmp_m, ndcg_tmp_m = [], []  # for sup size >10
        for i in hr:
            if i <= 10:
                log.record('epoch0,size{},hr{},ndcg{},len{}'.format(i, np.mean(hr[i]), np.mean(ndcg[i]), len(hr[i])))
                hr_tmp.extend(hr[i])
                ndcg_tmp.extend(ndcg[i])
            else:
                hr_tmp_m.extend(hr[i])
                ndcg_tmp_m.extend(ndcg[i])
        log.record('epoch0,size>10,hr{},ndcg{},len{}'.format(np.mean(hr_tmp_m), np.mean(ndcg_tmp_m), len(hr_tmp_m)))
        hr_tmp.extend(hr_tmp_m)
        ndcg_tmp.extend(ndcg_tmp_m)
        hrs.append(np.mean(hr_tmp))
        ndcgs.append(np.mean(ndcg_tmp))
        log.record('=============epoch{},loss{},hr{},ndcg{}==============='.format(0, 0, hrs[-1], ndcgs[-1]))

        for epoch in range(1, args.local_epoch+5):
            loss_epoch = []
            pmf.train()
            for u in test_data.test_support_fill:
                u, user_list, item_list, label_list = test_data.test_support_fill[u]['user_list'][0], \
                                                      test_data.test_support_fill[u]['user_list'], \
                                                      test_data.test_support_fill[u]['item_list'], \
                                                      test_data.test_support_fill[u][
                                                          'label_list']
                loss = pmf.forward(user_list, item_list, label_list)
                loss_epoch.append(loss)
            tmploss = np.sum(loss_epoch)
            optim.zero_grad()
            tmploss.backward()
            optim.step()
            losses.append(tmploss)
            pmf.eval()

            hr, ndcg = evaluation.metrics_test(pmf, test_negative_dict, test_negative, test_data.sup_dict,args.topK)
            hr_tmp, ndcg_tmp = [], []
            hr_tmp_m, ndcg_tmp_m = [], []  # for sup size >10
            for i in hr:
                if i <= 10:
                    log.record('epoch{},size{},hr{},ndcg{},len{}'.format(epoch, i, np.mean(hr[i]), np.mean(ndcg[i]),
                                                                         len(hr[i])))
                    hr_tmp.extend(hr[i])
                    ndcg_tmp.extend(ndcg[i])
                else:
                    hr_tmp_m.extend(hr[i])
                    ndcg_tmp_m.extend(ndcg[i])
            log.record('epoch{},size>10,hr{},ndcg{},len{}'.format(epoch, np.mean(hr_tmp_m), np.mean(ndcg_tmp_m),
                                                                  len(hr_tmp_m)))
            hr_tmp.extend(hr_tmp_m)
            ndcg_tmp.extend(ndcg_tmp_m)
            hrs.append(np.mean(hr_tmp))
            ndcgs.append(np.mean(ndcg_tmp))
            log.record(
                '=============epoch{},loss{},hr{},ndcg{}==============='.format(epoch, tmploss, hrs[-1], ndcgs[-1]))
    else:
        hrs, ndcgs, losses = [], [], []
        hr, ndcg = evaluation.metrics_test(pmf, test_negative_dict, test_negative,test_data.sup_dict,args.topK)
        hr_tmp, ndcg_tmp = [], []
        for i in range(1, 11):
            log.record('epoch0,size{},hr{},ndcg{},len{}'.format(i, np.mean(hr[i]), np.mean(ndcg[i]), len(hr[i])))
            hr_tmp.extend(hr[i])
            ndcg_tmp.extend(ndcg[i])
        hrs.append(np.mean(hr_tmp))
        ndcgs.append(np.mean(ndcg_tmp))
        log.record('=============epoch{},loss{},hr{},ndcg{}==============='.format(0, 0, hrs[-1], ndcgs[-1]))

        for epoch in range(1, args.local_epoch+5):
            loss_epoch = []
            pmf.train()
            for u in test_data.test_support_fill:
                u, user_list, item_list, label_list = test_data.test_support_fill[u]['user_list'][0], \
                                                      test_data.test_support_fill[u]['user_list'], \
                                                      test_data.test_support_fill[u]['item_list'], \
                                                      test_data.test_support_fill[u][
                                                          'label_list']
                loss = pmf.forward(user_list, item_list, label_list)
                loss_epoch.append(loss)
            tmploss = np.sum(loss_epoch)
            optim.zero_grad()
            tmploss.backward()
            optim.step()
            losses.append(tmploss)
            pmf.eval()

            hr, ndcg = evaluation.metrics_test(pmf, test_negative_dict, test_negative,test_data.sup_dict, args.topK)
            hr_tmp, ndcg_tmp = [], []
            for i in range(1, 11):
                log.record(
                    'epoch{},size{},hr{},ndcg{},len{}'.format(epoch, i, np.mean(hr[i]), np.mean(ndcg[i]), len(hr[i])))
                hr_tmp.extend(hr[i])
                ndcg_tmp.extend(ndcg[i])

            hrs.append(np.mean(hr_tmp))
            ndcgs.append(np.mean(ndcg_tmp))
            log.record(
                '=============epoch{},loss{},hr{},ndcg{}==============='.format(epoch, tmploss, hrs[-1], ndcgs[-1]))

