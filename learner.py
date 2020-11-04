
from models.model1 import model1
from models.model2_linear import model2_linear
from models.model2_neural import model2_neural
from models.model3_linear import model3_linear
import torch.utils.data as data
import torch
import numpy as np
from models.model3_neural import model3_neural
from utils.Logging import Logging
import utils.data_prepare as data_prepare
def tensorToScalar(tensor):
    return tensor.cpu().detach().numpy()
class meta_learner():
    def __init__(self,args):

        if args.mod=='train':
            self.args = args
            self.log_str_path = './train_log/' + args.model + str(args.number) + args.mod + args.dataset +'GL'+str(args.global_lr)+'LL'+str(args.local_lr)+'LE'+str(args.local_epoch)+ '.log'
            self.mod_str_path = './saved_models/' + args.model + str(args.number) + args.mod + args.dataset+'GL'+str(args.global_lr)+'LL'+str(args.local_lr)+'LE'+str(args.local_epoch)
            self.log = Logging(self.log_str_path)
            eval_ = args.model + "(args)"
            self.model = eval(eval_)
            self.user_num, self.item_num, self.train_data, self.train_support_data, self.train_support_mat, self.val_negative_dict, self.val_negative, self.train_mat, self.val_mat = data_prepare.load_all(
                self.args)
            self.Train_data = data_prepare.Train_data(self.train_data, self.item_num, self.train_support_mat, self.args)
            self.Train_data.ng_train_sample()
            self.maml_train_batch_sampler = data.BatchSampler(data.RandomSampler(range(self.Train_data.idx)),
                                                              batch_size=self.args.batch_size,
                                                              drop_last=False)

        else:
            self.args = args
            self.log_str_path = './logs/' + args.model + str(args.number) + args.mod + args.dataset + '.log'
            self.mod_str_path = './saved_models/' + args.model + str(args.number) + args.mod + args.dataset
            self.log = Logging(self.log_str_path)
            eval_ = args.model + "(args)"
            self.model = eval(eval_)
            self.user_num, self.item_num, self.train_data, self.train_support_data, self.train_support_mat, self.val_negative_dict, \
            self.val_negative, self.train_mat, self.val_mat, self.train_query_negative, self.train_query_negative_dict, self.test_support_data, self.test_negative_dict, self.test_negative, self.test_mat, self.test_support_mat, self.test_rating_data = data_prepare.load_all(self.args)
            self.Train_data = data_prepare.Train_data(self.train_data,self.item_num,self.train_support_mat,self.args)
            self.Test_as_train_data = data_prepare.Train_data(self.test_rating_data,self.item_num,self.test_support_data,self.args)
            self.Train_data.ng_train_sample()
            self.Test_as_train_data.ng_train_sample()
            self.maml_train_batch_sampler = data.BatchSampler(data.RandomSampler(range(self.Train_data.idx)),
                                                         batch_size=self.args.batch_size,
                                                         drop_last=False)
    def exp(self):
        max_hr, max_ndcg = 0.0, 0.0
        val_hr, val_ndcg = self.model.evaluate(self.Train_data.train_support_fill, self.val_negative_dict,
                                            self.val_negative)
        train_query_hr, train_query_ndcg = self.model.evaluate(self.Train_data.train_support_fill,
                                                            self.train_query_negative_dict, self.train_query_negative)
        test_query_hr, test_query_ndcg = self.model.evaluate(self.Test_as_train_data.train_support_fill,
                                                          self.test_negative_dict, self.test_negative)
        str_to_log = '---epoch:{},train_query_loss:{},train_query_hr:{},train_query_ndcg:{},val_hr:{},val_ndcg:{},test_query_hr:{},test_query_ndcg:{}'.format(
            0, 0, train_query_hr, train_query_ndcg, val_hr, val_ndcg, test_query_hr, test_query_ndcg)
        self.log.record(str_to_log)
        for epoch in range(self.args.train_epoch):
            total_loss = 0.0
            for batch_idx_list in self.maml_train_batch_sampler:
                _, supp_x_batch, supp_y_batch, query_x_batch, query_y_batch \
                    = self.Train_data.get_train_batch(batch_idx_list)
                step_loss = self.model.global_forward(supp_x_batch, supp_y_batch, query_x_batch, query_y_batch)
                total_loss += step_loss

            val_hr,val_ndcg = self.model.evaluate(self.Train_data.train_support_fill, self.val_negative_dict, self.val_negative)
            train_query_hr,train_query_ndcg = self.model.evaluate(self.Train_data.train_support_fill, self.train_query_negative_dict, self.train_query_negative)
            test_query_hr,test_query_ndcg = self.model.evaluate(self.Test_as_train_data.train_support_fill,self.test_negative_dict,self.test_negative)
            str_to_log = '---epoch:{},train_query_loss:{},train_query_hr:{},train_query_ndcg:{},val_hr:{},val_ndcg:{},test_query_hr:{},test_query_ndcg:{}'.format(epoch,total_loss,train_query_hr,train_query_ndcg,val_hr,val_ndcg,test_query_hr,test_query_ndcg)
            self.log.record(str_to_log)
            if val_hr>max_hr or val_ndcg>max_ndcg:
                if self.args.model=='model1':
                    np.save(self.mod_str_path+'.npy' ,tensorToScalar(self.model.model.user_embeddings))
                elif self.args.model=='model2_linear' or self.args.model=='model2_neural' or self.args.model=='model3_linear':
                    torch.save(self.model.state_dict(),self.mod_str_path+'.mod')
                else:
                    torch.save(self.model.state_dict(),self.mod_str_path+'.mod')
                    np.save(self.mod_str_path+'shared_emb.npy',tensorToScalar(self.model.share_embedding))
            max_hr = max(max_hr,val_hr)
            max_ndcg = max(max_ndcg,val_ndcg)



    def train(self):
        max_hr, max_ndcg = 0.0, 0.0
        val_hr, val_ndcg = self.model.evaluate(self.Train_data.train_support_fill, self.val_negative_dict,
                                            self.val_negative)
        str_to_log = '---epoch:{},train_query_loss:{},val_hr:{},val_ndcg:{}'.format(
            0, 0, val_hr, val_ndcg)
        self.log.record(str_to_log)
        for epoch in range(self.args.train_epoch):
            total_loss = 0.0
            for batch_idx_list in self.maml_train_batch_sampler:
                _, supp_x_batch, supp_y_batch, query_x_batch, query_y_batch \
                    = self.Train_data.get_train_batch(batch_idx_list)
                step_loss = self.model.global_forward(supp_x_batch, supp_y_batch, query_x_batch, query_y_batch)
                total_loss += step_loss

            val_hr,val_ndcg = self.model.evaluate(self.Train_data.train_support_fill, self.val_negative_dict, self.val_negative)
            str_to_log = '---epoch:{},train_query_loss:{},val_hr:{},val_ndcg:{}'.format(epoch,total_loss,val_hr,val_ndcg)
            self.log.record(str_to_log)
            if val_hr>max_hr or val_ndcg>max_ndcg:
                if self.args.model=='model1':
                    np.save(self.mod_str_path+'.npy' ,tensorToScalar(self.model.model.user_embeddings))
                elif self.args.model=='model2_linear' or self.args.model=='model2_neural' or self.args.model=='model3_linear':
                    torch.save(self.model.state_dict(),self.mod_str_path+'.mod')
                else:
                    torch.save(self.model.state_dict(),self.mod_str_path+'.mod')
                    np.save(self.mod_str_path+'shared_emb.npy',tensorToScalar(self.model.share_embedding))
            max_hr = max(max_hr,val_hr)
            max_ndcg = max(max_ndcg,val_ndcg)




