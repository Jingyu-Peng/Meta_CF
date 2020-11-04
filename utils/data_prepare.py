import torch.utils.data as data
from copy import deepcopy
import numpy as np
from collections import defaultdict
import random
def load_all(args):
    file_name = './dataset/'+args.dataset+'/'
    if args.dataset=='movielens':
        user_num = 6040
        item_num = 3952
    else:
        user_num = 27879
        item_num = 9449
    if args.mod=='train':
        train_mat = {}
        train_support_mat, test_support_mat = {}, {}
        val_mat = {}
        test_mat = {}
        train_data, train_support_data = defaultdict(list), defaultdict(list)
        val_data = defaultdict(list)
        val_negative = []
        val_negative_dict = defaultdict(list)
        with open(file_name+'train.rating') as f:
            for line in f:
                arr = line.split('\t')
                u, v = int(arr[0]), int(arr[1])
                train_data[u].append(v)
                train_mat[(u, v)] = 1.0
        val_rating = []
        with open(file_name+'val_train.rating') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                u, v = int(arr[0]), int(arr[1])
                val_data[u].append(v)
                val_mat[(u, v)] = 1.0
                val_rating.append([u, v])
        for idx, (u, v) in enumerate(val_rating):
            val_negative_dict[u].append(idx)
            tmp_val_item_list = [v]
            # select negative samples for each user -- first strategy
            for _ in range(args.num_negative_evaluation):
                j = np.random.randint(0, item_num)
                while (u, j) in val_mat or (u, j) in train_mat:
                    j = np.random.randint(0, item_num)
                tmp_val_item_list.append(j)
            val_negative.append(tmp_val_item_list)
            # split the test rating into test support and query data, and generate test query positive & negative for hr and ndcg calculation
        cnt = 0

        # this is for train rating loss calculation
        if args.dataset == 'movielens':
            for u in train_data:
                K = 1 + u % 10
                tmp_item_list = train_data[u]
                for v in tmp_item_list[:K]:
                    train_support_mat[(u, v)] = 1
                    train_support_data[u].append(v)
            # this is for val rating loss calculation
        else:
            for u in train_data:
                tmp_item_list = train_data[u]
                K = len(tmp_item_list) // 2
                if K > 0:
                    for v in tmp_item_list[:K]:
                        train_support_mat[(u, v)] = 1
                        train_support_data[u].append(v)

        return user_num,item_num,train_data, train_support_data, train_support_mat, val_negative_dict, val_negative,  train_mat, val_mat

    elif args.mod=='test':
        test_support_mat = {}
        test_mat = {}
        test_support_data = defaultdict(list)
        test_query_data = defaultdict(list)
        test_support,  test_negative = [], []
        test_negative_dict = defaultdict(list)
        test_rating_data = defaultdict(list)
        with open(file_name+'test.rating') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                u, v = int(arr[0]), int(arr[1])
                user_num = max(user_num, u)
                item_num = max(item_num, v)
                test_rating_data[u].append(v)
                test_mat[(u, v)] = 1.0
            # generate val positive & negative for hr and ndcg calculation

            # split the test rating into test support and query data, and generate test query positive & negative for hr and ndcg calculation
        if args.dataset=='movielens':
            cnt = 0
            for u in test_rating_data:
                K = 1 + u % 10
                if len(test_rating_data[u]) > K:
                    test_rating_item_list = test_rating_data[u]
                    # this is for generating the test support data and test query data
                    maxlen = min(len(test_rating_item_list), 4 * K)

                    test_support_data[u] = deepcopy(test_rating_item_list[:K])
                    test_query_data[u] = deepcopy(test_rating_item_list[K:maxlen])
                    # this if for generating the test query positive & negative data
                    for v in test_rating_item_list[:K]:
                        test_support_mat[(u, v)] = 1
                    test_negative_dict[u].append(cnt)
                    test_negative_dict[u].append(len(test_query_data[u]))
                    tmp_query_item_list = deepcopy(test_query_data[u])
                    for j in range(0, item_num):
                        if (u, j) not in test_mat:
                            tmp_query_item_list.append(j)
                    test_negative.append(tmp_query_item_list)
                    cnt += 1

        else:
            cnt = 0
            for u in test_rating_data:
                test_rating_item_list = test_rating_data[u]
                if (len(test_rating_item_list) > 1):
                    # this is for generating the test support data and test query data
                    K = len(test_rating_item_list) // 2
                    test_support_data[u] = deepcopy(test_rating_item_list[:K])
                    test_query_data[u] = deepcopy(test_rating_item_list[K:])
                    # this if for generating the test query positive & negative data
                    for v in test_rating_item_list[:K]:
                        test_support_mat[(u, v)] = 1
                    test_negative_dict[u].append(cnt)
                    test_negative_dict[u].append(len(test_query_data[u]))
                    tmp_query_item_list = deepcopy(test_query_data[u])
                    for j in range(0, item_num):
                        if (u, j) not in test_mat:
                            tmp_query_item_list.append(j)
                    test_negative.append(tmp_query_item_list)
                    cnt += 1

            # this is for train rating loss calculation

        return user_num,item_num,test_support_data,  test_negative_dict, test_negative, test_mat, test_support_mat

    else:
        train_mat = {}
        train_support_mat, test_support_mat = {}, {}
        val_mat = {}
        train_data, train_support_data = defaultdict(list), defaultdict(list)
        val_data = defaultdict(list)
        val_negative = []
        val_negative_dict = defaultdict(list)
        test_support_mat = {}
        test_mat = {}
        test_support_data = defaultdict(list)
        test_query_data = defaultdict(list)
        test_support, test_negative = [], []
        test_negative_dict = defaultdict(list)
        train_query_negative = []
        train_query_negative_dict = defaultdict(list)
        test_rating_data = defaultdict(list)
        with open(file_name+'train.rating') as f:
            for line in f:
                arr = line.split('\t')
                u, v = int(arr[0]), int(arr[1])
                train_data[u].append(v)
                train_mat[(u, v)] = 1.0
        val_rating = []
        with open(file_name+'val_train.rating') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                u, v = int(arr[0]), int(arr[1])
                val_data[u].append(v)
                val_mat[(u, v)] = 1.0
                val_rating.append([u, v])

        with open(file_name+'test.rating') as f:
            for line in f:
                arr = line.rstrip().split('\t')
                u, v = int(arr[0]), int(arr[1])
                user_num = max(user_num, u)
                item_num = max(item_num, v)
                test_rating_data[u].append(v)
                test_mat[(u, v)] = 1.0
        for idx, (u, v) in enumerate(val_rating):
            val_negative_dict[u].append(idx)
            val_negative_dict[u].append(1)
            tmp_val_item_list = [v]
            # select negative samples for each user -- first strategy
            for _ in range(args.num_negative_evaluation):
                j = np.random.randint(0, item_num)
                while (u, j) in val_mat or (u, j) in train_mat:
                    j = np.random.randint(0, item_num)
                tmp_val_item_list.append(j)
            val_negative.append(tmp_val_item_list)




        if args.dataset == 'movielens':
            cnt = 0
            for u in train_data:
                K = 1 + u % 10
                tmp_item_list = train_data[u]
                for v in tmp_item_list[:K]:
                    train_support_mat[(u, v)] = 1
                    train_support_data[u].append(v)
                train_query_negative_dict[u].append(cnt)

                max_len = min(4 * K, len(tmp_item_list))
                tmp_train_query_item_list = []
                train_query_negative_dict[u].append(len(tmp_item_list[K:max_len]))
                tmp_train_query_item_list.extend(tmp_item_list[K:max_len])
                for _ in range(args.num_negative_evaluation):
                    j = np.random.randint(0, item_num)
                    while (u, j) in val_mat or (u, j) in train_mat:
                        j = np.random.randint(0, item_num)
                    tmp_train_query_item_list.append(j)
                train_query_negative.append(tmp_train_query_item_list)
                cnt+=1
            cnt = 0
            for u in test_rating_data:
                K = 1 + u % 10
                if len(test_rating_data[u]) > K:
                    test_rating_item_list = test_rating_data[u]
                    # this is for generating the test support data and test query data
                    maxlen = min(len(test_rating_item_list), 4 * K)

                    test_support_data[u] = deepcopy(test_rating_item_list[:K])
                    test_query_data[u] = deepcopy(test_rating_item_list[K:maxlen])
                    # this if for generating the test query positive & negative data
                    for v in test_rating_item_list[:K]:
                        test_support_mat[(u, v)] = 1
                    test_negative_dict[u].append(cnt)
                    test_negative_dict[u].append(len(test_query_data[u]))
                    tmp_query_item_list = deepcopy(test_query_data[u])
                    for _ in range(args.num_negative_evaluation):
                        j = np.random.randint(0, item_num)
                        while (u, j) in val_mat or (u, j) in test_mat:
                            j = np.random.randint(0, item_num)
                        tmp_query_item_list.append(j)
                    test_negative.append(tmp_query_item_list)
                    cnt += 1


        else:
            cnt = 0
            for u in train_data:
                tmp_item_list = train_data[u]
                K = len(tmp_item_list) // 2
                if K > 0:
                    for v in tmp_item_list[:K]:
                        train_support_mat[(u, v)] = 1
                        train_support_data[u].append(v)
                    train_query_negative_dict[u].append(cnt)
                    tmp_train_query_item_list = []
                    train_query_negative_dict[u].append(len(tmp_item_list[K:]))
                    tmp_train_query_item_list.extend(tmp_item_list[K:])
                    for _ in range(args.num_negative_evaluation):
                        j = np.random.randint(0, item_num)
                        while (u, j) in val_mat or (u, j) in train_mat:
                            j = np.random.randint(0, item_num)
                        tmp_train_query_item_list.append(j)
                    train_query_negative.append(tmp_train_query_item_list)
                    cnt += 1
            cnt = 0
            for u in test_rating_data:
                test_rating_item_list = test_rating_data[u]
                if (len(test_rating_item_list) > 1):
                    # this is for generating the test support data and test query data
                    K = len(test_rating_item_list) // 2
                    test_support_data[u] = deepcopy(test_rating_item_list[:K])
                    test_query_data[u] = deepcopy(test_rating_item_list[K:])
                    # this if for generating the test query positive & negative data
                    for v in test_rating_item_list[:K]:
                        test_support_mat[(u, v)] = 1
                    test_negative_dict[u].append(cnt)
                    test_negative_dict[u].append(len(test_query_data[u]))
                    tmp_query_item_list = deepcopy(test_query_data[u])
                    for _ in range(args.num_negative_evaluation):
                        j = np.random.randint(0, item_num)
                        while (u, j) in val_mat or (u, j) in test_mat:
                            j = np.random.randint(0, item_num)
                        tmp_query_item_list.append(j)
                    test_negative.append(tmp_query_item_list)
                    cnt += 1

    return user_num,item_num,train_data, train_support_data, train_support_mat, val_negative_dict, val_negative,  train_mat, val_mat,train_query_negative,train_query_negative_dict,test_support_data,  test_negative_dict, test_negative, test_mat, test_support_mat,test_rating_data




class Train_data(data.Dataset):
    def __init__(self, train_data, item_num, train_support_mat,args):
        super(Train_data, self).__init__()
        self.train_data = train_data
        self.item_num = item_num
        self.train_support_mat = train_support_mat

        self.args = args
        self.num_ng = self.args.num_negative_support
    def len(self):
        return len(self.train_data.keys())

    def ng_train_sample(self):

        train_data = deepcopy(self.train_data)
        train_support_fill, train_query_fill = defaultdict(dict), defaultdict(dict)
        idx = 0
        if self.args.dataset=='movielens':
            for (u, item_list) in train_data.items():
                K = 1+u%10
                if len(item_list) > K:
                    random.shuffle(item_list)
                    support_user_list, support_item_list, support_label_list = [], [], []
                    for v in item_list[:K]:
                        support_user_list.append(u)
                        support_item_list.append(v)
                        support_label_list.append(1.0)

                        for _ in range(self.num_ng):
                            j = np.random.randint(0,self.item_num)
                            while (u, j) in self.train_support_mat:
                                j = np.random.randint(0,self.item_num)
                            support_user_list.append(u)
                            support_item_list.append(j)
                            support_label_list.append(0.0)
                    data_dict = {
                        'user': u,
                        'user_list': support_user_list,
                        'item_list': support_item_list,
                        'label_list': support_label_list
                    }
                    train_support_fill[idx] = data_dict

                    max_len = min(4*K, len(item_list))

                    query_user_list, query_item_list, query_label_list = [], [], []
                    for v in item_list[K:max_len]:
                        query_user_list.append(u)
                        query_item_list.append(v)
                        query_label_list.append(1.0)

                        for _ in range(self.num_ng):
                            j = np.random.randint(0,self.item_num)
                            while (u, j) in self.train_support_mat:
                                j = np.random.randint(0,self.item_num)
                            query_user_list.append(u)
                            query_item_list.append(j)
                            query_label_list.append(0.0)
                    data_dict = {
                        'user': u,
                        'user_list': query_user_list,
                        'item_list': query_item_list,
                        'label_list': query_label_list
                    }
                    train_query_fill[idx] = data_dict

                    idx += 1
                    self.idx = idx


        else:
            for (u, item_list) in train_data.items():
                K = len(item_list) // 2
                if K > 0:
                    random.shuffle(item_list)
                    support_user_list, support_item_list, support_label_list = [], [], []
                    for v in item_list[:K]:
                        support_user_list.append(u)
                        support_item_list.append(v)
                        support_label_list.append(1.0)

                        for _ in range(self.num_ng):
                            j = np.random.randint(0, self.item_num)
                            while (u, j) in self.train_support_mat:
                                j = np.random.randint(0, self.item_num)
                            support_user_list.append(u)
                            support_item_list.append(j)
                            support_label_list.append(0.0)
                    data_dict = {
                        'user': u,
                        'user_list': support_user_list,
                        'item_list': support_item_list,
                        'label_list': support_label_list
                    }
                    train_support_fill[idx] = data_dict
                    query_user_list, query_item_list, query_label_list = [], [], []
                    for v in item_list[K:]:
                        query_user_list.append(u)
                        query_item_list.append(v)
                        query_label_list.append(1.0)

                        for _ in range(self.num_ng):
                            j = np.random.randint(0, self.item_num)
                            while (u, j) in self.train_support_mat:
                                j = np.random.randint(0, self.item_num)
                            query_user_list.append(u)
                            query_item_list.append(j)
                            query_label_list.append(0.0)
                    data_dict = {
                        'user': u,
                        'user_list': query_user_list,
                        'item_list': query_item_list,
                        'label_list': query_label_list
                    }
                    train_query_fill[idx] = data_dict
                    idx += 1
                    self.idx = idx
            self.train_support_fill = train_support_fill
            self.train_query_fill = train_query_fill
        self.train_support_fill = train_support_fill
        self.train_query_fill = train_query_fill
    '''
    fill:{idx:{'item_lst':...,'label_lst':...,'user_lst':...}}
    '''
    def get_train_batch(self, batch_idx_list):
        supp_x_batch, supp_y_batch, query_x_batch, query_y_batch = [], [], [], []
        train_support_fill = self.train_support_fill
        train_query_fill = self.train_query_fill
        ret_user_list = []
        for idx in batch_idx_list:
            if idx in train_support_fill:
                user1 = train_support_fill[idx]['user']
                user_list = train_support_fill[idx]['user_list']
                item_list = train_support_fill[idx]['item_list']
                label_list = train_support_fill[idx]['label_list']
                supp_x_batch.append([user_list, item_list])
                supp_y_batch.append(label_list)

                user2 = train_query_fill[idx]['user']
                user_list = train_query_fill[idx]['user_list']
                item_list = train_query_fill[idx]['item_list']
                label_list = train_query_fill[idx]['label_list']
                query_x_batch.append([user_list, item_list])
                query_y_batch.append(label_list)

                assert user1 == user2
                ret_user_list.append(user1)
        return ret_user_list, supp_x_batch, supp_y_batch, query_x_batch, query_y_batch

class Test_data(data.Dataset):
    def __init__(self, test_support_data, item_num, test_support_mat, args):
        super(Test_data, self).__init__()
        self.test_support_data = test_support_data
        self.item_num = item_num
        self.test_support_mat = test_support_mat
        self.args = args
        self.num_ng = args.num_negative_support
        self.sup_dict = defaultdict(int)

    def ng_test_sample(self):
        test_support_data = deepcopy(self.test_support_data)
        test_support_fill = defaultdict(dict)
        if self.args.dataset=='movielens':
            for (u, tmp_item_list) in test_support_data.items():
                user_list, item_list, label_list = [], [], []
                mod = 1+(u%10)
                # import pdb; pdb.set_trace()
                for v in tmp_item_list:
                    user_list.append(u)
                    item_list.append(v)
                    label_list.append(1.0)

                    for _ in range(self.num_ng):
                        j = np.random.randint(0,self.item_num)
                        while  (u, j) in self.test_support_mat:
                            j = np.random.randint(0,self.item_num)
                        user_list.append(u)
                        item_list.append(j)
                        label_list.append(0.0)

                test_support_fill[u]['user_list'] = user_list[:mod * (self.num_ng + 1)]
                test_support_fill[u]['item_list'] = item_list[:mod * (self.num_ng+ 1)]
                test_support_fill[u]['label_list'] = label_list[:mod * (self.num_ng + 1)]
                self.sup_dict[u] = mod
        else:
            for (u, tmp_item_list) in test_support_data.items():
                user_list, item_list, label_list = [], [], []
            # import pdb; pdb.set_trace()
                for v in tmp_item_list:
                    user_list.append(u)
                    item_list.append(v)
                    label_list.append(1.0)
                    for _ in range(self.num_ng):
                        j = np.random.randint(0, self.item_num)
                        while (u, j) in self.test_support_mat:
                            j = np.random.randint(0, self.item_num)
                        user_list.append(u)
                        item_list.append(j)
                        label_list.append(0.0)

                test_support_fill[u]['user_list'] = user_list
                test_support_fill[u]['item_list'] = item_list
                test_support_fill[u]['label_list'] = label_list
                self.sup_dict[u] = len(tmp_item_list)

        self.test_support_fill = test_support_fill
