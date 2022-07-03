'''
Explain iot identification using DeepExplain on UNSW dataset

explain wrong instances, including wrongly predicted iot/niot instances, and wrongly predicted known/new type instances(especially the confusion-prone devices)

step 1. obtain wrong predicted instances.
step 2. construct explain model 
step 3. obtain explanations

'''
import os
import sys
import joblib
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow.python.training import saver as tf_saver
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn_ops, gen_nn_ops
from collections import Counter

curdir = os.path.dirname(os.path.abspath(__file__))
pardir = os.path.dirname(curdir)
sys.path.append(pardir)

from configloader import ConfigLoader
from iot_explain_utils.log_utils import init_log
from iot_explain_utils.plot_utils import plot_feature_distribution, plot_hbar, plot_scatter
import time

plot_detail_flag = False # whether plot the detail explanation results (feature distribution of explanation results)

class Explainer(object):
    def __init__(self,config_ins,logger) -> None:
        self.config_ins = config_ins
        self.logger = logger
        self.new_devices_list = config_ins.data.new_devices_list
        self.new_devices_list_str = '_'.join([str(i) for i in self.new_devices_list])
        self.data_path = '{}/{}'.format(config_ins.data.data_path,self.new_devices_list_str)
        self.model_path = '{}/{}'.format(config_ins.data.model_path,self.new_devices_list_str)
        self.top_K = config_ins.explain.top_K
        self.config_ins = config_ins

    def load_wrong_instances(self):
        '''obtain wrong predicted instances'''
        self.wrong_instances_pd = pd.read_csv(self.data_path + '/wrong_iot_niot_instances.csv')
        self.logger.info('num of wrong iot/niot instances:{}'.format(len(self.wrong_instances_pd)))

    def explain(self):
        from iot_explain_utils.dispose_utils import one_hot_encoding
        # restore graph
        with open(self.model_path + '/model_name','r') as f:
            model_name = f.readline().strip('\n')        
        
        sess = tf.Session()
        
        # get gradient
        wrong_instances = self.wrong_instances_pd.values[:,1:-3]
        wrong_ins_labels = self.wrong_instances_pd.values[:,0]
        wrong_ins_labels_one_hot = one_hot_encoding(wrong_ins_labels,max_label=1)

        # use DeepExplain
        from deepexplain.tensorflow.methods import DeepExplain
        from examples.utils import plot, plt
        with DeepExplain(session=sess) as de:
            g = tf.get_default_graph()
            saver = tf.compat.v1.train.import_meta_graph(self.model_path + '/' + model_name +'.meta')
            saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
            X = sess.graph.get_tensor_by_name('eval_in:0')
            # pred_class = sess.graph.get_tensor_by_name('compute_logits_name_scope/fully_connected/BiasAdd:0') # specific class logits
            logits = sess.graph.get_tensor_by_name('compute_logits_name_scope/fully_connected_1/BiasAdd:0') # 
            # logits = pred_iot_niot # 计算图的一部分，通过de.explain的参数传入
            # We run `explain()` several time to compare different attribution methods
            attributions = {
                # # Gradient-based
                'Saliency maps':        de.explain('saliency', logits * wrong_ins_labels_one_hot, X, wrong_instances), # logits * yi表示仅保留目标类别的logit
                'Gradient * Input':     de.explain('grad*input', logits * wrong_ins_labels_one_hot, X, wrong_instances),
                'Integrated Gradients': de.explain('intgrad', logits * wrong_ins_labels_one_hot, X, wrong_instances),
                'Epsilon-LRP':          de.explain('elrp', logits * wrong_ins_labels_one_hot, X, wrong_instances),
                'DeepLIFT (Rescale)':   de.explain('deeplift', logits * wrong_ins_labels_one_hot, X, wrong_instances),
                # #Perturbation-based
                '_Occlusion [1x1]':      de.explain('occlusion', logits * wrong_ins_labels_one_hot, X, wrong_instances),
                # '_Occlusion [3x3]':      de.explain('occlusion', logits * wrong_ins_labels_one_hot, X, wrong_instances, window_shape=(3,))
            }
            self.logger.info('Done')

        return attributions # need to be modified, return explanation results

    def analyze_expl_res(self,attributions):
        '''analyze the explanation results'''
        # get original features of instances
        origin_new_devices_pd = pd.read_csv(self.data_path + '/new_devices_train_data_before_norm_for_expl.csv')
        origin_old_iot_pd, origin_old_niot_pd = self.obtain_old_devices_train_pd()
        # for each wrong instance, plot feature distribution in iot/non-iot instances and this feature.
        feature_col_names = list(origin_new_devices_pd.columns)[1:-1]
        print(len(feature_col_names))

        fidelity_dict = {}

        for key, value in attributions.items():
            fidelity_dict[key] = []
            expl_res, time_dur = value
            self.logger.info('Explanation method:{}, whole time used:{}, average_time used:{}'.format(key,time_dur,round(time_dur/len(expl_res),5)))
            top_K_collection = []
            feature_grad_dict = {}

            # get expls for each instance
            line_idx = 0
            for expl in expl_res:
                index = self.wrong_instances_pd.iloc[line_idx,-3]
                prob = self.wrong_instances_pd.iloc[line_idx,-1]
                label = self.wrong_instances_pd.iloc[line_idx,0]
                self.logger.info('label:{}, instance index:{}, wrong label prob:{}'.format(label,index,prob))
                # get K important features for wrong prediction
                sorted_index = np.argsort(-expl)

                # fidelity test
                if self.config_ins.evaluation.fidelity_flag == 1:
                    scores = self.fidelity_score(index,prob,sorted_index)
                    self.logger.info('fidelity score:{}'.format(scores))
                    fidelity_dict[key].append(scores)

                if plot_detail_flag is True:
                    # obtain top_K features
                    top_features = [feature_col_names[i] for i in sorted_index[:self.top_K]]
                    # for fidx in sorted_index[:self.top_K]:
                    for fidx in sorted_index:
                        if fidx not in feature_grad_dict.keys():
                            feature_grad_dict[fidx] = []
                        fvalue = origin_new_devices_pd[origin_new_devices_pd['index']==index].values[0,fidx+1]
                        feature_grad_dict[fidx].append((fvalue,expl[fidx]))
                        
                    top_K_collection += list(sorted_index[:self.top_K])
                    self.logger.info('top {} features:{}'.format(self.top_K,top_features))
                    # plot top_K bar 
                    fig_path = self.config_ins.data.fig_path + '/' + self.new_devices_list_str + '/{}'.format(key)
                    if not os.path.exists(fig_path):
                        os.makedirs(fig_path)
                    top_grads = [expl[i] for i in sorted_index[:self.top_K]]
                    plot_hbar(top_features,top_grads,xlabel='Gradients',ylabel='Feature',path=fig_path,filename='{}_top{}'.format(index,self.top_K))
                    
                    # observe top_K features for iot/non-iot instances
                    for feature_idx in sorted_index[:self.top_K]:
                        iot_features = origin_old_iot_pd.iloc[:,feature_idx+1].values
                        niot_features = origin_old_niot_pd.iloc[:,feature_idx+1].values
                        # plot 
                        fvalue = origin_new_devices_pd[origin_new_devices_pd['index']==index].values[0,feature_idx+1]
                        plot_feature_distribution(iot_features,niot_features,fvalue,label1='IoT devices',label2='Non-IoT devices',xlabel=feature_col_names[feature_idx],ylabel='Number of instances',index=index,path=fig_path)
                line_idx += 1

            if plot_detail_flag is True:
                # get statistics for all expls
                # overall top K
                topk_count_values = Counter(top_K_collection)
                sorted_topk_count_values = sorted(topk_count_values.items(),key=lambda x:x[1],reverse=True)
                features_top_K = [item[0] for item in sorted_topk_count_values[:self.top_K]]
                count_top_K = [item[1] for item in sorted_topk_count_values[:self.top_K]]
                top_features = [feature_col_names[i] for i in features_top_K]
                plot_hbar(top_features,count_top_K,xlabel='Count',ylabel='Feature',path=fig_path,filename='overall_top{}_and_instance_count'.format(self.top_K),plus=False)
                
                # plot scatter of features_top_k, feature value: grads
                for i,fidx in enumerate(features_top_K):
                    items = feature_grad_dict[fidx]
                    fvalues = [item[0] for item in items]
                    grads = [item[1] for item in items]
                    plot_scatter(fvalues,grads,feature_col_names[fidx],'Gradients',path=fig_path,filename='overall_top{}_{}_scatter'.format(i,feature_col_names[fidx]))

        # save fidelity_dict
        path = self.config_ins.data.fig_path + '/' + self.new_devices_list_str
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/fidelity_dict.joblib','wb') as f:
            joblib.dump(fidelity_dict,f)
            f.close()
        self.logger.info('fidelity_dict has been saved')
        self.logger.info('Done')

    def fidelity_score(self,index,prob,sorted_index):
        '''get fidelity_score for an instance'''

        # load model
        with open(self.model_path + '/model_name','r') as f:
            model_name = f.readline().strip('\n')
        sess = tf.Session()
        g = tf.get_default_graph()
        saver = tf.compat.v1.train.import_meta_graph(self.model_path + '/' + model_name +'.meta')
        saver.restore(sess, tf.train.latest_checkpoint(self.model_path))
        X = sess.graph.get_tensor_by_name('eval_in:0')
        pred_iot_niot = sess.graph.get_tensor_by_name('compute_logits_name_scope/fully_connected_1/BiasAdd:0')
        pred_label = tf.argmax(pred_iot_niot,dimension=1)
        
        # dispose features
        features = self.wrong_instances_pd[self.wrong_instances_pd['index']==index].values[:,1:-3]
        # ori_logits = sess.run(pred_iot_niot,feed_dict={X:features})
        N = self.config_ins.evaluation.fidelity_N
        percents = self.config_ins.evaluation.fidelity_percents
        scores = [] 
        sorted_index_r = sorted_index[::-1]
        # sorted_index_r = sorted_index # random important features
        for percent in percents:
            alter_feature_num = int(len(features[0]) * percent)
            alter_feature_idxes = sorted_index_r[:alter_feature_num]
            features_now = None
            for _ in range(N):
                # alter the target features
                random_nums = np.random.rand(1,alter_feature_num)
                for fidx, rand in zip(alter_feature_idxes, random_nums[0]):
                    features[0,fidx] = rand
                if features_now is None:
                    features_now = features
                else:
                    features_now = np.concatenate([features_now,features],axis=0)

            # load model and compute pred
            preds = sess.run([pred_label],feed_dict={X:features_now})
            score = round(np.sum(preds)/len(preds[0]),6)
            scores.append(score)
        return scores
    


    #=======================================called functions===================================

    #---------------------------------------data functions-------------------------------------
    def obtain_old_devices_train_pd(self):
        '''obtain original old devices data'''
        origin_old_devices_pd = pd.read_csv(self.data_path + '/old_devices_train_data_before_norm_for_expl.csv')
        # obtain iot/non-iot instances
        old_labels = np.unique(origin_old_devices_pd['label'].values)
        niot_label = max(old_labels)
        origin_old_niot_pd = origin_old_devices_pd[origin_old_devices_pd['label']==niot_label]
        origin_old_iot_pd = origin_old_devices_pd.drop(origin_old_devices_pd[origin_old_devices_pd['label']==niot_label].index,inplace=False)
        return origin_old_iot_pd, origin_old_niot_pd
    

def main():
    time0 = time.time()
    config_path = "./iot_explain/explain_config_unsw.json"
    config_ins = ConfigLoader(config_path).args
    logger = init_log(config_ins.data.log_path + '/explain_{}_{}.log'.format('DeepExplain',config_ins.data.new_devices_list))
    explainer = Explainer(config_ins,logger)
    explainer.load_wrong_instances()
    attributions = explainer.explain()
    # time_expl = time.time()-time0
    # logger.info('Explain time:{}, instance_num:{}, average:{}'.format(time_expl,len(expl_res),time_expl/len(expl_res)))
    explainer.analyze_expl_res(attributions)


if __name__ == '__main__':
    main()