from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from xoa.commons.logger import *
from trainers.proto import TrainerPrototype


class TrainEmulator(TrainerPrototype):
    def __init__(self, lookup):
    
        self.lookup = lookup
        self.acc_curves = lookup.get_all_test_acc_per_epoch()
        self.total_times = lookup.get_all_exec_times()
        self.data_type = lookup.data_type

        super(TrainEmulator, self).__init__()

    def get_min_train_epoch(self):
        if self.lookup.data_type == "CIFAR10-ResNet":
            return 30
        elif self.lookup.data_type == "CIFAR10-VGG" or \
            self.lookup.data_type == "CIFAR100-VGG":
            return 15
        else:
            return 4  

    def train(self, cand_index, train_epoch=None):
        acc_curve = self.acc_curves.loc[cand_index].values
        if train_epoch == None:
            train_epoch = len(acc_curve)
        total_time = self.total_times[cand_index]
        return {
                "test_error": 1.0 - max(acc_curve), 
                "exec_time" : total_time,
                "train_epoch" : train_epoch, 
                'early_terminated' : False
        } 

    def get_interim_error(self, model_index, cur_dur=0):
        total_dur = self.total_times[model_index]
        cur_epoch = int(cur_dur / total_dur * self.lookup.num_epochs)
        error = 0.9 # random initial performance for 10 classification problem
        if cur_epoch == 0:
            
            if self.data_type == 'CIFAR100-VGG':
                error = 0.99 # for CIFAR-100
            elif self.data_type == 'PTB-LSTM':
                error = 0.7 # for PTB
        else:
            pre_errors = self.lookup.get_all_test_errors(cur_epoch)
            error = pre_errors[model_index]
        #debug("training model #{} at {:.1f} may return loss {:.4f}.".format(model_index, cur_dur, error))
        return error, cur_epoch



class EarlyTerminationTrainer(TrainEmulator):

    def __init__(self, lookup):

        super(EarlyTerminationTrainer, self).__init__(lookup)
        self.early_terminated_history = []

    def initialize(self):
        super(EarlyTerminationTrainer, self).initialize()        
        self.early_terminated_history = []

    def get_train_time(self, cand_index, stop_epoch):
        # XXX: consider preparation time later
        total_time = self.total_times[cand_index]
        acc_curve = self.acc_curves.loc[cand_index].values
        epoch_length = len(acc_curve)
        elapsed_time = stop_epoch * (total_time / epoch_length)
        debug("Evaluation time saving is {:.1f} sec".format(total_time - elapsed_time))
        return elapsed_time

    def get_preevaluated_result(self, cand_index):
        acc_curve = self.acc_curves.loc[cand_index].values
        train_time = self.total_times[cand_index]
        min_loss = 1.0 - max(acc_curve)

        return {
            "curve": acc_curve, 
            "measure": "test_accuracy",
            "train_time" : train_time, 
            "test_error" : min_loss
        }

    def add_train_history(self, curve, train_time, cur_epoch, 
                          early_terminated=False):
        super(EarlyTerminationTrainer, self).add_train_history(curve, train_time, cur_epoch)
        self.early_terminated_history.append(early_terminated)      


class EarlyStopTerminateBoilerplate(EarlyTerminationTrainer):
    ''' Sample code for your ETR logic. 
        You will implement the following methods:   
    
    '''
    def __init__(self, lookup, **kwargs):
        # TODO: you can add more attributes here (if required)       
        super(EarlyStopTerminateBoilerplate, self).__init__(lookup)

    def train(self, cand_index, train_epoch=None):
        # Firstly, you should add current candidate index to 
        
        # You can access the accuracy curve as below
        acc_curve = self.acc_curves.loc[cand_index].values
        if train_epoch == None:
            train_epoch = len(acc_curve)

        # You can also access a previous learning curve as follows:
        for h in self.history:
            print(h) 

        # TODO: Your algorithm here
        early_termination = True # if your algorithm fires, append the result as below:
        if early_termination:
            self.early_terminated_history.append(True)
            stopped_error = 0.1
            train_epoch = 5 
            eval_time = 100

            # TODO: you should return the error when it stopped and time to execute here.
            return {
                    "test_error": stopped_error, 
                    "exec_time" : eval_time,
                    "train_epoch" : train_epoch, 
                    'early_terminated' : True
            }  
        else:
            self.early_terminated_history.append(False)
            return {
                    "test_error": 1.0 - max(acc_curve), 
                    "exec_time" : self.total_times[cand_index],
                    "train_epoch" : train_epoch,  
                    'early_terminated' : False
            }  
