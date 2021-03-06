from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path

import pickle
import gzip
import json

from xoa.commons.logger import *


class HistorySaver(object):
    def __init__(self, data_type, run_mode, 
                target_goal, time_expired, config, 
                path='./results/', postfix=""):

        self.data_type = data_type
        self.run_mode = run_mode
        self.target_goal = target_goal
        self.time_expired = time_expired

        self.path = path
        self.config = config
        self.postfix = postfix

    def save(self, name1, name2, trials, results, internals=None):
        directory = self.path + str(self.data_type)

        if self.run_mode == 'GOAL':
            directory += "/G" + str(self.target_goal)
        elif self.run_mode == 'TIME':
            directory += "/T" + str(int(self.time_expired)) + "S"

        if not os.path.exists(directory):
            debug('Creating ' + directory)
            os.makedirs(directory)

        file_path = directory + '/'
        if name1 != 'DIV' or name2 != 'RANDOM':
            file_path = file_path + name1 + '-' + name2
        elif 'title' in self.config:
            file_path = file_path + self.config['title'].replace(" ", "_")
        
        file_path = "{}{}({})".format(file_path, self.postfix, str(trials))
        for retry in range(1, 100):        
            if os.path.exists("{}.json".format(file_path)):
                file_path = "{}.retry".format(file_path)
            else:
                break
            
        info("The optimization result saved as {}.json".format(file_path))
        with open(file_path + '.json', 'w') as json_file:
            json_file.write(json.dumps(results))

        if internals is not None:
            with gzip.open(file_path + '.pkl.gz', 'wb') as pkl_gz:
                pickle.dump(internals, pkl_gz)
        
    def load(self, optimizer, aquisition_func, num_trials):
        directory = self.path + str(self.data_type) 
        if self.run_mode == 'GOAL':
            directory += "/G" + str(self.target_goal)
        elif self.run_mode == 'TIME':
            directory += "/T" + str(int(self.time_expired)) + "S"

        if not os.path.exists(directory):
            debug('Creating ' + directory)
            os.makedirs(directory)
        file_name = "{}-{}.{}({}).json".format(optimizer, aquisition_func, self.postfix, num_trials)
        file_path = directory + '/' + file_name

        if os.path.isfile(file_path):
            with open(file_path) as json_temp:
                temp_results = json.load(json_temp)
                iters = [int(key) for key in temp_results.keys()]
                sorted_iters = sorted(iters)
                return temp_results, sorted_iters[-1] + 1
        else:
            warn("File not found: {}".format(file_path))
            return {}, 0


class TemporaryHistorySaver(object):
    
    def __init__(self, data_type, optimizer, 
                 aquisition_func, num_trials, config,
                 path='./temp/'):
        title = ""
        if "title" in config:
            title = config["title"]
        else:
            title = data_type

        self.temp_name = "{}.{}-{}({})".format(title, 
                        optimizer, aquisition_func, num_trials)

        self.path = path
        if not os.path.exists(self.path):
            debug('Creating ' + self.path)
            os.makedirs(self.path)        
        
        self.temp_file_path = None

    def save(self, results):
        self.temp_file_path = self.path + self.temp_name + '.json'
        with open(self.temp_file_path, 'w') as json_file:
            json_file.write(json.dumps(results))

    def restore(self):
        self.temp_file_path = self.path + self.temp_name + '.json'
        if os.path.isfile(self.temp_file_path):
            with open(self.temp_file_path) as json_temp:
                try:                
                    temp_results = json.load(json_temp)
                    iters = [int(key) for key in temp_results.keys()]
                    sorted_iters = sorted(iters)
                    return temp_results, sorted_iters[-1] + 1
                except Exception as ex:
                    warn("temp file loading error: {}".format(ex))
                    return {}, 0
        else:
            return {}, 0

    def remove(self):
        if self.temp_file_path is not None:
            try:
                os.remove(self.temp_file_path)
            except OSError:
                pass

    
