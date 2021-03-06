import json
import time
import sys

import numpy as np
import math

from xoa.commons.logger import *

from xoa.connectors.remote_job import RemoteJobConnector


class RemoteOptimizerConnector(RemoteJobConnector):
    
    def __init__(self, ip_addr, port, cred, **kwargs):
        
        self.ip_addr = ip_addr
        self.port = port
        
        url = "http://{}:{}".format(ip_addr, port)
        
        super(RemoteOptimizerConnector, self).__init__(url, cred, **kwargs)

    def validate(self):
        try:
            profile = self.get_profile()
            if profile and "spec" in profile and "node_type" in profile["spec"]:
                if profile["spec"]["node_type"] == "BO Node":
                    return True       

        except Exception as ex:
            warn("Validation failed: {}".format(ex))
            
        return False

