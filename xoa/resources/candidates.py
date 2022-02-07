import os
import time
import json

from xoa.commons.logger import * 

from flask import jsonify, request
from flask_restful import Resource, reqparse

class Candidates(Resource):
    def __init__(self, **kwargs):
        self.sm = kwargs['space_manager']

        super(Candidates, self).__init__()

    def get(self, space_id):
        parser = reqparse.RequestParser()
        parser.add_argument("Authorization", location="headers") # for security reason
        
        args = parser.parse_args()
        if not self.sm.authorize(args['Authorization']):
            return "Unauthorized", 401

        space = self.sm.get_space(space_id)
        if space == None:
            return "Search space {} is not available".format(space_id), 500

        result = {}

        result["candidates"] = space.get_candidates() 

        return result, 200 