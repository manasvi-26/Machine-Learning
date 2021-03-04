import json
import requests
import numpy as np
from config import *

API_ENDPOINT = 'http://10.4.21.156'
MAX_DEG = 11

def urljoin(root, path=''):
    if path: root = '/'.join([root.rstrip('/'), path.rstrip('/')])
    return root

def send_request(id, vector, path):
    api = urljoin(API_ENDPOINT, path)
    vector = json.dumps(vector)
    
    response = requests.post(api, data={'id':id, 'vector':vector}).text
    if "reported" in response:
        print(response)
        exit()

    return response

def get_errors(id, vector):
    for i in vector: assert 0<=abs(i)<=10
    assert len(vector) == MAX_DEG

    return json.loads(send_request(id, vector, 'geterrors'))
    

# def get_overfit_vector(id):
#     return json.loads(send_request(id,[0], 'getoverfit'))

# Replace 'SECRET_KEY' with your team's secret key (Will be sent over email)
if __name__ == "__main__":
    vector = [0.0, -1.6066728557765327e-12, -2.4562371616049357e-13, 4.6487301261255185e-11, -1.740033074341892e-10, -1.6144896397602713e-15, 8.891655886550115e-16, 2.357425803839365e-05, -2.062455168665507e-06, -1.55163546381005e-08, 9.467599204279566e-10]

    print(get_errors(SECRET_KEY, vector))
    
