
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

def get_overfit_vector(id):
    return json.loads(send_request(id, [0], 'getoverfit'))

def submit(id, vector):
    """
    used to make official submission of your weight vector
    returns string "successfully submitted" if properly submitted.
    """
    for i in vector: assert 0<=abs(i)<=10
    assert len(vector) == MAX_DEG
    return send_request(id, vector, 'submit')

# Replace 'SECRET_KEY' with your team's secret key (Will be sent over email)
if __name__ == "__main__":

    vector  = [ 0.00000000e+00, -1.56844432e-12, -2.29662979e-13,  5.13997944e-11, -1.63378653e-10 ,-1.91099748e-15 , 7.87986179e-16  ,2.66647411e-05 ,-2.04052537e-06, -1.67867929e-08,  9.67018523e-10]

    
    print(submit(SECRET_KEY,vector))

    

