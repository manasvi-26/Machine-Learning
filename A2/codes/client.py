
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

    # vector = [ 0.00000000e+00 ,-3.10744347e-12 ,-2.46882848e-13,  2.67415867e-11 ,  -2.57374805e-10, -8.77976264e-16,  4.54485346e-16 , 3.08747381e-05  , -1.74993530e-06 ,-1.85640056e-08 , 8.24879099e-10]


    # vector = [  0.00000000e+00, -1.57547227e-12 ,-2.40119075e-13,  4.89499128e-11,  -1.64218248e-10, -2.00376659e-15 , 7.78853081e-16,  2.48021851e-05  , -2.04194077e-06 ,-1.56731106e-08 , 9.64485030e-10]  
    #print(get_errors(SECRET_KEY, vector))

    # vector= [ 0.00000000e+00, -1.50228415e-12, -2.41971422e-13,  3.94417365e-11   ,-1.72591933e-10 ,-1.98499397e-15 , 8.55534786e-16,  2.30232227e-05   ,-2.15396963e-06 ,-1.43901548e-08,  1.02692233e-09] 

    # vector= [ 0.00000000e+00, -3.41060814e-12, -2.44057642e-13 , 3.32970255e-11 , -2.34489962e-10, -8.75105510e-16,  4.82925578e-16, 3.08769046e-05, -1.85212405e-06, -1.85648349e-08,  8.68486189e-10]   

    # vector = [ 0.00000000e+00, -1.66550241e-12 ,-2.29985424e-13 , 5.35018759e-11  , -1.81428447e-10 ,-2.05710193e-15  ,7.78156908e-16 , 2.36761052e-05  , -1.75974227e-06 ,-1.79462865e-08  9.06340314e-10] 
    vector =  [ 0.00000000e+00, -3.06722892e-12, -1.99002998e-13,  2.79446330e-11, -2.84586056e-10 ,-7.11713380e-16,  4.25717393e-16,  3.08746569e-05, -1.74988447e-06, -1.85640092e-08, 8.22020387e-10]  
    print(submit(SECRET_KEY,vector))
    

