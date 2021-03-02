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

    # overfit one
    # vector = [0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,  8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10]
    # 25-2 expand one we thought gave 9e10 - this is giving different shit
    # vector = [0.00000000e+00, -1.84288877e-12, -2.93010830e-13, 4.11635039e-11, -2.12794636e-10, -1.39418732e-15,  7.15111832e-16,  2.55880732e-05, -1.77766143e-06, -1.55306208e-08,  1.04842915e-09 ]
    # vector = [0.0, -1.842888771349534e-12, -2.930108304444428e-13, 4.116350387974107e-11, -2.1279463576563586e-10, -1.3941873211529634e-15, 7.151118317602366e-16, 2.558807323846453e-05, -1.7776614341251699e-06, -1.553062079657628e-08, 1.048429147285637e-09]
    # 25-2 first one in expand to test - this worked
    # vector = [0.0, -1.45799022e-12, -2.4178492537053024e-13, 4.407903141782155e-11, -1.6749042691061334e-10, -1.8676247767445545e-15, 7.783164100170254e-16, 2.3668771599425383e-05, -2.04721003e-06, -1.5111774259448022e-08, 9.731446629123692e-10]
    # 25-2 second one in expand to test
    
    # vector = [0.0, -1.6066728557765327e-12, -2.4562371616049357e-13, 4.6487301261255185e-11, -1.740033074341892e-10, -1.6144896397602713e-15, 8.891655886550115e-16, 2.357425803839365e-05, -2.062455168665507e-06, -1.55163546381005e-08, 9.467599204279566e-10]
    
    # vector = [0.00000000e+00, -1.52684317e-12, -2.28389751e-13, 4.51766947e-11, 1.81359412e-10, -2.05574006e-15, 8.26510581e-16, 2.16663576e-05, -2.03399836e-06, -1.71273179e-08, 1.01012641e-09]
   
    # vector = [ 0.00000000e+00, -1.45799022e-12, -2.29723347e-13, 5.05721113e-11, -1.88471677e-10, -2.01458347e-15, 7.69554794e-16, 2.44879733e-05, -2.07575171e-06, -1.59792834e-08, 9.98214034e-10]
    
    # vector = [0.00000000e+00, -1.53547055e-12, -2.29788258e-13, 4.80804432e-11, -1.63910484e-10, -1.90830141e-15, 8.41871349e-16, 2.30800838e-05, -2.18002985e-06, -1.71856498e-08, 9.49408707e-10]
    
    #in iteration 1 1-3.txt first vector the error is actually the one in the 5th row
    # vector = [ 0.00000000e+00, -1.53713619e-12, -2.38064603e-13,  4.62317745e-11, -1.75337227e-10, -1.97265530e-15, 8.57428035e-16, 2.44227940e-05, -2.05956489e-06, -1.79328545e-08, 1.00216466e-09] 

    vector = [[ 0.00000000e+00, -2.00108004e-12, -2.63098466e-13,  3.87031427e-11, -1.63058530e-10, -1.94424031e-15,  5.01630380e-16,  2.67026222e-05, -1.57991611e-06, -1.67642227e-08, 9.72768068e-10]


    print(get_errors(SECRET_KEY, vector))
    
