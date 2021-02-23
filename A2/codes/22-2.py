import numpy as np
import random
from client import *
import json

overfit = np.array([0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,
           8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10])

POP = 10
FEATURE = 11


# CALL SERVER:

def call_server(generation):
    error = []
    
    for v in generation:
        v = np.array(v)
        curr_error = get_errors(SECRET_KEY,v.tolist())
        error.append(curr_error)

    return error

    # error = []
    # for v in generation:
    #     v = np.array(v)
    #     curr_error = [random.uniform(1, 10), random.uniform(1, 10)]
    #     error.append(curr_error)
    

    # return error


# FITNESS FUNCTION

def fitness_function(error):

    fitness = []
    probability = []

    for curr_error in error:

        fitness.append(1/(curr_error[0] + curr_error[1]))

    total = sum(fitness)

    for val in fitness:
        probability.append(int(val/total * 100))

    return probability


def selection(generation, probability):

    pool = random.choices(generation, weights=probability, k=POP)
    return pool


def crossover(pool):
    crossOver_generation = []

    i = 0
    while i < POP:
        child1 = [0] * FEATURE
        child1[:5] = pool[i][:5]
        child1[5:] = pool[i+1][5:]

        crossOver_generation.append(child1)

        child2 = [0] * FEATURE

        child2[:5] = pool[i][:5]
        child2[5:] = pool[i+1][5:]

        crossOver_generation.append(child2)

        i += 2

    return crossOver_generation


def mutation(crossOver_generation):

    mutation_prob = 0.15

    for child in crossOver_generation:

        for feature_index in range(FEATURE):

            val = random.uniform(0, 1)
            if(val <= mutation_prob):
                new_feature = random.uniform(-10, 10)
                child[feature_index] = new_feature

    mutated_generation = crossOver_generation

    return mutated_generation


# INITIAL POPULATION

generation = []
generation.append(overfit)

for i in range(POP - 1):

    curr_vector = np.random.uniform(low=-10, high=10, size=FEATURE)
    generation.append(curr_vector)


for iter in range(40):

   

    # call server

    error = call_server(generation)
   


    # call fitness
    probability = fitness_function(error)

    # call selection
    pool = selection(generation, probability)

    # call crossover
    crossOver_generation = crossover(pool)

    # call mutation
    mutated_generation = mutation(crossOver_generation)


    # with open('../output_files/22-generation.txt', 'w') as write_file:
    #     json.dump(generation,write_file)

    with open('../output_files/22-error.txt', 'a') as write_file:
        json.dump(error,write_file)

    generation = mutated_generation
    


    
# #server call for final generation
# with open('../output_files/22-generation.txt', 'w') as write_file:
#     json.dump(generation,write_file)

error = call_server(generation)

with open('../output_files/22-error.txt', 'a') as write_file:
    json.dump(error,write_file)

min_error =1e15
min_index = 0
for i in range(len(error)):
    if(min_error  > (error[i][0] +error[i][1])):
        min_error=error[i][0] +error[i][1]
        min_index = i

print(min_error)
print(generation[min_index])



