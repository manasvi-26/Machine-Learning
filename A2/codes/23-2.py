import numpy as np
import random
from client import *
import json

overfit = np.array([0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,
           8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10])


POP = 10
FEATURE = 11

def generate_initial(overfit, prob, mutation_delta):

    generation = np.zeros(shape = (POP,FEATURE))

    for i in range(POP):

        for feature in range(FEATURE):

            val = random.uniform(0,1)
            if(val <= prob):
                delta = random.uniform(-mutation_delta,mutation_delta)
                new_feature = overfit[feature] + delta

                if(new_feature < -10):
                    new_feature = -10
                elif new_feature > 10:
                    new_feature = 10

                generation[i][feature] = new_feature
    
    return generation







# CALL SERVER:

def call_server(generation):

    error = np.zeros(shape = (POP, 2))
    
    
    for i in range(len(generation)):
        

       curr_error = get_errors(SECRET_KEY,generation[i].tolist())
       error[i] = np.array(curr_error)

    return error



    # for i in range(len(generation)):

    #     curr_error = [random.uniform(1, 10), random.uniform(1, 10)]
    #     error[i] = curr_error
    

# FITNESS FUNCTION

def fitness_function(error):

    fitness = np.zeros(shape = (POP))
    probability = np.zeros(shape = (POP))

    for i in range(len(error)):
        curr_error = error[i]
        fitness[i] = (1/(curr_error[0] + curr_error[1]))
    
    total = sum(fitness)

    for i in range(len(fitness)):
        probability[i] = (int(fitness[i]/total * 100))

    return probability


def selection(generation, probability):


    sorted_idx = np.argsort(-probability)

    probability = probability[sorted_idx]
    generation = generation[sorted_idx]

    

    temp_generation = np.zeros(shape = (POP,FEATURE))
    temp_generation[:5] = generation[:5]
    temp_generation[5:] = generation[:5]

 
    temp_prob = np.zeros(shape = (POP))
    temp_prob[:5] = probability[:5]
    temp_prob[5:] = probability[:5]



    pool = np.array(random.choices(temp_generation, weights=probability, k=POP))
    return pool


def pick_two(pool):

    parent1 = np.zeros(shape = (FEATURE))
    parent2 = np.zeros(shape = (FEATURE))

    
    parent1 = random.choice(pool)
    parent2 = random.choice(pool)

    return np.array([parent1,parent2])


def crossover(pool):

    crossOver_generation = np.zeros(shape= (POP,FEATURE))

    i = 0
    while i < POP:

        parents =  pick_two(pool)
        
        parent1 = parents[0]
        parent2 = parents[1]
        

        child1 = np.zeros(shape = (FEATURE))

        child1[:5] = parent1[:5]
        child1[5:] = parent2[5:]

        crossOver_generation[i] = child1

        child2 = np.zeros(shape = (FEATURE))

        child2[:5] = parent1[:5]
        child2[5:] = parent2[5:]

        crossOver_generation[i+1]= child2

        i += 2

    return crossOver_generation


def mutation(crossOver_generation, mutation_prob):

    mutation_delta = 0.1

    for child in crossOver_generation:

        for feature_index in range(FEATURE):

            val = random.uniform(0, 1)
            if(val <= mutation_prob):
                delta = random.uniform(-mutation_delta,mutation_delta)

                new_feature = child[feature_index] + delta

                if(new_feature < -10):
                    new_feature = -10
                elif new_feature > 10:
                    new_feature = 10
                
                child[feature_index] = new_feature

        
    mutated_generation = crossOver_generation

    return mutated_generation


# INITIAL POPULATION


# def Convert(v):
#     print(type(v))

#     # for i in len(range(v)):
#     #     print(v[i])
#     #     #v[i] = v[i].tolist()
    
#     new_v = v.tolist()

#     return new_v



# for i in range(3):

#     generation = generate_initial(overfit, 0.9, 0.1)

#     generation = Convert(generation)

#     print(type(generation))
    
#     with open('../output_files/temp.txt', 'a+') as write_file:
#         write_file.write("\n")
#         write
#         json.dump(generation,write_file)


def write_to_file(vector, text, iter):


    with open('../output_files/23-3.txt', 'a+') as write_file:

        if(text == 'generation'):
            write_file.write("\n")
            write_file.write(str(iter) + 'ITERATION')
            write_file.write("\n")

        
        write_file.write("\n")
        write_file.write(text)
        write_file.write("\n")

    

        json.dump(vector.tolist(),write_file)
            




generation = generate_initial(overfit, 0.9, 0.1)



for iter in range(40):

    write_to_file(generation,'generation',iter)

    # call server
    error = call_server(generation)
    write_to_file(error,'error',iter)
    



    # call fitness
    probability = fitness_function(error)
 


    # call selection
    pool = selection(generation, probability)
    write_to_file(pool,'pool',iter)


    # call crossover
    crossOver_generation = crossover(pool)
    write_to_file(crossOver_generation,'crossOver_generation',iter)


    # call mutation
    mutated_generation = mutation(crossOver_generation,0.15)
    write_to_file(mutated_generation,'mutated_generation',iter)

    generation = mutated_generation
    


    
# #server call for final generation
with open('../output_files/23-final.txt', 'a+') as write_file:
    json.dump(generation.tolist(),write_file)
    write_file.write("\nERRRORSS\n")

error = call_server(generation)



with open('../output_files/23-final.txt', 'a+') as write_file:
    json.dump(error.tolist(),write_file)

min_error =1e15
min_index = 0
for i in range(len(error)):
    if(min_error  > (error[i][0] +error[i][1])):
        min_error=error[i][0] +error[i][1]
        min_index = i

print(min_error)
print(generation[min_index])


