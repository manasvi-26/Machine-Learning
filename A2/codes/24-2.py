import numpy as np
import random
from client import *
import json

overfit = np.array([0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,
           8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10])

POP = 10
FEATURE = 11

def generate_initial(overfit, prob):

    generation = np.zeros(shape = (POP,FEATURE))

    for i in range(POP):

        for feature in range(FEATURE):

            val = random.uniform(0,1)
            
            if(val <= prob):
                
                #mutation_delta
                delta = random.uniform(0.9,1.1)
                new_feature = overfit[feature] * delta

                if(new_feature < -10):
                    new_feature = -10
                elif new_feature > 10:
                    new_feature = 10

                generation[i][feature] = new_feature
    
    return generation



# CALL SERVER:

def call_server(generation):

    error = np.zeros(shape = (POP, 2))
    
    
    # for i in range(len(generation)):
        

    #    curr_error = get_errors(SECRET_KEY,generation[i].tolist())
    #    error[i] = np.array(curr_error)



    for i in range(len(generation)):

        curr_error = [random.uniform(1, 10), random.uniform(1, 10)]
        error[i] = curr_error

    return error
    

# FITNESS FUNCTION

def fitness_function(error):

    fitness = np.zeros(shape = (POP))

    # fitness is sum of training and validation errors
    for i in range(len(error)):
        curr_error = error[i]
        fitness[i] = ((curr_error[0] + curr_error[1]))
    
    return fitness

def sort_generation(generation,fitness):

    sorted_idx = np.argsort(fitness)

    generation = generation[sorted_idx]

    fitness = np.sort(fitness)



    return generation,fitness


def selection(generation):

    pool = np.zeros(shape = (int(POP/2),FEATURE))

    pool[:5] = generation[:5]
    
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
        
        # it's ok if both parents are the same -> high mutation probability
        parent1 = parents[0]
        parent2 = parents[1]
        
        child1 = np.zeros(shape = (FEATURE))

        index = random.randint(2,9)

        child1[:index] = parent1[:index]
        child1[index:] = parent2[index:]

        crossOver_generation[i] = child1

        child2 = np.zeros(shape = (FEATURE))

        child2[:index] = parent1[:index]
        child2[index:] = parent2[index:]

        crossOver_generation[i+1]= child2

        i += 2

    return crossOver_generation


def mutation(crossOver_generation, mutation_prob):

    for child in crossOver_generation:

        for feature_index in range(FEATURE):

            val = random.uniform(0, 1)

            if(val <= mutation_prob):

                delta = random.uniform(0.9,1.1)
                new_feature = child[feature_index]* delta

                if(new_feature < -10):
                    new_feature = -10
                elif new_feature > 10:
                    new_feature = 10
                
                child[feature_index] = new_feature

        
    mutated_generation = crossOver_generation

    return mutated_generation

def create_new_gen(generation,mutated_generation,fitness,child_fitness):

    new_generation = np.zeros(shape = POP + int(POP/2) )

    # parent top5 pool and mutated generations fitness
    both_fitness = np.concatenate( (generation[:5],mutated_generation), axis=0)

    # parent top5 vectors and mutated generation vectors
    pot_new_generation = np.concatenate( (fitness[:5],child_fitness), axis = 0)

    top10 = sort_generation(pot_new_generation, both_fitness)

    new_generation = top10[:10]

    return new_generation

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


    with open('../output_files/24-3.txt', 'a+') as write_file:

        if(text == 'generation'):
            write_file.write("\n\n")
            write_file.write('ITERATION ' + str(iter))
            write_file.write("\n")

        
        write_file.write("\n")
        write_file.write(text)
        write_file.write("\n")

        json.dump(vector.tolist(),write_file)
            

generation = generate_initial(overfit, 0.9)


for iter in range(24):

    write_to_file(generation,'generation',iter)

    # call server
    error = call_server(generation)
    write_to_file(error,'error',iter)

    # call fitness - sum of errors 
    fitness = fitness_function(error)

    #sort generation in order of fitness and sort fitness
    generation,fitness = sort_generation(generation,fitness)
 
    # call selection
    pool = selection(generation)
    write_to_file(pool,'pool',iter)


    # call crossover
    crossOver_generation = crossover(pool)
    write_to_file(crossOver_generation,'crossOver_generation',iter)


    # call mutation
    mutated_generation = mutation(crossOver_generation,0.7)
    write_to_file(mutated_generation,'mutated_generation',iter)

   
    # call get_errors for mutated generation
    child_errors = call_server(mutated_generation)

    
    # calculate fitness for mutated generation
    child_fitness = fitness_function(child_errors)

    # take top 10 from ( mutated generation + top 5 of parents )
    new_generation = create_new_gen(generation,mutated_generation,fitness,child_fitness)



    generation = new_generation
    

    
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


