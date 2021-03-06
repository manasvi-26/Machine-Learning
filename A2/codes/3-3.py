import numpy as np
import random
from client import *
import json
import matplotlib.pyplot as plt
from tabulate import tabulate
import os

POP = 12
FEATURE = 11

# overfit_vector = np.array([0.00000000e+00, -1.53575443e-12, -2.38064603e-13,  4.67653589e-11,
# -1.84927508e-10 ,-1.97048684e-15  ,8.63845706e-16 , 2.27066239e-05,-2.07342035e-06 ,-1.67377079e-08  ,1.00508477e-09])

# generating new
overfit_vector = [ 0.00000000e+00, -1.64379458e-12, -2.36774131e-13,  4.78004881e-11, -1.82070627e-10, -1.84252091e-15,  7.10170109e-16,  2.21299915e-05, -1.80994584e-06, -1.62770324e-08, 8.99071483e-10]

def get_prev():

    f = open('../output_files/3-3/endVec.json')
    data = json.load(f)
    generation = np.array(data["Generation"])
    error = np.array(data["Error"])
    fitness = np.array(data["Fitness"])

    return generation,error,fitness

def write_to_json(generation,error,fitness):
   
    data = {"Generation" : generation.tolist(), "Error" : error.tolist() , "Fitness" : fitness.tolist()}
    with open('../output_files/3-3/endVec.json','w') as f:
        json.dump(data,f, indent=5)


def write_to_file(generation, error, fitness,iter,type):

    iter+=1
    table_values = np.zeros((POP,5),dtype=object)

    for i in range(POP):

        table_values[i][0] = generation[i]
        table_values[i][1] = error[i][0]
        table_values[i][2] = error[i][1]
        table_values[i][3] = fitness[i]
        table_values[i][4] = type[i] 

    headers = ["GENERATION","TRAINING ERRROR", "VALIDATION ERRROR", "FITNESS", "TYPE"]
    table= tabulate(table_values,headers,tablefmt="fancy_grid")
    with open('../output_files/3-3/3-3.txt', 'a+') as write_file:
        write_file.write((str('\n\nITERATION  :  ') + str(iter) + str('\n\n')))
        write_file.write(table)

def create_type(c,n):

    arr = np.zeros(shape = (n))
    for i in range(n):
        arr[i] = c
    
    return arr

#CALL SERVER
def call_server(generation):

    error = np.zeros(shape = (POP, 2))
    
    for i in range(len(generation)): 

       curr_error = get_errors(SECRET_KEY,generation[i].tolist())
       error[i] = np.array(curr_error)

    # for i in range(len(generation)):

    #     curr_error = [random.uniform(1, 10), random.uniform(1, 10)]
    #     error[i] = curr_error

    return error


def write_to_file_best(min_error1,min_error2,min_vector,min_fitness,iterations):
    
    table_values = np.zeros((iterations,5),dtype=object)

    for i in range(iterations):

        table_values[i][0] = i
        table_values[i][1] = min_vector[i]
        table_values[i][2] = min_error1[i]
        table_values[i][3] = min_error2[i]
        table_values[i][4] = min_fitness[i]

    headers = ["ITERATION NUMBER","BEST VECTOR","BEST TRAINING ERRROR", "BEST VALIDATION ERRROR", "BEST FITNESS"]
    table= tabulate(table_values,headers,tablefmt="fancy_grid")
    
    with open('../output_files/3-3/3-3-final.txt', 'a+') as write_file:
        # write_file.write((str('\n\nITERATION  :  ') + str(iter) + str('\n\n')))
        write_file.write(table)


def generate_initial(vector,MUTATE_PROB,MUTATE_RANGE):

    generation = np.zeros(shape = (POP,FEATURE))

    for i in range(POP):
        for feature in range(FEATURE):

            val = vector[feature]
            prob = random.uniform(0,1)
            if(prob <= MUTATE_PROB):
                
                delta = random.uniform(MUTATE_RANGE[0],MUTATE_RANGE[1])
                val = val * delta
            
            generation[i][feature] = val
   
    return generation



# FITNESS FUNCTION
def fitness_function(error):

    fitness = np.zeros(shape = (POP))

    # fitness is sum of training and validation errors
    for i in range(len(error)):
        curr_error = error[i]
        training = curr_error[0]
        validation = curr_error[1]
        fitness[i] = (( 0.7*training + validation ))
    
    return fitness


#SORT ERRORS, FITNESS AND GENERATION
def sort_generation(generation, error, fitness):

    sorted_idx = np.argsort(fitness)

    generation = generation[sorted_idx]
    error = error[sorted_idx]
    fitness = fitness[sorted_idx]

    return generation, error, fitness,sorted_idx

#SELECTION
def selection(generation):

    selection_var = 4
    pool = np.zeros(shape = (selection_var,FEATURE))
    pool = generation[:selection_var]

    return pool


def pick_two(pool):

    parent1 = random.choice(pool)
    parent2 = random.choice(pool)

    return parent1,parent2

#CROSSOVER
def crossover(pool):

    n_c = 3

    crossOver_generation = np.zeros(shape= (POP,FEATURE))

    i = 0
    while i < POP:

        parent1, parent2 =  pick_two(pool)

        u = random.uniform(0,1)
        
        if (u < 0.5):
            beta = (2 * u)**((n_c + 1)**-1)
        else:
            beta = ((2*(1-u))**-1)**((n_c + 1)**-1)

        child1 = 0.5*((1 + beta) * parent1 + (1 - beta) * parent2)
        child2 = 0.5*((1 - beta) * parent1 + (1 + beta) * parent2)

        crossOver_generation[i]= child1
        crossOver_generation[i+1] = child2

        i += 2
    
    return crossOver_generation


def mutation(crossOver_generation,MUTATE_PROB,MUTATE_RANGE):

    for child in crossOver_generation:
        for feature_index in range(FEATURE):

            prob = random.uniform(0, 1)

            if(prob <= MUTATE_PROB):

                delta = random.uniform(MUTATE_RANGE[0],MUTATE_RANGE[1])
                new_feature = child[feature_index]* delta

                if(new_feature < -10):
                    new_feature = -10
                elif new_feature > 10:
                    new_feature = 10
                
                child[feature_index] = new_feature

        
    mutated_generation = crossOver_generation

    return mutated_generation

def create_new_gen(generation,mutated_generation,error,child_error,fitness,child_fitness):

    new_generation = np.zeros(shape = (POP,FEATURE ))
    new_fitness = np.zeros(shape = POP )
    new_error = np.zeros(shape = (POP,2))

    # parent top5 pool and mutated generations vector
    pot_new_generation  = np.concatenate( (generation[:(int(POP/2))],mutated_generation), axis=0)

    # parent top5 vectors and mutated generation fitness
    pot_new_generation_fitness = np.concatenate( (fitness[:(int(POP/2))],child_fitness), axis = 0)

    pot_new_generation_error = np.concatenate((error[:(int(POP/2))],child_error), axis = 0)

    topgen,toperror,topfitness,sorted_idx = sort_generation(pot_new_generation, pot_new_generation_error, pot_new_generation_fitness)

    new_generation = topgen[:POP]
    new_error = toperror[:POP]
    new_fitness = topfitness[:POP]

    p = create_type(0,(int(POP/2)))
    c = create_type(1,POP)
    type  = np.concatenate((p,c),axis = 0)
    type = type[sorted_idx]
    type = type[:POP]

    return new_generation,new_error,new_fitness,type

def main_loop():

    #generate initial generation
    vector = overfit_vector

    MUTATE_PROB = 0.9
    MUTATE_RANGE = np.array([0.9,1.1])
    ITERATIONS = 6

    # generation = generate_initial(vector,MUTATE_PROB,MUTATE_RANGE)

    # #server call for initial generation
    # error = call_server(generation)

    '''
    this is for next part
    '''

    generation, error, fitness = get_prev()

    #get the fitness value of every vector in the generation
    fitness = fitness_function(error)

    #sort the errors,fitness and generation corresponding to it
    generation, error, fitness, sorted_idx = sort_generation(generation,error,fitness)

    min_error1 = np.zeros(shape =ITERATIONS+1)
    min_error2 = np.zeros(shape =ITERATIONS+1)
    min_fitness = np.zeros(shape =ITERATIONS+1)

    min_vector = np.zeros(shape = (ITERATIONS+1,FEATURE))

    best_error = error[0]
    best_gen = generation[0]
    best_fitness = fitness[0]

    min_error1[0] = best_error[0]
    min_error2[0] = best_error[1]

    min_fitness[0] = best_fitness
    min_vector[0] = best_gen

    type = create_type(0,POP)

    line =''
    for i in range(200):
        line = line + str('*')

    with open('../output_files/3-3/3-3.txt', 'a+') as write_file:
        write_file.write(str('\n\n') +str(line)+ str('\n\n'))
        write_file.write((str('GENERATIONS: 12 | ITERATIONS: 6 | RUN NUMBER: 8') + str('\n\n\n\n\n')))

    with open('../output_files/3-3/3-3-final.txt', 'a+') as write_file:
        write_file.write(str('\n\n')+ str(line)+ str('\n\n'))
        write_file.write((str('GENERATIONS: 12 | ITERATIONS: 6 | RUN NUMBER: 8') + str('\n\n\n\n\n')))


    write_to_file(generation,error,fitness,-1,type)

    # write_to_file_best(best_error,best_gen,best_fitness,-1)

    write_to_json(generation,error,fitness)

    MUTATE_PROB = 0.5

    for iter in range(ITERATIONS):

        #selection
        pool = selection(generation)

        #crossover 
        crossOver_generation = crossover(pool)

        #mutation
        curr_range = MUTATE_RANGE

        mutated_generation = mutation(crossOver_generation,MUTATE_PROB,curr_range)

        # call get_errors for mutated generation
        child_error = call_server(mutated_generation)

        # calculate fitness for mutated generation
        child_fitness = fitness_function(child_error)

        #sort the errors,fitness and generation corresponding to it
        mutated_generation, child_error, child_fitness,sorted_idx = sort_generation(mutated_generation,child_error,child_fitness)


        # take top 10 from ( mutated generation + top 5 of parents )
        new_generation,new_error,new_fitness,type = create_new_gen(generation,mutated_generation,error,child_error,fitness,child_fitness)

        generation = new_generation
        fitness = new_fitness
        error = new_error

        write_to_file(generation,error,fitness,iter,type)

        best_error = error[0]
        best_gen = generation[0]
        best_fitness = fitness[0]

        min_error1[iter+1] = best_error[0]
        min_error2[iter+1] = best_error[1]

        min_fitness[iter+1] = best_fitness
        min_vector[iter+1] = best_gen

        write_to_json(generation,error,fitness)

        if(iter == ITERATIONS - 1):
            
            write_to_file_best(min_error1,min_error2,min_vector,min_fitness,ITERATIONS)

            iters = range(ITERATIONS+1)

            plt.plot(iters,min_error1, label = 'training')
            plt.plot(iters,min_error2, label = 'validation')
            plt.plot(iters,min_fitness, label = 'fitness')
            plt.legend() 
            plt.xlabel("iterations")
            plt.ylabel("errors")
            plt.savefig('../output_files/3-3/expand1.jpeg') 

            return best_gen, best_fitness, best_error


##CALL FUNCTION

gen1 , fitness1, error1 = main_loop()



    