import numpy as np
import random
from client import *
import json
import matplotlib.pyplot as plt
from tabulate import tabulate


POP = 20
FEATURE = 11


overfit_vector = np.array([0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,
           8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10])


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
    with open('../output_files/26/26-2.txt', 'a+') as write_file:
        write_file.write((str('\n\nITERATION  :  ') + str(iter) + str('\n\n')))
        write_file.write(table)
    

def write_to_file_best(best_error,best_gen,best_fitness,iter):

    iter += 1
    with open('../output_files/26/26-final.txt', 'a+') as write_file:

        write_file.write((str('\n\nITERATION  :  ') + str(iter) + str('\n\n')))
        
        write_file.write((str('\nERROR :  ') +  str('\n')))    
        json.dump(best_error.tolist(),write_file)

        write_file.write((str('\nGENERATION :  ') + str('\n')))    
        json.dump(best_gen.tolist(),write_file)

        write_file.write((str('\nFITNESS :  ') + str('\n')))    
        json.dump(best_fitness.tolist(),write_file)

        write_file.write("\n\n\n")
    

def create_type(c,n):


    arr = np.zeros(shape = (n))
    for i in range(n):
        arr[i] = c
    
    return arr

    
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


# FITNESS FUNCTION
def fitness_function(error):

    fitness = np.zeros(shape = (POP))

    # fitness is sum of training and validation errors
    for i in range(len(error)):
        curr_error = error[i]
        fitness[i] = ((curr_error[0] + curr_error[1]))
    
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

    #POOL IS THE TOP int(POP/2) PARENTS

    pool = np.zeros(shape = (int(POP/2),FEATURE))
    pool = generation[:(int(POP/2))]
    
    return pool


def pick_two(pool):

    parent1 = random.choice(pool)
    parent2 = random.choice(pool)

    return parent1,parent2


#CROSSOVER
def crossover(pool):

    crossOver_generation = np.zeros(shape= (POP,FEATURE))

    i = 0
    while i < POP:

        parent1, parent2 =  pick_two(pool)
        
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

    # new_errors = np.zeros(shape = (POP,2) )


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


    # write to output file what we are doing in this GA
    line =''
    for i in range(200):
        line = line + str('*')
        
    with open('../output_files/26/26-2.txt', 'a+') as write_file:
        write_file.write(str(line)+ str('\n\n'))
        write_file.write((str('GENERATIONS: 20 | ITERATIONS: 24 | MUTATION PROBABILITY: 0.7 increased to 0.85') + str('\n\n\n\n\n')))

    with open('../output_files/26/26-final.txt', 'a+') as write_file:
        write_file.write(str(line)+ str('\n\n'))
        write_file.write((str('GENERATIONS: 20 | ITERATIONS: 24 | MUTATION PROBABILITY: 0.7 increased to 0.85') + str('\n\n\n\n\n')))
                

    #generate initial generation
    vector = overfit_vector

    MUTATE_PROB = 0.9
    MUTATE_RANGE = np.array([0.9,1.1])
    ITERATIONS = 24

    generation = generate_initial(vector,MUTATE_PROB,MUTATE_RANGE)

    #server call for initial generation
    error = call_server(generation)

    #get the fitness value of every vector in the generation
    fitness = fitness_function(error)

    #sort the errors,fitness and generation corresponding to it
    generation, error, fitness, sorted_idx = sort_generation(generation,error,fitness)


    min_error1 = np.zeros(shape =ITERATIONS+1)
    min_error2 = np.zeros(shape =ITERATIONS+1)
    min_fitness = np.zeros(shape =ITERATIONS+1)

    min_vector = np.zeros(shape = (ITERATIONS+1,11))


    best_error = error[0]
    best_gen = generation[0]
    best_fitness = fitness[0]

    min_error1[0] = best_error[0]
    min_error2[0] = best_error[1]

    min_fitness[0] = best_fitness
    min_vector[0] = best_gen


    type = create_type(0,POP)
    write_to_file(generation,error,fitness,-1,type)
    
    write_to_file_best(best_error,best_gen,best_fitness,-1)

    MUTATE_PROB = 0.7

    for iter in range(ITERATIONS):

        if(iter != 0 and iter%2 == 0):
            MUTATE_PROB += 0.015

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
        new_generation, child_error, child_fitness,sorted_idx = sort_generation(mutated_generation,child_error,child_fitness)


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

        write_to_file_best(best_error,best_gen,best_fitness,iter)

        if(iter == ITERATIONS - 1):
            

            iters = range(ITERATIONS+1)

            plt.plot(iters,min_error1, label = 'training')
            plt.plot(iters,min_error2, label = 'validation')
            plt.plot(iters,min_fitness, label = 'fitness')
            plt.legend() 
            plt.xlabel("iterations")
            plt.ylabel("errors")
            plt.savefig('../output_files/26/expand-26.jpeg') 

            return best_gen, best_fitness, best_error


##CALL FUNCTION

gen1 , fitness1, error1 = main_loop()

print(error1)







