import numpy as np
import random
from client import *
import json
from tabulate import tabulate

overfit_vector  = [1,1,1,1,1]


POP = 6
FEATURE = 2
MUTATE_PROB = 0.9
MUTATE_RANGE = [0.9,1.1]
POOL_SIZE = 3
PARENT = 2
CHILD = 4
ITERATIONS = 5


HEADER1 = ["POPULATION","TRAINING ERRROR", "VALIDATION ERRROR", "FITNESS"]

HEADER2 = ["PARENT1", "PARENT2","CROSSOVER VECTOR","MUTATED CHILD"]

def write_file(table_values,headers,text):

    table= tabulate(table_values,headers,tablefmt="fancy_grid")
    with open('a.txt', 'a+') as f:
        f.write(str("\n\n") + text + str("\n\n") + table)




def dumpLast(fitness,generation):

   
    data = {"Generation" : generation.tolist(), "Fitness" : fitness.tolist()}
    with open('a.json','w') as f:
        json.dump(data,f, indent=5)

def getLast():
    f = open('a.json')
    data = json.load(f)
    generation = np.array(data["Generation"])
    fitness = np.array(data["Fitness"])

    return fitness, generation


def generate_initial(vector):

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


def fitness_function(generation):

    fitness = np.zeros(shape = (POP,3))
    for i in range(POP):
        error = [random.uniform(0,1),random.uniform(0,1)]

        #error = get_errors(SECRET_KEY,generation[i].tolist())
        fitness[i][0] = error[0]
        fitness[i][1] = error[1]
        fitness[i][2] = (error[0] + error[1])

    sorted_idx = np.argsort(fitness[:,-1])
    
    fitness = np.column_stack((generation,fitness))
    fitness = fitness[sorted_idx]

    generation = generation[sorted_idx]

    #generate table for fitness
    table = np.zeros((POP,4),dtype=object)

    for i in range(POP):
        table[i][0] = generation[i]
        table[i][1] = fitness[i][FEATURE]
        table[i][2] = fitness[i][FEATURE+1]
        table[i][3] = fitness[i][FEATURE+2]

    write_file(table,HEADER1,"GENERATION FITNESS AND ERRORS")

    return fitness,generation

def selection(generation):
    
    pool = np.zeros(shape = (POOL_SIZE,FEATURE))
    pool = generation[:POOL_SIZE]

    return pool

def crossover(pool):

    n = 5
    crossOver_generation = np.zeros(shape= (POP,FEATURE))

    i = 0
    table = np.zeros((POP,4),dtype=object)
    while i < POP:

        parent1 = random.choice(pool)
        parent2 = random.choice(pool)

        table[i][0] = parent1
        table[i][1] = parent2

        table[i+1][0] = parent1
        table[i+1][1] = parent2


        u = random.uniform(0,1)
    
        if (u < 0.5):
            b = (2 * u)**((n + 1)**-1)
        else:
            b = ((2*(1-u))**-1)**((n + 1)**-1)

        child1 = 0.5*((1 + b) * parent1 + (1 - b) * parent2)
        child2 = 0.5*((1 - b) * parent1 + (1 + b) * parent2)
        
        table[i][2] = child1
        table[i+1][2] = child2

        crossOver_generation[i]= child1
        crossOver_generation[i+1] = child2

        i += 2
    
    return crossOver_generation,table

def mutation(crossOver_generation,table):
    i = 0
    for child in crossOver_generation:
        for feature_index in range(FEATURE):

            prob = random.uniform(0, 1)
            if(prob <= MUTATE_PROB):

                delta = random.uniform(MUTATE_RANGE[0],MUTATE_RANGE[1])
                new_feature = child[feature_index]* delta

                child[feature_index] = new_feature

        table[i][3] = child
        i+=1
    
    write_file(table,HEADER2,"CREATING CHILDREN")
    return crossOver_generation


def generate_children(pool):

   crossOver_generation,table = crossover(pool)
   children = mutation(crossOver_generation,table)

   return children

def create_newGeneration(parents_fitness,children_fitness,iter):


    parents_fitness = parents_fitness[:PARENT]
    children_fitness = children_fitness[:CHILD]
    generation_fitness = np.concatenate((parents_fitness,children_fitness))

    sorted_idx = np.argsort(generation_fitness[:,-1])

    generation_fitness = generation_fitness[sorted_idx]

    new_generation = generation_fitness[:,:FEATURE]

    #generate table for fitness
    table = np.zeros((POP,4),dtype=object)

    for i in range(POP):
        table[i][0] = new_generation[i]
        table[i][1] = generation_fitness[i][FEATURE]
        table[i][2] = generation_fitness[i][FEATURE+1]
        table[i][3] = generation_fitness[i][FEATURE+2]

    write_file(table,HEADER1,str("\n\nITERATION :  " + str(iter+1) + "\n\n\nNEXT GENERATION"))

    

    return generation_fitness,new_generation



def main():

    
    line =''
    for i in range(200):
        line = line + str('*')

    with open('a.txt', 'a+') as write_file:
        write_file.write(str('\n\n') +str(line)+ str('\n\n' + str("RUN NUMBER : 1")))
    

    '''
    vector = overfit_vector
    generation = generate_initial(vector)
    

    #This returns the sorted fitness and generation.
    fitness, generation = fitness_function(generation)

    '''
    

    fitness, generation = getLast()
    


    for iter in range(ITERATIONS):

        pool = selection(generation)

        children = generate_children(pool)

        #Calculate fitness function of children
        children_fitness,children = fitness_function(children)

        #New generation gets created from parent and children values
        new_fitness,new_generation = create_newGeneration(fitness,children_fitness,iter)

        fitness = new_fitness
        generation = new_generation

    

        dumpLast(fitness,generation)

        table = np.zeros((4),dtype=object)
        table[0] = generation[0]
        table[1] = fitness[0][FEATURE]
        table[2] = fitness[0][FEATURE+1]
        table[3] = fitness[0][FEATURE+2]
        
        
        best_file(table,HEADER1,str('ITERATION : ' + str(iter)))
        


if __name__ == '__main__':
    main()
