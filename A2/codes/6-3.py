import numpy as np
import random
from client import *
import json
from tabulate import tabulate

POP = 20
FEATURE = 11
MUTATE_PROB = 0.3
MUTATE_RANGE = [0.97,1.02]
POOL_SIZE = 5
PARENT = 5
CHILD = POP - PARENT
ITERATIONS = 8

NC = 5

TRAINING_FACTOR = -1
VALIDATION_FACTOR = 1

HEADER1 = ["INDEX","POPULATION","TRAINING ERRROR", "VALIDATION ERRROR", "FITNESS"]
HEADER2 = ["PARENT INDEXES","PARENT1", "PARENT2","CROSSOVER VECTOR","MUTATED CHILD"]
HEADER3 = ["INDEX", "PARENT"]

TRACE_FILE = '../output_files/14-3/trace.txt'
BEST_FILE = '../output_files/14-3/best_vector.txt'

RUN = 25

overfit_vector = np.array([0.0, -1.45799022e-12, -2.28980078e-13,  4.62010753e-11, -1.75214813e-10, -1.83669770e-15,
           8.52944060e-16,  2.29423303e-05, -2.04721003e-06, -1.59792834e-08,  9.98214034e-10])


def write_file(table,headers,text,filename):

    with open(filename, 'a+') as f:
        f.write(str("\n\n") + text + str("\n\n") + table)


def dumpLast(fitness,generation):
  
    data = {"Generation" : generation.tolist(), "Fitness" : fitness.tolist()}
    with open('../output_files/14-3/lastVec.json','w') as f:
        json.dump(data,f, indent=5)

def getLast():
    f = open('../output_files/13-3/lastVec.json')
    with open('../output_files/14-3/lastVec.json','w') as f:
        json.dump(data,f, indent=5)

def getLast():
    f = open('../output_files/14-3/lastVec.json')
    data = json.load(f)
    generation = np.array(data["Generation"])
    fitness = np.array(data["Fitness"])

    return fitness, generation


def makeTable(fitness,text,filename,type):

    
    #generate table for fitness
    table = np.zeros((len(fitness),5),dtype=object)
   

    for i in range(len(fitness)):

        if type == 0: table[i][0] = str('P' + str(i))
        if type == 1: table[i][0] = str('C' + str(i))
        if type == -1: table[i][0] = i
        table[i][1] = fitness[i][:FEATURE]
        table[i][2] = fitness[i][FEATURE]
        table[i][3] = fitness[i][FEATURE+1]
        table[i][4] = fitness[i][FEATURE+2]
       

    table= tabulate(table,HEADER1,tablefmt="fancy_grid")
    write_file(table,HEADER1,text,filename)


def generate_initial(vector):

    generation = np.zeros(shape = (POP,FEATURE))
    for i in range(POP):
        for feature in range(FEATURE):

            val = vector[feature]
            prob = random.uniform(0,1)
            if(prob <= 0.9):
                
                delta = random.uniform(MUTATE_RANGE[0],MUTATE_RANGE[1])
                val = val * delta

            generation[i][feature] = val
    
    return generation

def call_server(generation):

    fitness = np.zeros(shape = (POP,3))
    for i in range(POP):
        # error = [random.uniform(0,1),random.uniform(0,1)]
        error = get_errors(SECRET_KEY,generation[i].tolist())
        fitness[i][0] = error[0]
        fitness[i][1] = error[1]

    fitness = np.column_stack((generation,fitness))
    return fitness


def fitness_function(fitness, generation,type):

    for i in range(POP):
        fitness[i][FEATURE+2] = abs(fitness[i][FEATURE] -fitness[i][FEATURE+1])
        
    sorted_idx = np.argsort(fitness[:,-1])
    fitness = fitness[sorted_idx]
    generation = generation[sorted_idx]
    
    if type == 0: makeTable(fitness,"GENERATION FITNESS AND ERRORS",TRACE_FILE,type)
    if type == 1: makeTable(fitness,"CHILD GENERATION FITNESS AND ERRORS",TRACE_FILE,type)

    return fitness,generation


def get_fitness(generation,type):

    fitness = call_server(generation)
    fitness, generation = fitness_function(fitness, generation,type)

    return fitness, generation


def selection(generation):
    
    pool = np.zeros(shape = (POOL_SIZE,FEATURE))
    pool = generation[:POOL_SIZE]

    table = np.zeros((POOL_SIZE,2),dtype=object)

    for i in range(POOL_SIZE):
        table[i][0] = str('P' + str(i))
        table[i][1] = pool[i]
    
    table= tabulate(table,HEADER3,tablefmt="fancy_grid")
    write_file(table,HEADER3,"SELECTED MATING POOL",TRACE_FILE)
    # print(table)

    return pool

def crossover(pool):
   
    crossOver_generation = np.zeros(shape= (POP,FEATURE))

    i = 0
    table = np.zeros((POP,5),dtype=object)
    while i < POP:

        p1 = random.choice(list(enumerate(pool)))
        p2 = random.choice(list(enumerate(pool)))

        p1idx = "P" + str(p1[0])
        parent1 = p1[1]
        p2idx = "P" + str(p2[0])
        parent2 = p2[1]

        #parent indexes
        table[i][0] = p1idx + ", " + p2idx
        table[i+1][0] = p1idx + ", " + p2idx

        table[i][1] = parent1
        table[i][2] = parent2

        table[i+1][1] = parent1
        table[i+1][2] = parent2


        u = random.uniform(0,1)
    
        if (u < 0.5):
            b = (2 * u)**((NC + 1)**-1)
        else:
            b = ((2*(1-u))**-1)**((NC + 1)**-1)

        child1 = 0.5*((1 + b) * parent1 + (1 - b) * parent2)
        child2 = 0.5*((1 - b) * parent1 + (1 + b) * parent2)
        
        table[i][3] = child1
        table[i+1][3] = child2

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

        table[i][4] = child
        i+=1
    
    table= tabulate(table,HEADER2,tablefmt="fancy_grid")
    write_file(table,HEADER2,"CREATING CHILDREN",TRACE_FILE)
    return crossOver_generation


def generate_children(pool):

   crossOver_generation,table = crossover(pool)
   children = mutation(crossOver_generation,table)

   return children

def create_newGeneration(parents_fitness,children_fitness,iter):

    parents_fitness = parents_fitness[:PARENT]
    children_fitness = children_fitness[:CHILD]
    new_fitness = np.concatenate((parents_fitness,children_fitness))

    sorted_idx = np.argsort(new_fitness[:,-1])
    new_fitness = new_fitness[sorted_idx]
    new_generation = new_fitness[:,:FEATURE]

    makeTable(new_fitness,str("\n\nITERATION :  " + str(iter+1) + "\n\n\nNEXT GENERATION"),TRACE_FILE,0)

    return new_fitness,new_generation

 
def main():

    line =''
    for i in range(200):
        line = line + str('*')

    with open(TRACE_FILE, 'a+') as write_file:
        write_file.write(str('\n\n') +str(line)+ str('\n\n' + str("RUN NUMBER : ") + str(RUN)))


    '''

    vector = overfit_vector
    generation = generate_initial(vector)
    

    #This returns the sorted fitness and generation.
    fitness, generation = get_fitness(generation,0)

    '''
    
    fitness, generation = getLast()
    fitness, generation = fitness_function(fitness,generation,0)

    best_fitness = np.zeros(shape = (ITERATIONS+1,FEATURE+3))
    best_fitness[0] = fitness[0]

    for iter in range(ITERATIONS):

        pool = selection(generation)

        children = generate_children(pool)

        #Calculate fitness function of children
        children_fitness,children = get_fitness(children,1)

        #New generation gets created from parent and children values
        new_fitness,new_generation = create_newGeneration(fitness,children_fitness,iter)

        fitness = new_fitness
        generation = new_generation

        dumpLast(fitness,generation)
        best_fitness[iter+1] = fitness[0]
    
    makeTable(best_fitness,str("BEST VECTORS OF RUN :  " + str(RUN)),BEST_FILE,-1)
        

if __name__ == '__main__':
    main()
