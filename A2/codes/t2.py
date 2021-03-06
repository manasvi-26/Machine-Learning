
    
    line =''
    for i in range(200):
        line = line + str('*')

    with open('a.txt', 'a+') as write_file:
        write_file.write(str('\n\n') +str(line)+ str('\n\n' + str("RUN NUMBER : 1")))
    



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






