x = [1,2]
y = [2,3]

a =  []
a.append(x)
a.appedn(y)

with open('../output_files/bs.txt', 'w') as write_file:
        json.dump(list,write_file)