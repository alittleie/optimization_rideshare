import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import choice
import time
from gurobipy import *
start_time = time.time()

num_rideshare = 1
num_evacuate = 8
num_exit = 1
balnce_value = 2.25


rnd = np.random
rnd.seed(112)

n= num_rideshare + num_evacuate + num_exit
xc = []
yc = []

for i in range(0,n+1):
    xc.append((rnd.random())*200)
    yc.append((rnd.random())*100)

weight = []
capacity = []
max_node_cap = 5

exit_x = xc[num_rideshare]
exit_y = yc[num_rideshare]
xc.remove(exit_x)
yc.remove(exit_y)
xc.append(exit_x)
yc.append(exit_y)

for i in range(num_rideshare,num_rideshare + num_evacuate+1):
    weight.append(rnd.randint(1,11))
    capacity.append(rnd.randint(1,max_node_cap))
distance_between_node = []

for j in range(0,n+1):
    node_hold = []
    for i in range(0,n+1):
        d = (math.sqrt(((xc[i] - xc[j]) ** 2) + ((yc[i] - yc[j]) ** 2)))
        node_hold.append(d)
    distance_between_node.append(node_hold)
capacity0 =  capacity
weight0 =  weight
veh_cap = []
for i in range(0, num_rideshare):
    veh_cap.append(4)
    capacity = [0]+capacity
    weight = [0]+weight




Arc = [(i, j) for i in range(0, n+1) for j in range(0, n+1)]


A = []
for i in range(0,num_rideshare):
    A.append(i)

# B=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
B = []
for i in range(num_rideshare,num_rideshare+ num_evacuate+1):
    B.append(i)

#O = [16]
O = []
for i in range(num_rideshare+ num_evacuate+1,num_rideshare+num_evacuate+ num_exit+1):
    O.append(i)

AB = A+B
ABO = AB + O
D = weight
print(D)

C = capacity
T = distance_between_node


Tload = 0
Tmax = 250
Cmax = 4
print(capacity0)
print(capacity)

location_AB = len(AB)
location_A = len(A)
location_ABO = len(ABO)
print(location_AB)
print(location_A)
print(location_ABO)
m = Model()
# whether a node is traveled to
d = m.addVars(location_AB, vtype=GRB.BINARY, name='d')
# which paths from i to j are taken
x = m.addVars(location_ABO, location_ABO, vtype=GRB.BINARY, name='x')
u = m.addVars(location_AB, vtype=GRB.CONTINUOUS, name='u')
v = m.addVars(location_AB, vtype=GRB.CONTINUOUS, name='v')
m.update()
m.setAttr('ModelSense',-1)
m.setObjectiveN(sum(D[i] * (d[i]) for i in range(0, location_AB))+sum(C[i]**balnce_value * (d[i]) for i in range(0, location_AB)),1)


m.addConstrs((sum(x[i, j] for j in range(0, location_ABO) if i != j) == d[i] for i in range(0, location_AB)), name='c0')
m.addConstrs((sum(x[i, j] for i in range(0, location_AB) if i != j) == d[j] for j in range(location_A, location_AB)),name='c1')
m.addConstr((sum(x[i, j] for i in range(0, location_AB) for j in range(location_AB, location_ABO) if i != j) == sum( d[i] for i in range(0, location_A))), name='c2')
m.addConstrs((u[j] >= u[i] +C[j] - 1000 * (1 - x[i,j]) for i in range(0,location_AB) for j in range(0,location_AB) if i !=j ), name = 'c3')
m.addConstrs((v[j] >= v[i] + T[i][j] + Tload * C[j] - 1000 * (1 - x[i, j]) for i in range(0, location_AB) for j in range(0, location_AB) if i != j), name='c4')
m.addConstrs((v[i] <= Tmax - T[i][j] * x[i, j] for i in range(0, location_AB) for j in range(location_AB, location_ABO) if i != j), name='c5')
m.addConstrs((u[i] <= Cmax for i in range(0, location_AB)), name='c6')
m.addConstrs((u[i] == C[i] * d[i] for i in range(0, location_A)), name='c7')

m.optimize()
print("--- %s seconds ---" % (time.time() - start_time))
m.printAttr('x')
print("Optimal Objective Value", m.objVal)
start = []
end = []


for v in m.getVars():
    if 'x[' in str(v.varName) and v.x == 1:
        print(v)

        num = str(v.varName).strip('x[')
        num = num.strip(']')
        num = num.split(',')
        print(num[0])
        start.append(int(num[0]))
        end.append(int(num[1]))
        print(num[0],'  ',num[1])



for i in range(0,num_rideshare):
    plt.plot(xc[i], yc[i], c='#043fa1', marker='X')
print(num_rideshare+ num_evacuate)
print(num_rideshare+num_evacuate+ num_exit)
for i in range((num_rideshare+ num_evacuate+1),(num_rideshare+num_evacuate+ num_exit+1)):
    print (i)
    plt.plot(xc[i], yc[i], c='#28e739', marker='X')
plt.scatter(xc[num_rideshare:num_rideshare+num_evacuate+1], yc[num_rideshare:num_rideshare+num_evacuate+1], c='#e61207')
for i in range(len(weight0)):
    plt.annotate(weight0[i],[xc[i+num_rideshare], yc[i+num_rideshare]], xytext=(xc[i+num_rideshare]+1.5, yc[i+num_rideshare]+1.5))
    plt.annotate(capacity0[i], [xc[i + num_rideshare], yc[i + num_rideshare]], xytext=(xc[i + num_rideshare] + 5, yc[i + num_rideshare] + 5),c='b')
for i in range(0,len(start)):
    plt.plot([xc[start[i]], xc[end[i]]], [yc[start[i]], yc[end[i]]],c='b',alpha=0.5)
print(xc)
print(yc)
plt.show()