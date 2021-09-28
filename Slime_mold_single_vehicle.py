import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import choice
import time
start_time = time.time()

rnd = np.random
rnd.seed(114)

n= 50
xc = []
yc = []
for i in range(0,n+1):
    xc.append((rnd.random())*200)
    yc.append((rnd.random())*100)

weight = []
capacity = []
max_node_cap = 5

for i in range(2,n+1):
    weight.append(rnd.randint(1,11))
    capacity.append(rnd.randint(1,max_node_cap))

max_growth = 1000
growth_rate =20
max_distance = 200
capcity_rate = .7
anaerobic_rate = 10
cmax = 10
path_cost_keep = []
path_keep = []
path_out_keep = []
max_evac_keep = 0
capcity_keep = 0

for p in range(0,1000):
    node_i_mem = []
    path_cost = []
    path_out_cost = []
    path = []
    evac_score = []
    r = rnd.randint(10,max_growth)
    x_current = xc[0]
    y_current = yc[0]
    capacity_count = 0
    O2 = (capacity_count/cmax)+1* anaerobic_rate
    t=0
    while capacity_count < cmax and t < n+1:

        r = r+ growth_rate
        path_dic = {}
        node_keep = []
        # plt.plot(xc[0], yc[0], c='#043fa1', marker='X')
        # plt.plot(xc[1], yc[1], c='#28e739', marker='X')
        # plt.scatter(xc[2:], yc[2:], c='#e61207')
        for i in range(2,n+1):
            d = (math.sqrt(((xc[i]-x_current)**2)+((yc[i]-y_current)**2)))
            d_out = (math.sqrt(((xc[i]-xc[1])**2)+((yc[i]-yc[1])**2)))

            if d <= r and i not in node_i_mem :

                node_keep.append(i)
                # plt.plot(xc[i], yc[i], c='b', marker='x')
                path_dic[i] = [d,d_out,xc[i],yc[i],weight[i-2],capacity[i-2]]
        # plt.show()
        distance = {}
        Smell = {}
        Capa = {}
        d_out_hold = {}
        for i in range(len(node_keep)):
            distance[node_keep[i]]= path_dic[node_keep[i]][0]
            Smell[node_keep[i]]= path_dic[node_keep[i]][4]
            Capa[node_keep[i]]= path_dic[node_keep[i]][5]
            d_out_hold[node_keep[i]]= path_dic[node_keep[i]][1]

        distance_keys = list(distance.keys())
        distance_values = list(distance.values())
        smell_keys = list(Smell.keys())
        smell_values = list(Smell.values())
        d_out_hold_keys = list(d_out_hold.keys())
        d_out_hold_values = list(d_out_hold.values())

        distance_probs = []
        smell_probs = []
        d_out_probs = []
        for i in range(len(distance_keys)):
            distance_probs.append(1/(distance_values[i]/sum(distance_values)))
            d_out_probs.append(1/(d_out_hold_values[i]/sum(d_out_hold_values)))
            smell_probs.append(smell_values[i]/sum(smell_values))
        d_probs_new =[]
        for i in range(len(d_out_probs)):
            d_probs_new.append(d_out_probs[i]*O2)
        weight_list = []
        for i in range(len(d_probs_new)):
            weight_list.append((distance_probs[i]/sum(distance_probs)) * (d_probs_new[i]/sum(d_probs_new)) * (smell_probs[i]/sum(smell_probs)))
        prob_final = []
        for i in range(len(weight_list)):
            prob_final.append(weight_list[i]/sum(weight_list))

        capcity_check = False
        try:
            for i in range(len(smell_keys)):
                draw = choice(smell_keys, 1, p=prob_final,replace= False)
                cap_chek_var = capacity_count + path_dic[draw[0]][5]
                if cap_chek_var <= cmax:
                    capcity_check = True
                if capcity_check == True:
                    break

            if capcity_check == True:
                x_current = path_dic[draw[0]][2]
                y_current = path_dic[draw[0]][3]
                path_cost.append(path_dic[draw[0]][0])
                path_out_cost.append(path_dic[draw[0]][1])
                path.append(draw[0])

                evac_score.append(path_dic[draw[0]][4])
                node_i_mem.append(draw[0])
                capacity_count += path_dic[draw[0]][5]

                O2 = (capacity_count/cmax)+1* anaerobic_rate
        except:
            pass
        t +=1

    if sum(evac_score) > max_evac_keep and (sum(path_cost)+path_out_cost[-1]) < max_distance and capacity_count >= cmax*capcity_rate:
        print(p)
        max_evac_keep = sum(evac_score)
        path_cost_keep = path_cost
        path_keep = path
        path_out_keep = path_out_cost
        capcity_keep = capacity_count



plt.plot(xc[0], yc[0], c='#043fa1', marker='X')
plt.plot(xc[1], yc[1], c='#28e739', marker='X')
plt.scatter(xc[2:], yc[2:], c='#e61207')

for i in range(len(weight)):
    plt.annotate(weight[i],[xc[i+2], yc[i+2]], xytext=(xc[i+2]+1.5, yc[i+2]+1.5))
    plt.annotate(capacity[i], [xc[i + 2], yc[i + 2]], xytext=(xc[i + 2] + 5, yc[i + 2] + 5),c='b')
plt.plot([xc[0],xc[path_keep[0]]],[yc[0],yc[path_keep[0]]],c='b',alpha=0.5)
for i in range(1,len(path_keep)):
    plt.plot([xc[path_keep[i-1]], xc[path_keep[i]]], [yc[path_keep[i-1]], yc[path_keep[i]]],c='b',alpha=0.5)
plt.plot([xc[path_keep[len(path_keep)-1]], xc[1]], [yc[path_keep[len(path_keep)-1]], yc[1]],c='b',alpha=0.5)
print(max_evac_keep)
print(capcity_keep)
print(sum(path_cost_keep)+path_out_keep[-1])
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()
