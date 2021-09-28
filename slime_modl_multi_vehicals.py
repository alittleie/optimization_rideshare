import numpy as np
import matplotlib.pyplot as plt
import math
from numpy.random import choice
import time
start_time = time.time()

rnd = np.random
rnd.seed(1)


number_of_start_vehicles = 4
number_of_exit_nodes = 1
n= 18


start_positon = number_of_start_vehicles+number_of_exit_nodes


xc = []
yc = []
for i in range(0,n+1):
    xc.append((rnd.random())*200)
    yc.append((rnd.random())*100)



weight = []
capacity = []
max_node_cap = 5
start_nodes_list = []



for i in range(0,start_positon - number_of_exit_nodes):
    start_nodes_list.append(i)


for i in range(start_positon,n+1):
    weight.append(rnd.randint(1,11))
    capacity.append(rnd.randint(1,max_node_cap))



max_growth = 25
min_growth = 10
growth_rate =20
max_distance = 250
capcity_rate = .7
anaerobic_rate = 10
cmax = 10
path_cost_keep = []
path_keep = []
max_evac_keep = 0
capcity_keep = 0
r = rnd.randint(min_growth,max_growth)


for p in range(0,100):

    x_current_all_vehicals = []
    y_current_all_vehicals = []
    capcity_All = {}
    for i in range(0, number_of_start_vehicles):
        capcity_All[i] = 0
    node_i_mem = []
    distances = []
    path_cost = []
    path_all_vehicals = []
    path = []
    evac_score = []


    working_node_x = {}
    working_node_y = {}
    info_dic = {}
    for i in range(0, start_positon - number_of_exit_nodes):
        info_dic[i] = {}
    capacity_count = 0

    O2 = (capacity_count/cmax)+1* anaerobic_rate
    t=0
    for i in range(0,10):
        short_term_mem = []
        node_short_term_mem = []
        if i == 0:
            for j in range(0, start_positon - number_of_exit_nodes):
                x_current_all_vehicals.append(xc[j])
                working_node_x[j] = xc[j]
                working_node_y[j] = yc[j]
                y_current_all_vehicals.append(yc[j])
        elif i == 1:
            for j in range(0, start_positon - number_of_exit_nodes):
                try:
                    x_current_all_vehicals.append(info_dic[j][i-1][1])
                    working_node_x[j] = x_current_all_vehicals.append(info_dic[j][i-1][1])
                    y_current_all_vehicals.append(info_dic[j][i-1][2])
                    working_node_y[j] = x_current_all_vehicals.append(info_dic[j][i - 1][2])
                except:
                    x_current_all_vehicals.append(xc[j])
                    y_current_all_vehicals.append(yc[j])
        else:
            for j in range(0, start_positon - number_of_exit_nodes):
                try:
                    x_current_all_vehicals.append(info_dic[j][i - 1][1])
                    working_node_x[j] = x_current_all_vehicals.append(info_dic[j][i - 1][1])
                    y_current_all_vehicals.append(info_dic[j][i - 1][2])
                    working_node_y[j] = x_current_all_vehicals.append(info_dic[j][i - 1][2])

                except:

                    x_current_all_vehicals.append(working_node_x[j])
                    y_current_all_vehicals.append(working_node_y[j])


        color = ['#ff78ef', '#466f80', '#2b16df', '#df9c16']
        marker = ['v', '^', '<', 'x']
        path_List_all_vehicals = []
        node_keep = []
        node_keep_all_vehicals = []

        for k in range(0,start_positon- number_of_exit_nodes):

            node_keep = []
            path_dic = {}
            certain_spot_bool = False
            bool_hold = False
            multi=1
            count = 0
            s = 0

            while (certain_spot_bool == False) and count < 50:
                s = r * multi

                for c in range(start_positon,n+1):
                    d = (math.sqrt(((xc[c]-x_current_all_vehicals[k])**2)+((yc[c]-y_current_all_vehicals[k])**2)))
                    d_out = (math.sqrt(((xc[c]-xc[start_positon - number_of_exit_nodes])**2)+((yc[c]-yc[start_positon - number_of_exit_nodes])**2)))

                    if d <= s and c not in node_i_mem and c not in short_term_mem  :

                        bool_hold = True
                        node_keep.append(c)
                        short_term_mem.append(c)
                        path_dic[c] = [d,d_out,xc[c],yc[c],weight[c-start_positon - number_of_exit_nodes],capacity[c-start_positon - number_of_exit_nodes]]
                if bool_hold == True:
                    certain_spot_bool = True
                else:
                    multi = multi*2
                count += 1
            path_List_all_vehicals.append(path_dic)
            node_keep_all_vehicals.append(node_keep)
            print(node_keep_all_vehicals)


        r = r + growth_rate
        distance_list = []
        Smell_list = []
        Capa_list = []
        d_out_hold_list = []
        d_out_hold = {}
        for k in range(0, start_positon - number_of_exit_nodes):
            distance = {}
            Smell = {}
            Capa = {}
            for v in range(len(node_keep_all_vehicals[k])):
                distance[node_keep_all_vehicals[k][v]]= path_List_all_vehicals[k][node_keep_all_vehicals[k][v]][0]
                Smell[node_keep_all_vehicals[k][v]]= path_List_all_vehicals[k][node_keep_all_vehicals[k][v]][4]
                Capa[node_keep_all_vehicals[k][v]]= path_List_all_vehicals[k][node_keep_all_vehicals[k][v]][5]
                d_out_hold[node_keep_all_vehicals[k][v]]= path_List_all_vehicals[k][node_keep_all_vehicals[k][v]][1]
            distance_list.append(distance)
            Smell_list.append(Smell)
            Capa_list.append(Capa)
            d_out_hold_list.append(d_out_hold)
        node_mem_value = [10000]
        for k in range(0, start_positon - number_of_exit_nodes):
            node_value = 10000
            while node_value in node_mem_value:
                node_value = choice(start_nodes_list, 1, replace=False)[0]
            node_mem_value.append(node_value)


            distance_keys = list(distance_list[node_value].keys())
            distance_values = list(distance_list[node_value].values())
            smell_keys = list(Smell_list[node_value].keys())
            smell_values = list(Smell_list[node_value].values())
            d_out_hold_keys = list(d_out_hold_list[node_value].keys())
            d_out_hold_values = list(d_out_hold_list[node_value].values())

            distance_probs = []
            smell_probs = []
            d_out_probs = []
            for v in range(len(distance_keys)):
                distance_probs.append(1/(distance_values[v]/sum(distance_values)))
                d_out_probs.append(1/(d_out_hold_values[v]/sum(d_out_hold_values)))
                smell_probs.append(smell_values[v]/sum(smell_values))
            d_probs_new =[]
            for v in range(len(d_out_probs)):
                d_probs_new.append(d_out_probs[v]*O2)
            weight_list = []
            for v in range(len(d_probs_new)):
                weight_list.append((distance_probs[v]/sum(distance_probs)) * (d_probs_new[v]/sum(d_probs_new)) * (smell_probs[v]/sum(smell_probs)))
            prob_final = []
            for v in range(len(weight_list)):
                prob_final.append(weight_list[v]/sum(weight_list))

            capcity_check = False

            #try:
            for l in range(len(smell_keys)):
                draw = choice(smell_keys, 1, p=prob_final,replace= False)
                cap_chek_var = capcity_All[node_value] + path_List_all_vehicals[node_value][draw[0]][5]
                if cap_chek_var <= cmax:
                    capcity_check = True
                if capcity_check == True:
                    break

            if capcity_check == True:
                capcity_All[node_value] += path_List_all_vehicals[node_value][draw[0]][5]

                info_dic[node_value][i] = [path_List_all_vehicals[node_value][draw[0]][5],path_List_all_vehicals[node_value][draw[0]][2],path_List_all_vehicals[node_value][draw[0]][3],
                                           path_List_all_vehicals[node_value][draw[0]][0],draw[0],path_List_all_vehicals[node_value][draw[0]][4],path_List_all_vehicals[node_value][draw[0]][1] ]

                node_i_mem.append(draw[0])


                O2 = (capacity_count/cmax)+1* anaerobic_rate
            #except:
                #pass
            t +=1
        if all(value >= cmax*capcity_rate for value in capcity_All.values()):
            break

    for k in range(0, start_positon - number_of_exit_nodes):
        distance_hold = 0
        evac_hold = 0
        node_hold = []
        for key in info_dic[k]:
            node_hold.append(info_dic[k][key][1])
            node_hold.append(info_dic[k][key][2])
            evac_hold += info_dic[k][key][4]
            distance_hold += info_dic[k][key][3]

        try:
            distance_hold += list(info_dic[k].values())[-1][6]
        except:
            distance_hold = 0
        distances.append(distance_hold)
        evac_score.append(evac_hold)
        path.append(node_hold)


    if all(value <= max_distance for value in distances) and sum (evac_score) > max_evac_keep:

        max_evac_keep = sum(evac_score)
        path_cost_keep = distances
        path_keep = path
        capcity_keep = capcity_All


print(max_evac_keep)
print(path_cost_keep)
print(path_keep)
print(capcity_keep)
print("--- %s seconds ---" % (time.time() - start_time))





for l in range(0, start_positon - 1):
    plt.plot(xc[l], yc[l], c=color[l], marker='X')
for l in range(start_positon - 1, start_positon):
    plt.plot(xc[l], yc[l], c='#28e739', marker='X')
plt.scatter(xc[start_positon:], yc[start_positon:], c='#e61207')


for i in range(len(weight)):
    plt.annotate(weight[i],[xc[i+5], yc[i+5]], xytext=(xc[i+5]+1.5, yc[i+5]+1.5))
    plt.annotate(capacity[i], [xc[i + 5], yc[i + 2]], xytext=(xc[i + 5] + 5, yc[i + 5] + 5),c='b')





for k in range(0, start_positon - number_of_exit_nodes):
   try:
    for i in range(0,len(path_keep[k]),2) :
        if i == 0:
            plt.plot([xc[k],path_keep[k][i] ], [yc[k], path_keep[k][i+1]], c=color[k], alpha=0.5)
        else:
            plt.plot([path_keep[k][i-2], path_keep[k][i]], [path_keep[k][i-1], path_keep[k][i+1]], c=color[k], alpha=0.5)
    plt.plot([path_keep[k][len(path_keep[k])-2], xc[4]], [path_keep[k][len(path_keep[k])-1], yc[4]], c=color[k], alpha=0.5)
   except:
       pass
# plt.plot([xc[0],xc[path_keep[0]]],[yc[0],yc[path_keep[0]]],c='b',alpha=0.5)
# for i in range(1,len(path_keep)):
#     plt.plot([xc[path_keep[i-1]], xc[path_keep[i]]], [yc[path_keep[i-1]], yc[path_keep[i]]],c='b',alpha=0.5)
# plt.plot([xc[path_keep[len(path_keep)-1]], xc[1]], [yc[path_keep[len(path_keep)-1]], yc[1]],c='b',alpha=0.5)
plt.show()

