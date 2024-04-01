# -*- coding: utf-8 -*-
"""
@author: Román Zúñiga Macías
"""

from numpy import arange,array,dot,asarray,zeros,apply_along_axis,around,sort,shape,savetxt,array_equal,max,argmin,argmax,fill_diagonal,diag,argsort
from numpy.linalg import eig
from matplotlib.pyplot import plot,figure,title,legend,xlabel,ylabel,grid,axhline,axvline,savefig,imshow,show,scatter,hist,bar,subplots,Normalize,cm
from math import sqrt
from datetime import datetime,timedelta
from sklearn.cluster import AgglomerativeClustering
import matplotlib.patches as mpatches

#######################################################################################################
"""#Implemented functions to import and manage data"""
#Filtering data
def count_month(ind_month,mat_cases):
    n = len(ind_month)-1
    n1,m1 = shape(mat_cases)
    mat_month = zeros((n1,n))
    for i in range(0,n):
        sum_mun = zeros((n1,1))
        ini_month = ind_month[i]
        fin_month = ind_month[i+1]
        for j in range(ini_month,fin_month):
            sum_mun[:,0] = sum_mun[:,0]+mat_cases[:,j]
        mat_month[:,i] = sum_mun[:,0]
    return mat_month

#Convert rows of matrix into lists
def conv_to_list(matrix):
    n,m = shape(matrix)
    lists = []
    for i in range(0,n):
        row_act = matrix[i,:]
        lists.append(row_act.tolist())
    return lists

#n is total of municipalities or nodes
def fill_mat(lab,count,n):
    out = zeros((n))
    m = len(count)
    for i in range(0,m):
        out[lab[i]-1] = count[i]
    return out

#######################################################################################################
"""#Implemented functions to construct the algorithms"""

#Show results
#Print communities of a partition
def print_com(W,com,lab):
    n = len(com)
    for i in range(0,n):
        com_act = com[i]
        print("Community ",i)
        print(com_act)
        #Print the names of the municipalities of each community
        m = len(com_act)
        names_act = []
        for j in range(0,m):
            names_act.append(lab[com_act[j]])
        print(names_act)
    #Get partition modularity
    mod = modularity(W,com)
    print("This partition has modularity equal to: ",mod)
    return

def delta(i,j):
    if i==j:
        out = 1
    else:
        out = 0
    return out

def create_com(com_prev,u,threshold=0):
    #The community list corresponds to the vector u
    #All the positive values ​​of u will be assigned to one community and all the negative ones to another
    n = len(com_prev)
    s = zeros((n,1))
    com1,com2 = [],[]
    u11 = []
    u22 = []
    #The default threshold value for bipartitioning is zero
    for i in range(0,n):
        if u[i]>threshold:
            com1.append(com_prev[i])
            s[i,0] = 1
            u11.append(u[i])
        #The nodes corresponding to elements of u that are equal to zero are nodes with incident weights equal to zero, so they are discarded
        elif u[i]<threshold:
            com2.append(com_prev[i])
            s[i,0] = -1
            u22.append(u[i])
    return u11,u22,s,com1,com2

def bipartition(B,com):
    n_g = len(com)
    #Build the matrix corresponding to the community that will be split
    B_g = zeros((n_g,n_g))
    cf = 0
    for i in com:
        cc = 0
        for j in com:
            if i==j:
                sum_row = 0
                for k in com:
                    sum_row+=B[i,k]
            B_g[cf,cc] = B[i,j]-delta(i,j)*sum_row
            cc+=1
        cf+=1
    #Obtain the eigenvalue and the leading eigenvector
    eigenval,eigenvec = eig(B_g)
    eigenvec = eigenvec[:,argsort(eigenval)]
    eigenval = eigenval[argsort(eigenval)]
    beta1 = eigenval[-1]
    u1 = eigenvec[:,-1]
    u11,u22,s,com1,com2 = create_com(com,u1)
    #It remains to divide the increment by 4m
    inc_mod = dot(dot(s.T,B_g),s)
    return beta1,u11,u22,inc_mod,com1,com2

#A is the adjacency or weight matrix
def GeNA(A):
    n = len(A)
    #Sum the weights of edges that affect each of the nodes
    weight_inc = apply_along_axis(sum,1,A)
    weight_row = zeros((1,n))
    weight_row[0,:] = weight_inc
    m = apply_along_axis(sum,1,weight_row)/2
    mat_mod = zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            mat_mod[i,j] = A[i,j]-(weight_inc[i]*weight_inc[j])/(2*m)
    #When applying the eig function, the result is a list in which the first element is the array with the eigenvalues
    #The second element is the matrix whose columns are the normalized eigenvectors
    #Get the eigenvalue and the leading eigenvector
    eigenval,eigenvec = eig(mat_mod)
    eigenvec = eigenvec[:,argsort(eigenval)]
    eigenval = eigenval[argsort(eigenval)]
    beta1 = eigenval[-1]
    u1 = eigenvec[:,-1]
    #Lists to store the list of the final partition and the final leading eigenvectors
    com_fin,u_fin = [],[]
    #Create list of labels of the nodes that make up the graph
    com = list(range(0,n))
    #Determine if the obtained partition is trivial
    if beta1<=0.000001:
        com_fin.append(com)
        u_fin.append(u1)
    else:
        #Create communities according to the sign
        u11,u22,s,com1,com2 = create_com(com,u1)
        com_res,com_res_act = [com1,com2],[com1,com2]
        #List of eigenvectors u that contains the belonging level of each node to its respective community
        u_res,u_act = [u11,u22],[u11,u22]
        n_com = len(com_res)
        while(n_com!=0):
            for j in range(0,n_com):
                beta1,u11,u22,inc_mod,com11,com22 = bipartition(mat_mod,com_res[j])
                #Stop dividing the community when the leading eigenvalue is zero or very close to zero
                #A leading eigenvalue close to zero can produce trivial bipartitions
                if beta1<=0.000001:
                    com_fin.append(com_res[j])
                    u_fin.append(u_res[j])
                elif len(com11)<=1 or len(com22)<=1:
                    #In case a trivial partition has been leaked, the empty community will not be included
                    if len(com11)>0:
                        com_fin.append(com11)
                        u_fin.append(u11)
                    if len(com22)>0:
                        com_fin.append(com22)
                        u_fin.append(u22)
                else:
                    com_res_act.append(com11)
                    com_res_act.append(com22)
                    u_act.append(u11)
                    u_act.append(u22)
                com_res_act.remove(com_res[j])
                u_act.remove(u_res[j])
            com_res = com_res_act.copy()
            u_res = u_act.copy()
            n_com = len(com_res)
    #Calculate level of membership of nodes per each community through normalizing the values of u_fin
    level_memb = level_membership_GeNA(u_fin)
    return level_memb,com_fin

#Obtain belonging level of each node to its community in the original partition
def level_membership_GeNA(u):
    n = len(u)
    belong = []
    for i in range(0,n):
        belong.append([])
        sum_act=sum(u[i])
        for x in u[i]:
            belong_act = belong[i]
            belong_act.append(x/sum_act)
    return belong

###############################################################################################################
#Verify if a municipality x belongs to the community com
def comp(x,com):
    out = 0
    n = len(com)
    for i in range(0,n):
        if x==com[i]:
            out = 1
    return out

#Function to calculate the modularity of a partition
def modularity(W,partition):
    n = len(W)
    #Sum the weights of the edges that affect each of the nodes
    weight_inc = apply_along_axis(sum,1,W)
    weight_row = zeros((1,n))
    weight_row[0,:] = weight_inc
    L = apply_along_axis(sum,1,weight_row)/2
    #Modularity matrix
    B = zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            B[i,j] = W[i,j]-(weight_inc[i]*weight_inc[j])/(2*L[0])
    #Calculate the modularity perform the sum on the edges between nodes that belong to the same community
    m = len(partition)
    #Perform the sum in each community
    sum_com = zeros((1,m))
    #Generate arrays from which the necessary rows and columns will be removed
    mat_com = []
    for l in range(0,m):
        mat_com.append(B.copy())
    #Remove the values ​​of nodes that do not belong to the respective community
    for k in range(0,m):
        com_act = partition[k]
        mat_act = mat_com[k]
        for p in range(0,n):
            if comp(p,com_act)==0:
                mat_act[p,:] = zeros((n))
                mat_act[:,p] = zeros((n))
        row = zeros((1,n))
        row[0,:] = apply_along_axis(sum,1,mat_act)
        sum_com[0,k] = apply_along_axis(sum,1,row)
    suma = apply_along_axis(sum,1,sum_com)
    #Modularity value
    mod = suma[0]/(2*L[0])
    return mod

#######################################################################################################
#Functions to construct the algorithm FUSE
#Obtain reduced graph where the communities are the nodes and the weights are the sums of inter-community modularity
#The modularity matrix used will be that of the original graph
def graph_com_sum(com,W):
    n,m = shape(W)
    M_mod = mat_mod(W)
    l = len(com)
    graph_com = zeros((l,l))
    for k in range(0,l):
        com_k = com[k]
        for p in range(k+1,l):
            com_p = com[p]
            #Construct index vectors of communities k y p
            x_k = vec_ind(n,com_k)
            x_p = vec_ind(n,com_p)
            #Perform the sum over all those edges between nodes of the communities k and p
            #But make this sum only if the respective communities have inter-community links
            if dot(dot(x_k.T,W),x_p)!=0:
                graph_com[k,p] = dot(dot(x_k.T,M_mod),x_p)
    graph_comp = graph_com+graph_com.T
    return graph_comp

#Construct index vector of a community
def vec_ind(n,com):
    vec = zeros((n,1))
    for x in com:
        vec[x,0] = 1
    return vec

#Algorithm that reduces the number of communities in a partition down to a wanted number q
'''
Inputs. List of lists level_memb_ini: level of membership of the nodes per community from the initial partition,
        List of lists partition: initial partition that will be the base to decrease the number of communities,
        array W: weight matrix of the graph,
        int q: wanted number of communities.
Outputs. List of lists part_out: final partition,
        List of arrays W_com_mod: list of matrix of modularity inter-community obtained during the reduction,
        list of lists level_memb_out: level of membership of the nodes per community from the final partition.
'''
def FUSE(level_memb_ini,partition,W,q):
    n = len(partition)
    num_rem = n-q
    part_out = partition.copy()
    level_memb_out = level_memb_ini.copy()
    #Store the matrices with the sums of modularity of the connections between communities of each step
    W_com_mod = []
    for i in range(0,num_rem):
        W_graph = graph_com_sum(part_out,W)
        mat_max,pos = max_mat(W_graph)
        merge_com = part_out[pos[0]]+part_out[pos[1]]
        #Delete communities to join
        part_out.pop(pos[0])
        part_out.pop(pos[1]-1)
        #Delete the corresponding eigenvectors
        level_memb_out.pop(pos[0])
        level_memb_out.pop(pos[1]-1)
        #Sort the items in the united community in ascending order
        merge_com.sort()
        #AAdd the community that is the union of the two removed
        part_out.append(merge_com)
        #Add the level of membership of the joined communities
        level_new = level_membership_FUSE(W,merge_com)
        level_memb_out.append(level_new)
        #Add the matrix of sum modularity of the connections between communities
        W_com_mod.append(W_graph)
    #Add the sum of modularity matrix corresponding to the final partition
    W_graph_fin = graph_com_sum(part_out,W)
    W_com_mod.append(W_graph_fin)
    return part_out,W_com_mod,level_memb_out

#Obtain the belonging level of the nodes that are part of a community that was the result of FUSE
def level_membership_FUSE(W,com):
    W_mod = mat_mod(W)
    #u11 has the positive elements of the eigenvector u, while u22 has the negative values
    beta1,u11,u22,inc_mod,conj1,conj2 = bipartition(W_mod,com)
    n1,n2 = len(u11),len(u22)
    #Total number of nodes
    n = n1+n2
    #Vector that will contain the entries of both vectors, ordered according to the node to which they correspond
    u = zeros((n1+n2))
    #Transform u_11 and u_22 to arrays
    u11_ar,u22_ar = asarray(u11),asarray(u22)
    #The sum of u11_ar and u22_ar is the same, so only the sum of u11_ar will be calculated, which is the vector of positive elements
    u_sum = 2*u11_ar.sum()
    if n1!=0 and n2!=0:
        u[0:n1] = (n1/n)*2*u11_ar/u_sum
        u[n1:n] = (n2/n)*(-2*u22_ar/u_sum)
    else:
        u[0:n1] = u11_ar
        u[n1:n] = -u22_ar
    com_out = conj1+conj2
    #Order the elements of the vector u according to the node number, contained in com_out
    u_com = []
    for i in range(0,n):
        u_com.append((u[i],com_out[i]))
    u_com.sort(key = lambda valor:valor[1])
    u_ar = asarray(u_com)
    u_out = u_ar[:,0]
    return u_out

#Get the maximum element of an array, as well as its position
def max_mat(mat):
    M = mat.copy()
    n,m = shape(M)
    #Create list of elements above the main diagonal of the matrix
    l = int(n*(n-1)/2)
    arr_mat = zeros((l))
    #Fill the array with the elements of the array
    count = 0
    for i in range(0,n-1):
        arr_mat[count:count+n-i-1] = M[i,i+1:n]
        count+=n-i-1
    list_mat = arr_mat.tolist()
    #Sort the value list
    list_mat.sort()
    #Reverse the order of the list
    list_mat = list_mat[::-1]
    #Maximum non-zero list element
    max_val = list_mat[0]
    for k in range(1,l):
        if max_val!=0:
            break
        else:
            max_val = list_mat[k]
    pos = [0,0]
    #Traverse the upper triangular part of the array to detect the position of the maximum element
    for i in range(0,n-1):
        for j in range(i+1,m):
            if max_val==mat[i,j]:
                pos = [i,j]
                break
    return max_val,pos

#######################################################################################################
#Functions to construct the algorithm ADD
#Construct modularity matrix
#A is adjacency or weight matrix of the graph
def mat_mod(A):
    n = len(A)
    #Sum the weights of the edges that affect each of the nodes
    weight_inc = apply_along_axis(sum,1,A)
    weight_row = zeros((1,n))
    weight_row[0,:] = weight_inc
    #Sum of all the weights of the graph
    m = apply_along_axis(sum,1,weight_row)/2
    B = zeros((n,n))
    for i in range(0,n):
        for j in range(0,n):
            B[i,j] = A[i,j]-(weight_inc[i]*weight_inc[j])/(2*m)
    return B

#Construct dissimilarity matrix from the modularity matrix and normalizing with K
def mat_dis_norm_max(W,func_dist):
    n = len(W)
    #Calculate the total sum of weights of the edges
    weight_inc = apply_along_axis(sum,1,W)
    #Get the maximum weighted degree
    K = max(weight_inc)
    #Calculate the total sum of the weights
    weight_row = zeros((1,n))
    weight_row[0,:] = weight_inc
    L = apply_along_axis(sum,1,weight_row)/2
    #Obtain maximum weight
    W_max = max(W)
    #Define the sought bound based on K and W_max
    bounds = [W_max,K**2/(2*L[0])]
    bounds.sort()
    upper_bound = bounds[1]
    #Calculate modularty matrix
    B = mat_mod(W)
    M_D = zeros((n,n))
    #The entries of diagonal will be remain as zeros to assign zero dissimilarity of a node with itself
    for i in range(0,n):
        for j in range(i+1,n):
            M_D[i,j] = func_dist(B[i,j]/upper_bound)
    M_out = M_D+M_D.T
    return M_out

def dis_fact_half(B_ij):
      B_out = sqrt((1-B_ij)/2)
      return B_out

#Separate the indices of the two subcommunities to generate the output of the algorithm
def sep_labels(clas_lab,lab):
    sub_com1,sub_com2 = [],[]
    n = len(clas_lab)
    for i in range(0,n):
        if clas_lab[i]==0:
            sub_com1.append(lab[i])
        else:
            sub_com2.append(lab[i])
    return sub_com1,sub_com2

#A is the adjacency or weight matrix
#Function to split a community in two through hierarchical clustering with modularity-based dissimilarity
def div_com(W,com_min,lab,func_dis,mat_dis):
    D = mat_dis(W,func_dis)
    n1 = len(com_min)
    D_com = zeros((n1,n1))
    for i in range(0,n1):
        for j in range(0,n1):
            D_com[i,j] = D[com_min[i],com_min[j]]
    #Try different linkage functions
    cluster = AgglomerativeClustering(n_clusters=2,metric='precomputed',linkage='average')
    cluster.fit_predict(D_com)
    classif = cluster.labels_
    sub_com1,sub_com2 = sep_labels(classif,lab)
    return sub_com1,sub_com2

#This algorithm is to perform the internal sums of modularity and choose the community that presents the minimum
def com_sum_int(com,B):
    n = len(B)
    l = len(com)
    sum_com = []
    ind_com = []
    #Make the diagonal of B all zeros
    fill_diagonal(B,0)
    for k in range(0,l):
        com_k = com[k]
        x_k = vec_ind(n,com_k)
        #Perform the sum over all pairs of nodes in community k
        sum_k = dot(dot(x_k.T,B),x_k)
        #Current community size
        len_k = len(com_k)
        #Only take into account those communities composed of more than one node, otherwise they are already indivisible
        if len_k>1:
          #Calculate average per possible link inside the community
          sum_com.append(sum_k/(len_k**2-len_k))
          ind_com.append(k)
    #It indicates the position of the community with the minimum average of those that are not yet indivisible
    com_min = argmin(sum_com)
    return ind_com[com_min]

#Algorithm that increases the number of communities in a partition up to a wanted number n_p
'''
Inputs. List of lists partition: initial partition that will be the base to increase the number of communities,
        int n_p: wanted number of communities,
        array W: weight matrix of the graph.
Outputs. List of lists part_out: final partition.
'''
def ADD(partition,n_p,W,func_dis=dis_fact_half,mat_dis=mat_dis_norm_max):
    n = len(partition)
    B = mat_mod(W)
    part_out = partition.copy()
    for i in range(0,n_p-n):
        #Community with lowest sum of internal modularity of the current partition
        ind_min = com_sum_int(part_out,B)
        com_min = part_out[ind_min]
        part_out.pop(ind_min)
        #Split the com_min community in two
        sub1,sub2 = div_com(W,com_min,com_min,func_dis,mat_dis)
        #Add obtained sub-communities
        part_out.append(sub1)
        part_out.append(sub2)
    return part_out

#######################################################################################################
#ReVAM algorithm to generate a partition in k communities
'''
Inputs. Array W: Weight matrix,
        int k: number of communities searched k.
Output. Partition in k communities.
'''
def ReVAM(W,k):
    #Find first partition through Generalization of Newman's Algorithm
    coef_level_ini,part_ini = GeNA(W)
    #Calculate number of communities obtained so far
    num_com_GeNA = len(part_ini)
    #Decide which algorithm is necessary to apply, or the initial partition is the searched
    #In this case, apply ADD to increase the number of communities
    if k>num_com_GeNA:
        part_ADD = ADD(part_ini,k,W)
        return part_ADD
    #In this case, apply FUSE to decrease the number of communities
    elif k<num_com_GeNA:
        part_FUSE,W_FUSE,u_FUSE = FUSE(coef_level_ini,part_ini,W,k)
        return part_FUSE
    else:
        return part_ini
#######################################################################################################

"""#Implemented functions to manage maps"""

def transf(com):
    #Labels of the municipalities that the INEGI shp files manage
    new_lab = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,26,15,16,17,18,19,20,21,27,28,29,30,
               31,32,22,23,24,25,33,34,35,36,37,38,39,40,41,42,43,44,45]
    n_com = len(com)
    new_com = []
    for i in range(0,n_com):
        new_com.append([])
        com_act = com[i]
        n_act = len(com_act)
        new_com_act = new_com[i]
        for j in range(0,n_act):
            new_com_act.append(new_lab[com_act[j]])
    return new_com

#Change the lists of communities for a single list with the group number of each municipality
def com_to_group(com):
    n_com = len(com)
    groups = zeros((46))
    for i in range(0,n_com):
        com_act = com[i]
        n_act=len(com[i])
        for j in range(0,n_act):
            groups[com_act[j]] = i
    groups_out = groups.tolist()
    return groups_out

#Change group labels to corresponding color names
def change(groups,colors):
    n = len(groups)
    colors_out = []
    for i in range(0,n):
        colors_out.append(colors[int(groups[i])])
    return colors_out

def generate_patches(colors):
    list_patches = []
    n_colors = len(colors)
    for i in range(0,n_colors):
        patch_act = mpatches.Patch(color = colors[i], label = str(i))
        list_patches.append(patch_act)
    return list_patches

def transf_cases_gto(com_lab):
    #Labels of the municipalities that the INEGI shp files manage
    new_lab = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,26,15,16,17,18,19,20,21,27,28,29,30,
               31,32,22,23,24,25,33,34,35,36,37,38,39,40,41,42,43,44,45]
    n_mun = len(new_lab)
    new_com = zeros((n_mun))
    for i in range(0,n_mun):
           new_com[new_lab[i]] = com_lab[i]
    return new_com

def col_map_level(level,com):
    mun_level = zeros((46))
    n_com = len(com)
    for i in range(0,n_com):
        mun_level[com[i]] = level[i]
    return mun_level

def change_label_ord(group,list_com_sort):
    #Amount of communities
    m = len(list_com_sort)
    #Amount of municipalities
    n = len(group)
    new_group = zeros((n))
    for i in range(0,n):
        #Community that has to be seek in the list
        com_list = int(group[i])
        #Community position in the list
        ind_new = list_com_sort.index(com_list)
        new_group[i] = m-ind_new-1
    return new_group