# -*- coding: utf-8 -*-
"""
@author: Román Zúñiga Macías
"""

from numpy import arange,array,dot,asarray,zeros,apply_along_axis,around,sort,shape,savetxt,array_equal,max,argmin,argmax,fill_diagonal,diag,argsort
from numpy.linalg import eig
from matplotlib.pyplot import plot,figure,title,legend,xlabel,ylabel,grid,axhline,axvline,savefig,imshow,show,scatter,hist,bar,subplots,Normalize,cm,xticks,yticks
from math import sqrt
import pandas as pd
from datetime import datetime,timedelta
from sklearn.cluster import AgglomerativeClustering
import matplotlib.patches as mpatches
import geopandas
pd.options.display.max_rows = 10
from ProposedAlgorithms import count_month,WeGA,FUSE,level_membership_WeGA,level_membership_FUSE,print_com,transf,com_to_group,change,generate_patches,quality,col_map_level,transf_cases_gto,ADD,dis_fact_half,mat_dis_norm_max

"""#Import the data to create maps"""
mun_gto_map = geopandas.read_file('mgm_gto2020.shp')
mun_gto_map.head()

"""#Import the epidemiological data"""

df = pd.read_csv('Casos_Municipio_20220515.csv')
#Delete records with null value
df = df.dropna()
print("Database size:")
print(df.shape)
print("Data Overview:")
print(df.info())
#How many different values ​​each variable has
print("Number of different values ​​for each variable:")
print(df.nunique())
#Check for null values
print("Null values ​​in each variable")
print(df.isnull().sum())
#Apply filter to only obtain municipalities of Guanajuato
#Guanajuato is the 11th state
df1 = df[df['cve_ent']>=11000]
df2 = df1[df1['cve_ent']<12000]
df_ord = df2.sort_values('cve_ent')
#Save labels of the names of municipalities
name_mun = df_ord["nombre"].tolist()
#Delete columns that do not correspond to daily cases registered
df_ord.drop(['cve_ent','poblacion','nombre'], axis = 1, inplace = True)
#Pass dataframe values ​​to an array
data = df_ord.values
#The first record corresponds to February 26, 2020
#The last record corresponds to May 14, 2022
n,m = shape(data)
#Define list with indices that are beginning of each month
month_name = ['mar20','apr20','may20','jun20','jul20','aug20','sep20','oct20','nov20','dec20','jan21','feb21','mar21','apr21','may21',
          'jun21','jul21','aug21','sep21','oct21','nov21','dec21','jan22','feb22','mar22','apr22','may22']
month_ini = [4,35,65,97,127,158,189,219,250,280,311,342,370,401,431,462,492,523,554,584,615,645,676,707,735,766,796]

#Import the names of the municipalities in the format necessary for the map
df_etiq = pd.read_excel("municipios_gto.xlsx", sheet_name = "Hoja1")
#Import population of each municipality of Guanajuato
population = df_etiq["Poblacion"]

"""#Applying Generalized Algorithm to Weighted Algorithms

Generate weight matrix for the period from June 2021 to February 2022
"""

#Import Adjacency Matrix from Excel
df1 = pd.read_excel("mat_adj_gto.xlsx", sheet_name = "Sheet1")
mat_adj = df1.values

#Cases counted for each registered month
mat_count_month = count_month(month_ini,data)

#Determination of the weight matrix from June 2021 to February 2022
cases_per = zeros((46,1))
#There are 9 months that will need to be added for the considered period
for k in range(0,9):
  month_act = mat_count_month[:,k+15]
  cases_per[:,0] = cases_per[:,0]+month_act

#Export the number of average cases for each municipality from June 2021 to February 2022
etiq_map = df_etiq["Etiquetas_mapa"]
#Export municipalities community list along with average cases
cases_exp = pd.ExcelWriter('Gto_jun_feb_prom.xlsx')
df=pd.DataFrame(cases_per/9,columns = ["Casos_promedio"])
df['Municipio'] = etiq_map
df.to_excel(cases_exp,sheet_name = 'Casos_por_municipio', index=True)
cases_exp.save()
cases_exp.close()

#Construct the weight matrix using the adjacency matrix
W_jun_feb = zeros((46,46))
for i in range(0,46):
    for j in range(i+1,46):
        if mat_adj[i,j+1]==1:
            W_jun_feb[i,j] = (cases_per[i]+cases_per[j])/18
            W_jun_feb[j,i] = (cases_per[i]+cases_per[j])/18

#Applying WeGA
print("Communities from the period June 2021-February 2022:")
u_per,com_per = WeGA(W_jun_feb)
#Number of communities detected
n_com = len(com_per)
#Level of membership of each node to its respective community
belong_fact = level_membership_WeGA(u_per)
print_com(W_jun_feb,com_per,name_mun)
print()
print("Level of membership to each community:")
print(belong_fact)

"""Color map with geopandas"""

com_per_new = transf(com_per)
groups_per = com_to_group(com_per_new)
#List of 6 colors to use for communities
colors  =['red','green','hotpink','blue','purple','yellow']
colors_per = change(groups_per,colors)
#Color map
fig,ax = subplots(figsize = (25,16))
#Remove the axis
ax.axis('off')
#Assign community labels to colors
color_patches = generate_patches(colors)
ax.legend(handles = color_patches, prop = {'size':14}, loc = 4)
mun_gto_map.plot(color = colors_per, ax = ax)

"""Export image with a resolution of 600 dpi"""

fig.savefig("com_ini.png", dpi = 600)

"""#Find the reduction in number of communities that maximizes modularity"""

#List to store modularity of the partitions with less number of communities
red_mod = []
for i in range(n_com-1,1,-1):
    com_i_per,mat_per_i,u_i_per = FUSE(u_per,com_per,W_jun_feb,i)
    n_com_i = len(com_i_per)
    B,mod = quality(W_jun_feb,com_i_per)
    print(mod)
    red_mod.append(mod)
fig = figure(figsize = (10,8))
t = arange(n_com-1,1,-1)
print(t)
print(red_mod)
plot(t,red_mod, linestyle = "--", marker="o", markersize = 9, linewidth = 1.0, color = 'blue')
ylabel("Modularity", fontsize = 16)
xlabel("Number of communities", fontsize = 16)
axhline(y = 0.46676348, color = 'red', linestyle = "--", linewidth = 1.5)
xticks(fontsize = 15)
yticks(fontsize = 15)
#title("Quality of partitions decreasing the number of communities", fontsize = 14)
grid()

"""Export image with a resolution of 600 dpi"""

fig.savefig("graphic_dec_com.png", dpi = 600)

"""#Reduce number of communities to 4"""

#Partition using FUSE, that considers sums of modularity between pairs of communities
com4_jun_feb,W_mod_jun_feb,u4_jun_feb = FUSE(u_per,com_per,W_jun_feb,4)
print("Division into 4 communities for the period June 2021-February 2022:")
print_com(W_jun_feb,com4_jun_feb,name_mun)
#Level of membership of each node to its respective community
belong4_jun_feb = level_membership_WeGA(u4_jun_feb)
print()

"""Color map using geopandas"""

com4_new = transf(com4_jun_feb)
groups4 = com_to_group(com4_new)
#List of 4 colors to use for communities
colors = ['red','green','hotpink','blue']
colors4 = change(groups4,colors)
#Color mapa
fig,ax = subplots(figsize = (18,12))
#Remove the axis
ax.axis('off')
#Assign community labels to colors
color_patches = generate_patches(colors)
ax.legend(handles = color_patches, prop = {'size':14}, loc = 7)
mun_gto_map.plot(color = colors4, ax = ax)

"""Export image with a resolution of 600 dpi"""

fig.savefig("part_4_com.png", dpi = 600)

"""#Level of membership to the communities

Color level of membership using geopandas
"""
#Nodes of Community 0
niv0_4 = col_map_level(belong4_jun_feb[0],com4_jun_feb[0])
niv0_mod = transf_cases_gto(niv0_4)
mun_gto_map['Niv0_4'] = niv0_mod
#Set the range for the choropleth
titulo = '(a) Community 0'
col = 'Niv0_4'
vmin = mun_gto_map[col].min()
vmax = mun_gto_map[col].max()
cmap = 'Reds'
#Create figure and axes for Matplotlib
fig,axs = subplots(2, 2, figsize = (15,10))
ax0 = axs[0,0]
#Remove the axis
ax0.axis('off')
mun_gto_map.plot(column = col, ax = ax0, edgecolor = '0.1', linewidth = 1, cmap = cmap)
#Add a title
ax0.set_title(titulo, fontdict = {'fontsize': '15', 'fontweight': '3'})
#Create colorbar as a legend
sm=cm.ScalarMappable(norm = Normalize(vmin = vmin, vmax = vmax), cmap = cmap)
#Empty array for the data range
sm._A = []
#Add the colorbar to the figure
cbar = fig.colorbar(sm, ax = ax0)
cbar.ax.tick_params(labelsize = 13)

#Nodes of Community 1
niv1_4 = col_map_level(belong4_jun_feb[1],com4_jun_feb[1])
niv1_mod = transf_cases_gto(niv1_4)
mun_gto_map['Niv1_4'] = niv1_mod
#Set the range for the choropleth
titulo = '(b) Community 1'
col = 'Niv1_4'
vmin = mun_gto_map[col].min()
vmax = mun_gto_map[col].max()
cmap = 'Greens'
#Create figure and axes for Matplotlib
ax1=axs[0,1]
#Remove the axis
ax1.axis('off')
mun_gto_map.plot(column = col, ax = ax1, edgecolor = '0.1', linewidth = 1, cmap = cmap)
#Add a title
ax1.set_title(titulo, fontdict = {'fontsize': '15', 'fontweight': '3'})
#Create colorbar as a legend
sm = cm.ScalarMappable(norm = Normalize(vmin = vmin, vmax = vmax), cmap = cmap)
#Empty array for the data range
sm._A = []
#Add the colorbar to the figure
cbar = fig.colorbar(sm, ax = ax1)
cbar.ax.tick_params(labelsize = 13)

#Nodes of Community 2
niv2_4 = col_map_level(belong4_jun_feb[2],com4_jun_feb[2])
niv2_mod = transf_cases_gto(niv2_4)
mun_gto_map['Niv2_4'] = niv2_mod
#Set the range for the choropleth
titulo='(c) Community 2'
col = 'Niv2_4'
vmin = mun_gto_map[col].min()
vmax = mun_gto_map[col].max()
cmap = 'PuRd'
#Create figure and axes for Matplotlib
ax2 = axs[1,0]
#Remove the axis
ax2.axis('off')
mun_gto_map.plot(column = col, ax = ax2, edgecolor = '0.1', linewidth = 1, cmap = cmap)
#Add a title
ax2.set_title(titulo, fontdict = {'fontsize': '15', 'fontweight': '3'})
#Create colorbar as a legend
sm = cm.ScalarMappable(norm = Normalize(vmin = vmin, vmax = vmax), cmap = cmap)
#Empty array for the data range
sm._A = []
#Add the colorbar to the figure
cbar = fig.colorbar(sm, ax = ax2)
cbar.ax.tick_params(labelsize = 13)

#Nodes of Community 3g
niv3_4 = col_map_level(belong4_jun_feb[3],com4_jun_feb[3])
niv3_mod = transf_cases_gto(niv3_4)
mun_gto_map['Niv3_4'] = niv3_mod
#Set the range for the choropleth
titulo = '(d) Community 3'
col = 'Niv3_4'
vmin = mun_gto_map[col].min()
vmax = mun_gto_map[col].max()
cmap = 'Blues'
#Create figure and axes for Matplotlib
ax3 = axs[1,1]
#Remove the axis
ax3.axis('off')
mun_gto_map.plot(column = col, ax = ax3, edgecolor = '0.1', linewidth = 1, cmap = cmap)
#Add a title
ax3.set_title(titulo, fontdict = {'fontsize': '15', 'fontweight': '3'})
#Create colorbar as a legend
sm = cm.ScalarMappable(norm = Normalize(vmin = vmin, vmax = vmax), cmap = cmap)
#Empty array for the data range
sm._A = []
#Add the colorbar to the figure
cbar = fig.colorbar(sm, ax = ax3)
cbar.ax.tick_params(labelsize = 13)

"""Export image with a resolution of 600 dpi"""

fig.savefig("level_membership_FUSE.png", dpi = 600)

"""#Find the increase in number of communities that maximizes modularity"""

#List to store modularity of the partitions with less number of communities
aum_mod = []
for i in range(n_com+1,n_com+11):
    com_i_per = ADD(com_per,i,W_jun_feb,dis_fact_half,mat_dis_norm_max)
    n_com_i = len(com_i_per)
    B,mod = quality(W_jun_feb,com_i_per)
    print(mod)
    aum_mod.append(mod)
fig = figure(figsize = (10,8))
t=arange(n_com+1,n_com+11)
print(t)
print(aum_mod)
plot(t, aum_mod, linestyle = "--", marker = "o", markersize = 9, linewidth = 1.0, color = 'blue')
ylabel("Modularity", fontsize = 16)
xlabel("Number of communities", fontsize = 16)
axhline(y = 0.46676348, color = 'red', linestyle = "--", linewidth = 1.5)
xticks(fontsize = 15)
yticks(fontsize = 15)
#title("Quality of partitions increasing the number of communities", fontsize = 14)
grid()

"""Export image with a resolution of 600 dpi"""

fig.savefig("graphic_10_inc.png", dpi = 600)

"""#Increase number of communities"""

#Detect four communities to get a partition into 10 communities
com10_per = ADD(com_per,10,W_jun_feb,dis_fact_half,mat_dis_norm_max)
print("Partition in 10 communities from the period from June 2021 to February 2022 with upper bound normalization:")
print_com(W_jun_feb,com10_per,name_mun)

"""Color map using geopandas"""

com10_new = transf(com10_per)
groups10 = com_to_group(com10_new)
#List of 10 colors to use for communities
colors = ['red','green','hotpink','blue','purple','yellow','cyan','magenta','brown','olive']
colors10 = change(groups10,colors)
#Color mapa
fig,ax = subplots(figsize = (24,23))
#Remove the axis
ax.axis('off')
#Assign community labels to colors
color_patches = generate_patches(colors)
ax.legend(handles = color_patches, prop = {'size':14}, loc = 4)
mun_gto_map.plot(color = colors10, ax = ax)

"""Export image with a resolution of 600 dpi"""

fig.savefig("part_10_inc.png", dpi = 600)