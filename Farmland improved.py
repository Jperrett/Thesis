import numpy as np
from numpy import random
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd
from libpysal.weights import lat2W
from esda.moran import Moran
import libpysal as lps
from matplotlib.pyplot import cm
from itertools import chain
import math
pylab.rcParams['figure.figsize'] = 16, 12

### Loads in the image
image = Image.open('E:/Farmland/farm01.png')
im = Image.open('E:/Farmland/farm01.png')
width, height = im.size
image = np.array(image)
original_shape = image.shape

### Flattening the image.
X = np.reshape(image, [-1, 3])
plt.imshow(image)

### Setting the Bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.07, n_samples=500) #0.07
print(bandwidth)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
flat = ms.fit(X)
labels1 = ms.labels_
print(labels1.shape)

cluster_centers = ms.cluster_centers_
print(cluster_centers.shape)

labels_unique = np.unique(labels1)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)


segmented_image = np.reshape(labels1, original_shape[:2])

plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.axis('off')
plt.title('Image After Mean Shift')

# Getting pixel location and RGB values
orig_pixel_map = im.load()
i,j = 0,0 
pixels1 =[]
locationx=[]
locationy=[]
for i in range(im.width-1):
    for j in range(im.height-1):
        pixel = orig_pixel_map[i, j] #Changed I and J round 10/07
        #print(i,j)
        #print(f"\n pixel: {pixel}")
        pixels1.append(pixel)
        locationx.append(i)
        locationy.append(j)
        j = j+1
    i = i+1
arr = np.array(pixels1)  
df = pd.DataFrame(data = arr, columns = ('R','G','B'))
df['X'] = locationx
df['Y'] = locationy

x = cluster_centers[:,0]
y = cluster_centers[:,1]
z = cluster_centers[:,2]

test = {}
center =[]
fig = plt.figure()
ax = plt.axes()
plt.figure()
plt.clf()
W = []
V = []
rowsX =[]
rowsY =[]
pixel = []
cluster =[]
colors = color = ['b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k']
for k, col in zip(range(n_clusters_), colors):
    print(k)
    my_members = labels1 == k
    cluster_center = cluster_centers[k]
    plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    ### Splitting the data into the different clusters.
    test[k] = X[my_members]
    center.append(cluster_center)
    test1 = test[k]
    pixel.append(test1)
    g = 0
    while g in range(len(test1)):
        cluster.append(k)
        g = g+1
    
pixels = list(chain.from_iterable(pixel))  
ds = pd.DataFrame(data = pixels, columns = ('R', 'G', 'B'))
ds['Cluster'] = cluster
ds1 = ds.merge(df, how='right', on=['R', 'G', 'B'], sort=False)
ds1 = ds1.drop_duplicates()
ds1 = ds1.sort_values(by=['X','Y'])

import seaborn
seaborn.set(style = 'whitegrid')
seaborn.set_palette(seaborn.color_palette("Paired"))
cluster1 = range(n_clusters_)
fg = seaborn.FacetGrid(data=ds1,hue = 'Cluster' , aspect=1.61)
fg.map(plt.scatter, 'X', 'Y').add_legend()
plt.ylim(max(plt.ylim()), min(plt.ylim()))


test3 =[]
test4 =[]
for b in range(im.width-1):
    for g in range(im.height-1):
        # print(b,g)
        origin = ds1.loc[(ds1['X'] == b) & (ds1['Y'] == g)]
        rows = ds1.loc[(ds1['X'] == b+1) & (ds1['Y'] == g) |(ds1['X'] == b) & 
                       (ds1['Y'] == g+1) |(ds1['X'] == b-1) & (ds1['Y'] == g)|
                       (ds1['X'] == b) & (ds1['Y'] == g-1)]
        test3.append(rows['Cluster'].tolist())   
        test4.append(origin['Cluster'].tolist())

i = 0
j = 0
te = []
for i in range(len(test3)-1):
    for j in range(len(test3[i])-1):
        print(i,j)
        if test4[i][0] != test3[i][j]:
            print(test4[i], test3[i][j])
            t = test4[i][0], test3[i][j]
            te.append(t)
            

df = pd.DataFrame(te, columns = ('Original', 'Next'))
# df1 = df['Original'].value_counts()
m = 0
l = 0
d = np.zeros((k,k))
dw = np.zeros((k,k))
df1 = pd.DataFrame()
for m in range(27):
    rows1 = df.loc[(df['Original'] == m)]
    row = rows1['Next'].unique()
    row = [x for x in row if math.isnan(x) == False]
    row = sorted(row)
    # for l in range(27- len(row)):
    #     row.append(0)
    #     l = l+1
    # df1[m] = row
    # m = m+1
    for n in range(k):
        if n in row:
            dw[m,n] = (1/len(row))
            d[m,n] = 1
        else:
            d[m,n] = 0
            dw[m,n] = 0

R = []
G = []
B = []
cluster = ds1.loc[(ds1['Cluster'] == 0)]
R.append(cluster['R'].tolist())
G.append(cluster['G'].tolist())
B.append(cluster['B'].tolist())
R = list(chain.from_iterable(R))               
G = list(chain.from_iterable(G))
B = list(chain.from_iterable(B))
r = sum(R)/len(R)
g = sum(G)/len(G)
b = sum(B)/len(B)


R1 =[]
R1.append(ds1['R'].tolist())
R1 = list(chain.from_iterable(R1))
r1 = sum(R1)/len(R1)
G1 =[]
G1.append(ds1['G'].tolist())
G1 = list(chain.from_iterable(G1))
g1 = sum(G1)/len(G1)
B1 =[]
B1.append(ds1['B'].tolist())
B1 = list(chain.from_iterable(B1))
b1 = sum(B1)/len(B1)

xi = ((r - r1), (g-g1), (b-b1))
xi = sum(xi)/len(xi)
Xi2 = xi **2