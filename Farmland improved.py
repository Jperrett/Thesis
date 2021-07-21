import numpy as np
from numpy import random
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd
import libpysal as lps
from matplotlib.pyplot import cm
from itertools import chain
import math
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as patches
pylab.rcParams['figure.figsize'] = 16, 12

### Loads in the image
image = Image.open('E:/Farmland/farm27.png') #Labeln_01 #farm01
im = Image.open('E:/Farmland/farm27.png')
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

test ={}
center =[]
plt.figure()
plt.clf()
pixel = []
cluster =[]
colors = color = ['b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k']
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


# sns.set(style = 'darkgrid')
# sns.set_palette(sns.color_palette("Paired"))
# cluster1 = range(n_clusters_)
# fg = sns.FacetGrid(data=ds1,hue = 'Cluster' , aspect=1.61)
# fg.map(plt.scatter, 'X', 'Y').add_legend()
# plt.ylim(max(plt.ylim()), min(plt.ylim()))

c = 0
# X1 = []
# Y1 = []
sns.set(style = "darkgrid")
fig = plt.figure()
for c in range(k):
    X1 = []
    Y1 = []
    cluster = ds1.loc[(ds1['Cluster'] == c)]
    X1.append(cluster['X'].tolist())
    Y1.append(cluster['Y'].tolist())
    X2 = list(chain.from_iterable(X1))
    Y2 = list(chain.from_iterable(Y1))
    sns.scatterplot(X2, Y2, palette = colors[c])
    c = c+1
plt.ylim(max(plt.ylim()), min(plt.ylim()))

### Weight Matrix
test3 =[]
test4 =[]
for b in range(im.width-1):
    for g in range(im.height-1):
        # print(b,g)
        origin = ds1.loc[(ds1['X'] == b) & (ds1['Y'] == g)]
        rows = ds1.loc[(ds1['X'] == b+1) & (ds1['Y'] == g) |(ds1['X'] == b) & 
                       (ds1['Y'] == g+1) |(ds1['X'] == b-1) & (ds1['Y'] == g) |
                       (ds1['X'] == b) & (ds1['Y'] == g-1) | 
                       (ds1['X'] == b-1) & (ds1['Y'] == g+1) | 
                       (ds1['X'] == b-1) & (ds1['Y'] == g-1) | 
                       (ds1['X'] == b+1) & (ds1['Y'] == g+1) | 
                       (ds1['X'] == b+1) & (ds1['Y'] == g-1)]
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
for m in range(k):
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

### XY Moran's I
R1 = []
G1 = []
B1 = []
Rclust = []
Gclust = []
Bclust = []
for m in range(n_clusters_ -1):
    cluster = ds1.loc[(ds1['Cluster'] == m)]
    R1.append(cluster['R'].tolist())
    G1.append(cluster['G'].tolist())
    B1.append(cluster['B'].tolist())
    R = list(chain.from_iterable(R1))
    B = list(chain.from_iterable(B1))
    G = list(chain.from_iterable(G1))
    r1 = sum(R)/len(R)
    g1 = sum(G)/len(G)
    b1 = sum(B)/len(B)
    Rclust.append(r1)
    Gclust.append(g1)
    Bclust.append(b1)
    m = m+1
    
Rbar = sum(Rclust)/len(Rclust)
Gbar = sum(Gclust)/len(Gclust)
Bbar = sum(Bclust)/len(Bclust)

RP = []
RQ = []
GP = []
GQ = []
BP = []
BQ = []
for n in range(len(Rclust)):
    Rp = Rclust[n] - Rbar
    Gp = Gclust[n] - Gbar
    Bp = Bclust[n] - Bbar
    RP.append(Rp)
    GP.append(Gp)
    BP.append(Bp)
    Rq = (Rp **2)
    Gq = (Gp **2)
    Bq = (Bp **2)
    RQ.append(Rq)
    GQ.append(Gq)
    BQ.append(Bq)
    n = n+1

denomr = sum(RQ)
denomg = sum(BQ)
denomb = sum(GQ)

CPr = np.zeros((k,k))
CPg = np.zeros((k,k))
CPb = np.zeros((k,k))
sCPr =[]
sCPg =[]
sCPb =[]
a = 0
b = 0
for a in range(k-1):
    for b in range(k-1):
        if dw[a,b] != 0:
            Cpr = RP[a]*RP[b]*dw[a,b]
            Cpg = GP[a]*GP[b]*dw[a,b]
            Cpb = BP[a]*BP[b]*dw[a,b]
            CPr[a,b] = Cpr
            CPg[a,b] = Cpg
            CPb[a,b] = Cpb
            sCPr.append(Cpr)
            sCPg.append(Cpg)
            sCPb.append(Cpb)
        else:
            CPr[a,b] = 0
            CPg[a,b] = 0
            CPb[a,b] = 0
        b = b+1
    a = a+1

SCPR =sum(sCPr)
SCPG =sum(sCPg)
SCPB =sum(sCPb) 

### Moran's I = (N/sum of weight matrix) * (sum of cross products*weight matrix /((X-Xbar)sq))
Moransr = ((k/np.sum(dw))* (SCPR/denomr))
Moransg = ((k/np.sum(dw))* (SCPG/denomg))
Moransb = ((k/np.sum(dw))* (SCPB/denomb))
Morans = [Moransr, Moransg, Moransb]

# moransn = []
# for i in range(len(Morans)):
#    moransn.append((Morans[i]-min(Morans))/(max(Morans)-min(Morans)))

moransn1 = []
for j in range(len(Morans)):
    moransn1.append((Morans[j]+1)/2)

### Weighted variance
# WV = (sum of (segment area * variance)/(sum of area))
c= 0
Nomr = []
Nomg = []
Nomb = []
for c in range(n_clusters_):
    R = []
    G = []
    B = []
    cluster = ds1.loc[(ds1['Cluster'] == c)]
    R.append(cluster['R'].tolist())
    G.append(cluster['G'].tolist())
    B.append(cluster['B'].tolist())
    R1 = list(chain.from_iterable(R))
    G1 = list(chain.from_iterable(G))
    B1 = list(chain.from_iterable(B))
    r1 = sum(R1)/len(cluster)
    b1 = sum(B1)/len(cluster)
    g1 = sum(G1)/len(cluster)
    
    e = 0
    reuclidean = []
    geuclidean = []
    beuclidean = []
    for e in range(len(R1)):
        rdist = R1[e] - int(r1)
        gdist = G1[e] - int(g1)
        bdist = B1[e] - int(b1)
        r = np.array([R1[e], int(r1)])
        g = np.array([B1[e], int(b1)])
        b = np.array([G1[e], int(g1)])
        euchr = np.var(r)
        euchg = np.var(g)
        euchb = np.var(b)
        # euch = math.sqrt((xdist**2)+(ydist**2))
        # euch1 = euch**2
        reuclidean.append(euchr)
        geuclidean.append(euchg)
        beuclidean.append(euchb)

    # variance = np.var(euclidean)
    Nomr.append(sum(reuclidean)*len(cluster))
    Nomg.append(sum(geuclidean)*len(cluster))
    Nomb.append(sum(beuclidean)*len(cluster))
    c = c+1

NomR = sum(Nomr)
NomG = sum(Nomg)
NomB = sum(Nomb)
WVr = NomR/len(pixels)
WVg = NomG/len(pixels)
WVb = NomB/len(pixels)
WV = [WVr, WVg, WVb]

# WVn = []
# for i in range(len(WV)):
#    WVn.append((WV[i]-min(WV))/(max(WV)-min(WV)))

Rt = []
Gt = []
Bt = []
Vr = []
Vg = []
Vb = []
Rt.append(ds1['R'].tolist())
Gt.append(ds1['G'].tolist())
Bt.append(ds1['B'].tolist())
R1t = list(chain.from_iterable(Rt))
G1t = list(chain.from_iterable(Gt))
B1t = list(chain.from_iterable(Bt))
r1t = sum(R1t)/len(R1t)
g1t = sum(G1t)/len(G1t)
b1t = sum(B1t)/len(B1t)
for e in range(len(R1t)):
    r = np.array([R1t[e], int(r1t)])
    g = np.array([B1t[e], int(b1t)])
    b = np.array([G1t[e], int(g1t)])
    Vr.append(np.var(r))
    Vg.append(np.var(g))
    Vb.append(np.var(b))
WVR = WVr/sum(Vr)
WVG = WVb/sum(Vg)
WVB = WVg/sum(Vb)
WVN = [WVR, WVG, WVB]

print('Weighted Variance:', WVN)
print('Moran\'s I:', moransn1)

### Global Score
GS = (1/3) * sum(WVN + moransn1)

# pd.DataFrame(CP).to_csv(r'C:\Users\jorda\OneDrive\Documents\Thesis.csv')

### IoU (X,Y)

c = 0
# X1 = []
# Y1 = []
sns.set(style = "darkgrid")
fig = plt.figure()
for c in range(k):
    X1 = []
    Y1 = []
    dist =[]
    cluster = ds1.loc[(ds1['Cluster'] == c)]
    X1.append(cluster['X'].tolist())
    Y1.append(cluster['Y'].tolist())
    X2 = list(chain.from_iterable(X1))
    Y2 = list(chain.from_iterable(Y1))
    x1 = sum(X2)/len(cluster)
    y1 = sum(Y2)/len(cluster)
    sns.scatterplot(X2, Y2, palette = colors[c])
    for e in range(len(X2)):
        xdist = X2[e] - int(x1)
        ydist = Y2[e] - int(y1)
        euch = math.sqrt((xdist**2)+(ydist**2))
        dist.append(euch)
        maxd= max(dist)
    cornerx = (x1 - maxd)
    cornerx1 = (x1 + maxd)
    cornery = (y1 - maxd)
    cornery1 = (y1 + maxd)
    # ax = fig.add_subplot(111)
    # if cornerx >= 0 and cornery >= 0:
    #     rect1 = patches.Rectangle((cornerx, cornery),
    #                                   maxd, maxd,color = 'black', fill = False)
    #     ax.add_patch(rect1)
    # else:
    #     rect1 = patches.Rectangle((cornerx1, cornery1),
    #                                   -(maxd), -(maxd),
    #                                   color = 'black', fill = False)
    #     ax.add_patch(rect1)
    c = c+1
plt.ylim(max(plt.ylim()), min(plt.ylim()))

interarea = []
IOU = []
for c in range(n_clusters_-1):
    X1 = []
    Y1 = []
    X12 = []
    Y12 = []
    dist = []
    dist1 = []
    cluster = ds1.loc[(ds1['Cluster'] == c)]
    X1.append(cluster['X'].tolist())
    Y1.append(cluster['Y'].tolist())
    X2 = list(chain.from_iterable(X1))
    Y2 = list(chain.from_iterable(Y1))
    x1 = sum(X2)/len(cluster)
    y1 = sum(Y2)/len(cluster)
    for e in range(len(X2)):
        xdist = X2[e] - int(x1)
        ydist = Y2[e] - int(y1)
        euch = math.sqrt((xdist**2)+(ydist**2))
        dist.append(euch)
        maxd= max(dist)
    cornerx = (x1 - maxd)
    cornerx1 = (x1 + maxd)
    cornery = (y1 - maxd)
    cornery1 = (y1 + maxd)
    # print(cornerx, cornery)
    # print(cornerx1, cornery1)
    
    if c < (n_clusters_ - 1):
        cluster1 = ds1.loc[(ds1['Cluster'] == c+1)]
        X12.append(cluster1['X'].tolist())
        Y12.append(cluster1['Y'].tolist())
        X22 = list(chain.from_iterable(X12))
        Y22 = list(chain.from_iterable(Y12))
        x12 = sum(X22)/len(cluster1)
        y12 = sum(Y22)/len(cluster1)
        for e in range(len(X22)):
            xdist = X22[e] - int(x12)
            ydist = Y22[e] - int(y12)
            euch1 = math.sqrt((xdist**2)+(ydist**2))
            dist1.append(euch1)
            maxd1= max(dist1)
        cornerx2 = (x12 - maxd1)
        cornerx12 = (x12 + maxd1)
        cornery2 = (y12 - maxd1)
        cornery12 = (y12 + maxd1)
        # print(cornerx2, cornery2)
        # print(cornerx12, cornery12)
    
    
        dx = min(cornerx1, cornerx12) - max(cornerx, cornerx2)
        dy = min(cornery1, cornery12) - max(cornery, cornery2)
        if (dx>=0) and (dy>=0):
                interarea.append(dx*dy)
        else:
            interarea.append(0.0)
    else:
        cornerx2 = 0
        cornerx12 = 0
        cornery2 = 0
        cornery12 = 0
    
    area = (cornerx1 - cornerx)*(cornery1-cornery)
    area1 = (cornerx12 - cornerx2)*(cornery12-cornery2)
    union = (area+area1)-interarea[c]
    if union > 0:
        iou = (interarea[c]/union)
        IOU.append(iou)
    else:
        IOU.append(0.0)




### 3D plot

sns.set(style = "darkgrid")
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlabel("R")
ax.set_ylabel("G")
ax.set_zlabel("B")


max_dist=[]
c = 0
R1 = []
G1 = []
B1 = []
while c in range(n_clusters_):
    cluster = ds1.loc[(ds1['Cluster'] == c)]
    R = cluster['R'].tolist()
    R1.append(R)
    G = cluster['G'].tolist()
    G1.append(G)
    B = cluster['B'].tolist()
    B1.append(B)
    e = 0
    dist = []
    while e in range(len(cluster)):
        Xdist = R[e] - cluster_centers[c][0]
        Ydist = G[e] - cluster_centers[c][1]
        Zdist = B[e] - cluster_centers[c][2]
        part1 = math.sqrt((Xdist**2)+(Ydist**2))
        part2 = math.sqrt((part1**2)+(Zdist**2))
        dist.append(part2)
        e = e+1
    maxd = max(dist)
    max_dist.append(maxd)
    ax.scatter(R, G, B, color = colors[c])
    ax.plot
    c = c+1

### Label IOU  
image = Image.open('E:/Farmland/Labeln_27.png') #Labeln_01 #farm01
im = Image.open('E:/Farmland/Labeln_27.png')
width, height = im.size
image = np.array(image)
original_shape = image.shape
plt.figure()
plt.imshow(image)

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

unique = []
for e in range(len(arr)):
    RGB = list(arr[e])
    if RGB not in unique:
        unique.append(RGB)

interarea = []
IOUlabel = []
for c in range(len(unique)-1):
    X1 = []
    Y1 = []
    X12 = []
    Y12 = []
    dist = []
    dist1 = []
    cluster = df.loc[(df['R'] == unique[c][0]) & (df['G'] == unique[c][1]) & (df['B'] == unique[c][2])]
    X1.append(cluster['X'].tolist())
    Y1.append(cluster['Y'].tolist())
    X2 = list(chain.from_iterable(X1))
    Y2 = list(chain.from_iterable(Y1))
    x1 = sum(X2)/len(cluster)
    y1 = sum(Y2)/len(cluster)
    for e in range(len(X2)):
        xdist = X2[e] - int(x1)
        ydist = Y2[e] - int(y1)
        euch = math.sqrt((xdist**2)+(ydist**2))
        dist.append(euch)
        maxd= max(dist)
    cornerx = (x1 - maxd)
    cornerx1 = (x1 + maxd)
    cornery = (y1 - maxd)
    cornery1 = (y1 + maxd)
    # print(cornerx, cornery)
    # print(cornerx1, cornery1)
    
    if c < (len(unique)- 1):
        cluster1 = df.loc[(df['R'] == unique[c+1][0]) & (df['G'] == unique[c+1][1]) & (df['B'] == unique[c+1][2])]
        X12.append(cluster1['X'].tolist())
        Y12.append(cluster1['Y'].tolist())
        X22 = list(chain.from_iterable(X12))
        Y22 = list(chain.from_iterable(Y12))
        x12 = sum(X22)/len(cluster1)
        y12 = sum(Y22)/len(cluster1)
        for e in range(len(X22)):
            xdist = X22[e] - int(x12)
            ydist = Y22[e] - int(y12)
            euch1 = math.sqrt((xdist**2)+(ydist**2))
            dist1.append(euch1)
            maxd1= max(dist1)
        cornerx2 = (x12 - maxd1)
        cornerx12 = (x12 + maxd1)
        cornery2 = (y12 - maxd1)
        cornery12 = (y12 + maxd1)
        # print(cornerx2, cornery2)
        # print(cornerx12, cornery12)
    
    
        dx = min(cornerx1, cornerx12) - max(cornerx, cornerx2)
        dy = min(cornery1, cornery12) - max(cornery, cornery2)
        if (dx>=0) and (dy>=0):
                interarea.append(dx*dy)
        else:
            interarea.append(0.0)
    else:
        cornerx2 = 0
        cornerx12 = 0
        cornery2 = 0
        cornery12 = 0
    
    area = (cornerx1 - cornerx)*(cornery1-cornery)
    area1 = (cornerx12 - cornerx2)*(cornery12-cornery2)
    union = (area+area1)-interarea[c]
    if union > 0:
        iou = (interarea[c]/union)
        IOUlabel.append(iou)
    else:
        IOUlabel.append(0.0)

print('Bandwidth:', bandwidth)
print("Number of estimated clusters : %d" % n_clusters_)
print('Weighted Variance:', WV)
print('Moran\'s I:', Morans)
print('Mean-Shift IOU:', sum(IOU)/len(IOU))
print('Label IOU:', sum(IOUlabel)/len(IOUlabel))





