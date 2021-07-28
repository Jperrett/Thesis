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
import scipy.io as sio
pylab.rcParams['figure.figsize'] = 16, 12

### Loads in the image
image = Image.open('E:/Farmland/farm27.png') #Labeln_01 #farm01
im = Image.open('E:/Farmland/farm27.png')
width, height = im.size
image = np.array(image)
original_shape = image.shape
IR = sio.loadmat('E:\Farmland\cube.mat')
IRdf = IR['cube']
print(IRdf.shape)

### Flattening the image.
X = np.reshape(image, [-1, 3])
plt.imshow(image)


# Getting pixel location and RGB values
orig_pixel_map = im.load()
i,j = 0,0 
pixels1 =[]
locationx=[]
locationy=[]
IR =[]
IRpix = []
for i in range(im.width):
    for j in range(im.height):
        pixel = orig_pixel_map[i, j] #Changed I and J round 10/07
        #print(i,j)
        #print(f"\n pixel: {pixel}")
        locationx.append(i)
        locationy.append(j)
        ir = IRdf[i][j][5][27]
        IR.append(ir)
        irpix = (pixel[0], pixel[1], pixel[2], ir)
        irpix = list(irpix)
        IRpix.append(irpix)
        pixels1.append(pixel)
        
    
arr = np.array(pixels1)  
df = pd.DataFrame(data = arr, columns = ('R','G','B'))
df['X'] = locationx
df['Y'] = locationy
df['IR'] = IR



### Setting the Bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.07, n_samples=500) #0.07
print(bandwidth)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
flat = ms.fit(X)
labels = ms.labels_
print(labels.shape)

cluster_centers = ms.cluster_centers_
print(cluster_centers.shape)

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

### With IR bandwidth
bandwidth = estimate_bandwidth(IRpix, quantile=0.07, n_samples=500) #0.07
print(bandwidth)

ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
flatir = ms.fit(IRpix)
labelsir = ms.labels_
print(labelsir.shape)

cluster_centersir = ms.cluster_centers_
print(cluster_centersir.shape)

labels_uniqueir = np.unique(labelsir)
n_clusters_ir = len(labels_uniqueir)

print("number of estimated clusters : %d" % n_clusters_)

segmented_image = np.reshape(labels, original_shape[:2])

plt.figure(2)
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis('off')
plt.title('Original Image')
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.axis('off')
plt.title('Image After Mean Shift')

x = cluster_centers[:,0]
y = cluster_centers[:,1]
z = cluster_centers[:,2]

test ={}
center =[]
plt.figure()
plt.clf()
pixel = []
cluster =[]
colors = color = ['b','g','r','y','m','c','k','b','g','r','y','m','c','k','b','g','r','y','m','c','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k']
for k, col in zip(range(n_clusters_), colors):
        print(k)
        my_members = labels == k
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
ds1 = df.merge(ds, how='right', on=['R', 'G', 'B'], sort=False)
ds1 = ds1.drop_duplicates()
ds1 = ds1.sort_values(by=['X','Y'])


# sns.set(style = 'darkgrid')
# sns.set_palette(sns.color_palette("Paired"))
# cluster1 = range(n_clusters_)
# fg = sns.FacetGrid(data=ds1,hue = 'Cluster' , aspect=1.61)
# fg.map(plt.scatter, 'X', 'Y').add_legend()
# plt.ylim(max(plt.ylim()), min(plt.ylim()))

# c = 0
# X1 = []
# Y1 = []
# sns.set(style = "darkgrid")
# fig = plt.figure()
# for c in range(k):
#     X1 = []
#     Y1 = []
#     cluster = ds1.loc[(ds1['Cluster'] == c)]
#     X1.append(cluster['X'].tolist())
#     Y1.append(cluster['Y'].tolist())
#     X2 = list(chain.from_iterable(X1))
#     Y2 = list(chain.from_iterable(Y1))
#     sns.scatterplot(X2, Y2, palette = colors[c])
#     c = c+1
# plt.ylim(max(plt.ylim()), min(plt.ylim()))

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
IR1 = []
Rclust = []
Gclust = []
Bclust = []
IRclust = []
for m in range(n_clusters_ -1):
    cluster = ds1.loc[(ds1['Cluster'] == m)]
    R1.append(cluster['R'].tolist())
    G1.append(cluster['G'].tolist())
    B1.append(cluster['B'].tolist())
    IR1.append(cluster['IR'].tolist())
    R = list(chain.from_iterable(R1))
    B = list(chain.from_iterable(B1))
    G = list(chain.from_iterable(G1))
    Ir1 = list(chain.from_iterable(IR1))
    r1 = sum(R)/len(R)
    g1 = sum(G)/len(G)
    b1 = sum(B)/len(B)
    ir1 = sum(Ir1)/len(Ir1)
    Rclust.append(r1)
    Gclust.append(g1)
    Bclust.append(b1)
    IRclust.append(ir1)
    m = m+1
    
Rbar = sum(Rclust)/len(Rclust)
Gbar = sum(Gclust)/len(Gclust)
Bbar = sum(Bclust)/len(Bclust)
IRbar = sum(IRclust)/len(IRclust)

RP = []
RQ = []
GP = []
GQ = []
BP = []
BQ = []
IRP = []
IRQ = []
for n in range(len(Rclust)):
    Rp = Rclust[n] - Rbar
    Gp = Gclust[n] - Gbar
    Bp = Bclust[n] - Bbar
    IRp = IRclust[n] - IRbar
    RP.append(Rp)
    GP.append(Gp)
    BP.append(Bp)
    IRP.append(IRp)
    Rq = (Rp **2)
    Gq = (Gp **2)
    Bq = (Bp **2)
    IRq = (IRp **2)
    RQ.append(Rq)
    GQ.append(Gq)
    BQ.append(Bq)
    IRQ.append(IRq)
    n = n+1

denomr = sum(RQ)
denomg = sum(BQ)
denomb = sum(GQ)
denomIR = sum(IRQ)

CPr = np.zeros((k,k))
CPg = np.zeros((k,k))
CPb = np.zeros((k,k))
CPIR = np.zeros((k,k))
sCPr =[]
sCPg =[]
sCPb =[]
sCPIR =[]
a = 0
b = 0
for a in range(k-1):
    for b in range(k-1):
        if dw[a,b] != 0:
            Cpr = RP[a]*RP[b]*dw[a,b]
            Cpg = GP[a]*GP[b]*dw[a,b]
            Cpb = BP[a]*BP[b]*dw[a,b]
            CpIR = IRP[a]*IRP[b]*dw[a,b]
            CPr[a,b] = Cpr
            CPg[a,b] = Cpg
            CPb[a,b] = Cpb
            CPIR[a,b] = CpIR
            sCPr.append(Cpr)
            sCPg.append(Cpg)
            sCPb.append(Cpb)
            sCPIR.append(CpIR)
        else:
            CPr[a,b] = 0
            CPg[a,b] = 0
            CPb[a,b] = 0
            CPIR[a,b] = 0
        b = b+1
    a = a+1

SCPR =sum(sCPr)
SCPG =sum(sCPg)
SCPB =sum(sCPb)
SCPIR =sum(sCPIR)

### Moran's I = (N/sum of weight matrix) * (sum of cross products*weight matrix /((X-Xbar)sq))
Moransr = ((k/np.sum(dw))* (SCPR/denomr))
Moransg = ((k/np.sum(dw))* (SCPG/denomg))
Moransb = ((k/np.sum(dw))* (SCPB/denomb))
MoransIR = ((k/np.sum(dw))* (SCPIR/denomIR))
Morans = [Moransr, Moransg, Moransb, MoransIR]

moransn = []
for i in range(len(Morans)):
   moransn.append((Morans[i]-min(Morans))/(max(Morans)-min(Morans)))

moransn1 = []
for j in range(len(Morans)):
    moransn1.append((Morans[j]+1)/2)

### Weighted variance
# WV = (sum of (segment area * variance)/(sum of area))
c= 0
Nomr = []
Nomg = []
Nomb = []
Nomir = []
for c in range(n_clusters_):
    R = []
    G = []
    B = []
    IR = []
    cluster = ds1.loc[(ds1['Cluster'] == c)]
    R.append(cluster['R'].tolist())
    G.append(cluster['G'].tolist())
    B.append(cluster['B'].tolist())
    IR.append(cluster['IR'].tolist())
    R1 = list(chain.from_iterable(R))
    G1 = list(chain.from_iterable(G))
    B1 = list(chain.from_iterable(B))
    IR1 = list(chain.from_iterable(IR))
    r1 = sum(R1)/len(cluster)
    b1 = sum(B1)/len(cluster)
    g1 = sum(G1)/len(cluster)
    ir1 = sum(IR1)/len(cluster)
    
    e = 0
    reuclidean = []
    geuclidean = []
    beuclidean = []
    ireuclidean = []
    for e in range(len(R1)):
        rdist = R1[e] - int(r1)
        gdist = G1[e] - int(g1)
        bdist = B1[e] - int(b1)
        irdist = IR1[e] - ir1
        r = np.array([R1[e], int(r1)])
        b = np.array([B1[e], int(b1)])
        g = np.array([G1[e], int(g1)])
        ir = np.array([IR1[e], ir1])
        euchr = np.var(r)
        euchg = np.var(g)
        euchb = np.var(b)
        euchir = np.var(ir)
        # euch = math.sqrt((xdist**2)+(ydist**2))
        # euch1 = euch**2
        reuclidean.append(euchr)
        geuclidean.append(euchg)
        beuclidean.append(euchb)
        ireuclidean.append(euchir)

    # variance = np.var(euclidean)
    Nomr.append(sum(reuclidean)*len(cluster))
    Nomg.append(sum(geuclidean)*len(cluster))
    Nomb.append(sum(beuclidean)*len(cluster))
    Nomir.append(sum(ireuclidean)*len(cluster))
    c = c+1

NomR = sum(Nomr)
NomG = sum(Nomg)
NomB = sum(Nomb)
NomIR = sum(Nomir)
WVr = NomR/len(pixels)
WVg = NomG/len(pixels)
WVb = NomB/len(pixels)
WVir = NomIR/len(pixels)
WV = [WVr, WVg, WVb, WVir]

WVn = []
for i in range(len(WV)):
    WVn.append((WV[i]-min(WV))/(max(WV)-min(WV)))

Rt = []
Gt = []
Bt = []
IRt = []
Vr = []
Vg = []
Vb = []
VIR = []
Rt.append(ds1['R'].tolist())
Gt.append(ds1['G'].tolist())
Bt.append(ds1['B'].tolist())
IRt.append(ds1['IR'].tolist())
R1t = list(chain.from_iterable(Rt))
G1t = list(chain.from_iterable(Gt))
B1t = list(chain.from_iterable(Bt))
IR1t = list(chain.from_iterable(IRt))
r1t = sum(R1t)/len(R1t)
g1t = sum(G1t)/len(G1t)
b1t = sum(B1t)/len(B1t)
ir1t = sum(IR1t)/len(IR1t)
for e in range(len(R1t)):
    r = np.array([R1t[e], int(r1t)])
    g = np.array([B1t[e], int(b1t)])
    b = np.array([G1t[e], int(g1t)])
    ir = np.array([IR1t[e], ir1t])
    Vr.append(np.var(r))
    Vg.append(np.var(g))
    Vb.append(np.var(b))
    VIR.append(np.var(ir))
WVR = WVr/sum(Vr)
WVG = WVb/sum(Vg)
WVB = WVg/sum(Vb)
WVIR = WVir/sum(VIR)
WVN = [WVR, WVG, WVB, WVIR]

print('Weighted Variance:', WVN)
print('Moran\'s I:', moransn1)

### Global Score
GS = (1/4) * sum(WVn + moransn)

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
image1 = Image.open('E:/Farmland/Labeln_27.png') #Labeln_01 #farm01
im1 = Image.open('E:/Farmland/Labeln_27.png')
plt.figure()
plt.imshow(image1)
width, height = im1.size
image = np.array(image)
original_shape = image.shape
plt.figure()
plt.imshow(image1)

orig_pixel_map = im1.load()
i,j = 0,0 
pixels_ =[]
locationx=[]
locationy=[]
for i in range(im1.width-1):
    for j in range(im1.height-1):
        pixel = orig_pixel_map[i, j] #Changed I and J round 10/07
        #print(i,j)
        #print(f"\n pixel: {pixel}")
        pixels_.append(pixel)
        locationx.append(i)
        locationy.append(j)
        j = j+1
    i = i+1
arr = np.array(pixels_)  
df_ = pd.DataFrame(data = arr, columns = ('R','G','B'))
df_['X'] = locationx
df_['Y'] = locationy

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
    cluster = df_.loc[(df_['R'] == unique[c][0]) & (df_['G'] == unique[c][1]) & (df_['B'] == unique[c][2])]
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
        cluster1 = df_.loc[(df_['R'] == unique[c+1][0]) & (df_['G'] == unique[c+1][1]) & (df_['B'] == unique[c+1][2])]
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

### Accuracy
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=len(unique), random_state=0).fit(cluster_centersir)
labels = kmeans.labels_
centers = kmeans.cluster_centers_

sns.set(style = "darkgrid")
fig = plt.figure()
ax = fig.add_subplot(111, projection = '3d')
ax.set_xlabel("R")
ax.set_ylabel("G")
ax.set_zlabel("B")
for l in range(n_clusters_):
    R = cluster_centersir[l][0]
    G = cluster_centersir[l][1]
    B = cluster_centersir[l][2]
    ax.scatter(R, G, B, color = colors[labels[l]], alpha = 0.5)
    ax.scatter(centers[labels[l]][0],centers[labels[l]][1],centers[labels[l]][2], color = colors[labels[l]],alpha = 1)

fig = plt.figure()
sns.set(style = "darkgrid")
for m in range(k):
    clustera = ds1.loc[(ds1['Cluster'] == m)]
    Xa = clustera['X'].tolist()
    Ya = clustera['Y'].tolist()
    sns.scatterplot(Xa, Ya, palette = colors[labels[m]])
plt.ylim(max(plt.ylim()), min(plt.ylim()))

### label
test1 = []
for t in range(len(unique)):
    test = df_.loc[(df_['R'] == unique[t][0]) & (df_['G'] == unique[t][1]) & (df_['B'] == unique[t][2])]
    test['Clust'] = t
    test1.append(test)

### Clustering
C1X = []
C1Y = []
C2X = []
C2Y = []
C3X = []
C3Y = []
C4X = []
C4Y = []
C5X = []
C5Y = []
for n in range(k):
    clusterb = ds1.loc[(ds1['Cluster'] == n)]
    if labels[n] == 0:
        C1X.append(clusterb['X'].tolist())
        C1Y.append(clusterb['Y'].tolist())
    if labels[n] == 1:
        C2X.append(clusterb['X'].tolist())
        C2Y.append(clusterb['Y'].tolist())
    if labels[n] == 2:
        C3X.append(clusterb['X'].tolist())
        C3Y.append(clusterb['Y'].tolist())
    if labels[n] == 3:
        C4X.append(clusterb['X'].tolist())
        C4Y.append(clusterb['Y'].tolist())
    if labels[n] == 4:
        C5X.append(clusterb['X'].tolist())
        C5Y.append(clusterb['Y'].tolist())

C1X = list(chain.from_iterable(C1X))
C1Y = list(chain.from_iterable(C1Y))
C2X = list(chain.from_iterable(C2X))
C2Y = list(chain.from_iterable(C2Y))
C3X = list(chain.from_iterable(C3X))
C3Y = list(chain.from_iterable(C3Y))
C4X = list(chain.from_iterable(C4X))
C4Y = list(chain.from_iterable(C4Y))
C5X = list(chain.from_iterable(C5X))
C5Y = list(chain.from_iterable(C5Y))

colors1 = ['#80FF80', '#FF0000', '808000', '#008000', '#FF80FF']
fig = plt.figure()
sns.set(style = "darkgrid")
sns.scatterplot(C1X, C1Y, palette = colors1[2], s=15)
sns.scatterplot(C2X, C2Y, palette = colors1[0], s=15)
sns.scatterplot(C3X, C3Y, palette = colors1[1], s=15)
sns.scatterplot(C4X, C4Y, palette = colors1[3], s=15)
sns.scatterplot(C5X, C5Y, palette = colors1[4], s=15)
plt.ylim(max(plt.ylim()), min(plt.ylim()))


xclust =[]
Clusts = [C1X, C2X, C3X, C4X, C5X]
ClustsY = [C1Y,C2Y,C3Y,C4Y,C5Y]
centr = []
centg = []
centb = []
f = 0
for f in range(len(Clusts)):
    for e in range(len(Clusts[f])):
            xclust.append(f)
            centr.append(centers[f][0])
            centg.append(centers[f][1])
            centb.append(centers[f][2])
        
Clustsx = list(chain.from_iterable(Clusts))
Clustsy = list(chain.from_iterable(ClustsY))
df1 = pd.DataFrame(data = Clustsx, columns = (['X']))
df1['Y'] = Clustsy
df1['Clust'] = xclust
df1['CentR'] = centr
df1['CentG'] = centg
df1['CentB'] = centb
df1 = df1.sort_values(by=['X','Y'])

# Label cluster 0 (Light Green) = clust 2 (green) 
# Label cluster 1 (Red) = clust 3 (Red)
# Label cluster 2 (Brown) = clust 1 (Blue)
# Label cluster 3 (Dark Green)  = clust 
# Label cluster 4 (Pink) = clust

PixClusts = []
PixClusts.append(df1['Clust'])
Pixclusts = list(chain.from_iterable(PixClusts))

Correct = []
Wrong = []
m = 0
i = 0
j = 0
for m in range(len(unique)):
    for i in range(im.width):
        for j in range(im.height):
            pix = df1.loc[(df1['X'] == i) & (df1['Y'] == j)]
            pixcluster = pix['Clust'].tolist()
            pix2 = test1[m].loc[(test1[m]['X'] == i) & (test1[m]['Y'] == j)]
            if len(pix2) == 1:
                pixcluster2 = pix2['Clust'].tolist()   
                if len(pixcluster) == 1:
                    if pixcluster[0] == pixcluster2[0]:
                        print('y')
                        result = [pix['X'], pix['Y'], pix['Clust']]
                        result = list(chain.from_iterable(result))
                        Correct.append(result)   
                    else:
                        print('n')    
                        # result = [pix['X'], pix['Y'], pix['Clust']]
                        # result = list(chain.from_iterable(result))
                        # results = result,m
                        # Wrong.append(results)




print('Bandwidth:', bandwidth)
print("Number of estimated clusters : %d" % n_clusters_)
print('Weighted Variance:', WVn)
print('Moran\'s I:', moransn)
print('Global Score:', GS)
print('Mean-Shift IOU:', sum(IOU)/len(IOU))
print('Label IOU:', sum(IOUlabel)/len(IOUlabel))

df2 = pd.DataFrame()
R = []
G = []
B = []
IR = []
for i in range(len(IRpix)):
    r = IRpix[i][0]
    g = IRpix[i][1]
    b = IRpix[i][2]
    ir = IRpix[i][3]
    R.append(r)
    G.append(g)
    B.append(b)
    IR.append(ir)
    
df2['R'] = R
df2['G'] = G
df2['B'] = B
df2['IR'] = IR

# Label cluster 0 (Light Green) = 128, 255, 128 
# Label cluster 1 (Red) = 255, 0, 0
# Label cluster 2 (Brown)  = 128, 128, 0
# Label cluster 3 (Dark Green) = 0, 128, 0 
# Label cluster 4 (Pink) = 255, 128, 255 

df1 = df1.drop(columns = ['CentR', 'CentG', 'CentB'], axis = 1)
q = df1.loc[df1['Clust'] == 0]

Cluster0 = unique[3]
Cluster1 = unique[2]
Cluster2 = unique[0]    
Cluster3 = unique[1]
Cluster4 = unique[4]
Clustert = [Cluster0, Cluster1, Cluster2, Cluster3, Cluster4]
attempt = []
r = 0
for r in range(len(Clustert)):
    NClust = []
    e = 0
    cluster = df_.loc[(df_['R'] == Clustert[r][0]) & (df_['G'] == Clustert[r][1]) & (df_['B'] == Clustert[r][2])]
    for e in range(len(cluster)):
        NClust.append(r)
    cluster['NClust'] = NClust
    attempt.append(cluster)

t = 0
u = 0
Correct =[] 
Wrong = []  
for t in range(im.width):
    for u in range(im.height):
        pos = df1.loc[(df1['X'] == t) & (df_['Y'] == u)]
        for i in range(len(attempt)):
            pos1 = attempt[i].loc[(df1['X'] == t) & (df_['Y'] == u)]
            if len(pos1) == 1:
                Pos = pos['Clust'].tolist()
                Pos1 = pos1['NClust'].tolist()
                if len(Pos) == 1:
                    if len(Pos1) == 1:
                        if Pos == Pos1:
                            print('y')
                            correct = [t,u,Pos[0]]
                            Correct.append(correct)
                        else:
                            print('n')
                            wrong = [t,u, Pos[0], Pos1[0]]
                            Wrong.append(wrong)