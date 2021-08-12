import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import pandas as pd
from itertools import chain
import math
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
from collections import Counter
import time
from sklearn import metrics
from sklearn.metrics import davies_bouldin_score
pylab.rcParams['figure.figsize'] = 16, 12

def imageload(image1):
    ''' Loads the image in and depicts the dimensions''' 
    images = Image.open(image1) #Labeln_01 #farm01
    global image, im, width, height, original_shape
    im = Image.open(image1)
    width, height = im.size
    image = np.array(images)
    original_shape = image.shape
    # plt.figure()
    # plt.imshow(image)
    return(image, im, width, height, original_shape)

train = sio.loadmat('E:\Farmland\ind_train.mat')
val = train.values()
vals = list(val)
GStrain = []
MSIOU = []
LabelIOU =[]
Accuracytrain = []
time_taken = []
Moranstest = []
WVtest = []
N_clusts = []
SHS = []
CHI = []
DBI = []

tr = 0

vals0 = [39,64]
# vals = list(vals[3][0][5])
for tr in range(len(vals0)):  #vals[3][0]
    vals1 = vals0[tr]
    if vals1 < 10:
        image1 = 'E:/Farmland/farm0'+str(vals1)+'.png'
    else:
        image1 = 'E:/Farmland/farm'+str(vals1)+'.png'
    imageload(image1)
    print(tr)
    IR = sio.loadmat('E:\Farmland\cube.mat')
    IRdf = IR['cube']
    print(IRdf.shape)
    
    ### Flattening the image.
    X = np.reshape(image, [-1, 3])
    plt.imshow(im)
    
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
            ir = IRdf[i][j][5][vals1]
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
    
    
    start_time = time.time()
    
    ### Setting the Bandwidth
    bandwidth = estimate_bandwidth(X, quantile=0.06, n_samples=500) #0.07
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
    bandwidth = estimate_bandwidth(IRpix, quantile=0.06, n_samples=500) #0.07
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
    
    time_taken.append(time.time() - start_time)
    print("--- %s seconds ---" % (time.time() - start_time))
    
    N_clusts.append(n_clusters_ir)
    segmented_image = np.reshape(labels, original_shape[:2])
    
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(im)
    plt.axis('off')
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image,cmap = 'plasma') #hot
    plt.axis('off')
    plt.title('Image After Mean Shift')
    
    x = cluster_centers[:,0]
    y = cluster_centers[:,1]
    z = cluster_centers[:,2]
    
    test ={}
    center =[]
    # plt.figure()
    # plt.clf()
    pixel = []
    cluster =[]
    colors = color = ['b','g','r','y','m','c','k','b','g','r','y','m','c','k','b','g','r','y','m','c','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k','b','g','r','c','m','y','k', 'b','g','r','y','m','c','k','b','g','r','y','m','c','k','b','g','r','y','m','c','k','b','g','r','y','m','c','k','b','g','r','y','m','c','k','b','g','r','y','m','c','k','b','g','r','y','m','c','k']
    for k, col in zip(range(n_clusters_), colors):
            print(k)
            my_members = labels == k
            cluster_center = cluster_centers[k]
            # plt.plot(X[my_members, 0], X[my_members, 1], col + '.')
            # plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            #           markeredgecolor='k', markersize=14)
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
    
    ### Pixels image
    fig = plt.figure()
    sns.set(style = "darkgrid")
    Colours = ['green', 'magenta','brown', 'red', 'gray', 'lime', 'black', 'yellow', 'midnightblue']
    m = 0
    for m in range(n_clusters_ir):
        clustera = ds1.loc[(ds1['Cluster'] == m)]
        Xa = clustera['X'].tolist()
        Ya = clustera['Y'].tolist()
        sns.scatterplot(Xa, Ya, palette = colors[m])
    plt.ylim(max(plt.ylim()), min(plt.ylim()))
    
    ## Weight Matrix
    oclust =[]
    nclust =[]
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
            oclust.append(rows['Cluster'].tolist())   
            nclust.append(origin['Cluster'].tolist())
    
    i = 0
    j = 0
    te = []
    for i in range(len(nclust)-1):
        for j in range(len(oclust[i])-1):
            print(i,j)
            if len(nclust[i]) >0:
                if nclust[i][0] != oclust[i][j]:
                    print(nclust[i], oclust[i][j])
                    t = nclust[i][0], oclust[i][j]
                    te.append(t)
                else:
                    t = 0, oclust[i][j]
                    te.append(t)
                
    df2 = pd.DataFrame(te, columns = ('Original', 'Next'))
    # df1 = df['Original'].value_counts()
    m = 0
    l = 0
    d = np.zeros((n_clusters_,n_clusters_))
    dw = np.zeros((n_clusters_,n_clusters_))
    df1 = pd.DataFrame()
    for m in range(n_clusters_):
        rows1 = df2.loc[(df2['Original'] == m)]
        row = rows1['Next'].unique()
        row = [x for x in row if math.isnan(x) == False]
        row = sorted(row)
        # for l in range(27- len(row)):
        #     row.append(0)
        #     l = l+1
        # df1[m] = row
        # m = m+1
        for n in range(n_clusters_):
            if n in row:
                dw[m,n] = (1/len(row))
                d[m,n] = 1
            else:
                d[m,n] = 0
                dw[m,n] = 0
    
    ### Moran's I
    R1 = []
    G1 = []
    B1 = []
    IR1 = []
    Rclust = []
    Gclust = []
    Bclust = []
    IRclust = []
    m=0
    for m in range(n_clusters_):
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
    
    CPr = np.zeros((n_clusters_,n_clusters_))
    CPg = np.zeros((n_clusters_,n_clusters_))
    CPb = np.zeros((n_clusters_,n_clusters_))
    CPIR = np.zeros((n_clusters_,n_clusters_))
    sCPr =[]
    sCPg =[]
    sCPb =[]
    sCPIR =[]
    a = 0
    b = 0
    for a in range(n_clusters_-1):
        for b in range(n_clusters_-1):
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
    Moransr = (((n_clusters_)/np.sum(dw))* (SCPR/denomr))
    Moransg = (((n_clusters_)/np.sum(dw))* (SCPG/denomg))
    Moransb = (((n_clusters_)/np.sum(dw))* (SCPB/denomb))
    MoransIR = (((n_clusters_)/np.sum(dw))* (SCPIR/denomIR))
    Morans = [Moransr, Moransg, Moransb, MoransIR]
    
    moransn = []
    for i in range(len(Morans)):
        moransn.append((Morans[i]-min(Morans))/(max(Morans)-min(Morans)))
    
    moransn1 = []
    for j in range(len(Morans)):
        moransn1.append((Morans[j]+1)/2)
    
    ## Weighted variance
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
        if len(cluster) > 0:
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
    GS = (1/4) * sum(WVN + moransn1)
    Moranstest.append(moransn1)
    WVtest.append(WVN)
    
    ### Loading in the data
    if vals1 < 10:
        image1 = 'E:/Farmland/Labeln_0'+str(vals1)+'.png'
    else:
        image1 = 'E:/Farmland/Labeln_'+str(vals1)+'.png'
    imageload(image1)
    plt.figure()
    plt.imshow(im)
    
    orig_pixel_map = im.load()
    i,j = 0,0 
    pixels_ =[]
    locationx=[]
    locationy=[]
    for i in range(im.width):
        for j in range(im.height):
            pixel = orig_pixel_map[i, j]
            #print(i,j)
            #print(f"\n pixel: {pixel}")
            pixels_.append(pixel)
            locationx.append(i)
            locationy.append(j)
            j = j+1
        i = i+1
    arr1 = np.array(pixels_)  
    df_ = pd.DataFrame(data = arr1, columns = ('R','G','B'))
    df_['X'] = locationx
    df_['Y'] = locationy
    
    unique = []
    for e in range(len(arr1)):
        RGB = list(arr1[e])
        if RGB not in unique:
            unique.append(RGB)
    
    
    ### Reducing the number of clusters
    nClusts = Counter(ds1['Cluster'])
    Vals = nClusts.values()
    Vals = list(Vals)
    Vals.sort(reverse =True)
    nums = Vals[:len(unique)]
    
    smallclust = []
    b = 0
    for b in range(len(nClusts)):
        if nClusts[b] < nums[-1]:
            smallclust.append(b)
            
    ds2 = ds1
    for x in range(im.height):
        for y in range(im.width):
            f = ds2.loc[(ds1['X'] == x) & (ds2['Y'] == y)]
            c = f['Cluster'].tolist()
            if len(c) > 0:
              if c[0] in smallclust:  
                    rows = ds1.loc[(ds2['X'] == x+1) & (ds2['Y'] == y) |
                                   (ds2['X'] == x) & (ds2['Y'] == y+1) |
                                   (ds2['X'] == x-1) & (ds2['Y'] == y) |
                                   (ds2['X'] == x) & (ds2['Y'] == y-1) | 
                                   (ds2['X'] == x-1) & (ds2['Y'] == y+1) | 
                                   (ds2['X'] == x-1) & (ds2['Y'] == y-1) | 
                                   (ds2['X'] == x+1) & (ds2['Y'] == y+1) | 
                                   (ds2['X'] == x+1) & (ds2['Y'] == y-1)|
                                   (ds2['X'] == x+2) & (ds2['Y'] == y) |
                                   (ds2['X'] == x) & (ds2['Y'] == y+2) |
                                   (ds2['X'] == x-2) & (ds2['Y'] == y) |
                                   (ds2['X'] == x) & (ds2['Y'] == y-2) | 
                                   (ds2['X'] == x-2) & (ds2['Y'] == y+2) | 
                                   (ds2['X'] == x-2) & (ds2['Y'] == y-2) | 
                                   (ds2['X'] == x+2) & (ds2['Y'] == y+2) | 
                                   (ds2['X'] == x+2) & (ds2['Y'] == y-2)]
                    
                    counts = rows['Cluster'].tolist()
                    countsn = []
                    j = 0
                    for j in range(len(counts)):
                        if counts[j] not in smallclust:
                            countsn.append(counts[j])
                    i = 0
                    amount =[]
                    for i in range(len(countsn)):
                        am = countsn.count(countsn[i])
                        if am not in amount:
                            am = str(countsn[i]), am
                            amount.append(am)
                    maxval = max(set(amount), key=amount.count)
                    maxval = int(maxval[0])
                    find = f.index.tolist()
                    ds2.at[find[0], 'Cluster'] = maxval

    Clusters = np.unique(ds2['Cluster'])
    Nclust = []
    c = 0
    for c in range(len(Clusters)):
        c1 = ds2.loc[ds2['Cluster'] == Clusters[c]]
        c1 = c1['Cluster'].tolist()
        for v in range(len(c1)):
            if c1[v] != c:
                Nclust.append(c)
            else:
                Nclust.append(Clusters[c])
        
    ds2['Nclust'] = Nclust
            
    point = Counter(ds2['Cluster'])
    points = np.unique(ds2['Cluster'])
    Centers =[]
    n = 0
    for n in range(len(point)):
        cluster2 = ds1.loc[ds1['Cluster']== points[n]]
        R = sum(cluster2['R'])/len(cluster2['R'])
        G = sum(cluster2['G'])/len(cluster2['G'])
        B = sum(cluster2['B'])/len(cluster2['B'])
        IR = sum(cluster2['IR'])/len(cluster2['IR'])
        Cluster_cent = [R, G, B, IR]
        Centers.append(Cluster_cent)

    ### Pixel Plot
    fig = plt.figure()
    sns.set(style = "darkgrid")
    Colours = ['green', 'magenta','brown', 'red', 'gray', 'lime', 'black', 'yellow', 'midnightblue']
    m = 0
    for m in range(len(unique)):
        clustera = ds2.loc[(ds2['Cluster'] == m)]
        Xa = clustera['X'].tolist()
        Ya = clustera['Y'].tolist()
        sns.scatterplot(Xa, Ya, palette = colors[m])
    plt.ylim(max(plt.ylim()), min(plt.ylim()))
    
     
    ### Graph showing what clusters were assigned where during cluster reduction,
    # sns.set(style = "darkgrid")
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection = '3d')
    # ax.set_xlabel("R")
    # ax.set_ylabel("G")
    # ax.set_zlabel("B")
    # for l in range(n_clusters_):
    #     R = cluster_centersir[l][0]
    #     G = cluster_centersir[l][1]
    #     B = cluster_centersir[l][2]
    #     ax.scatter(R, G, B, color = colors[labels[l]], alpha = 0.5)
    #     ax.scatter(centers[labels[l]][0],centers[labels[l]][1],centers[labels[l]][2], color = colors[labels[l]],alpha = 1)
    
    ### Accuracy
    ### label
    label = []
    for t in range(len(unique)):
        lab = df_.loc[(df_['R'] == unique[t][0]) & (df_['G'] == unique[t][1]) & (df_['B'] == unique[t][2])]
        lab['Clust'] = t
        label.append(lab)
    
    u = 0
    j = 0
    Label = pd.concat([label[0],label[1]], ignore_index=True)
    for u in range(len(label)): 
            Label = pd.concat([Label,label[u]], ignore_index=True)
            Label = Label.sort_values(by=(['X','Y']))
    Label = Label.drop_duplicates()
    
    ### Clustering    
    x = 0
    y = 0
    Correct = []
    Wrong = []
    for x in range(im.height):
        for y in range(im.width):
            pos = ds2.loc[(ds1['X']==x) & (ds2['Y']==y)]
            Pos = pos['Nclust'].tolist()
            pos1 = Label.loc[(Label['X']==x) & (Label['Y']==y)]
            Pos1 = pos1['Clust'].tolist()
            if len(Pos) > 0:
                if len(Pos1) > 0:
                    if Pos[0] == Pos1[0]:
                        dets = [x, y, Pos]
                        Correct.append(dets)
                    else:
                        dets1 = [x, y, Pos[0], Pos1[0]]
                        Wrong.append(dets1)
    
    Accuracy = (len(Correct)/len(arr))*100
    matrix = np.zeros((len(unique),len(unique)))
    
    s = []
    for a in range(len(Wrong)): 
        s.append((Wrong[a][2], Wrong[a][3]))
               
    a1 = Counter(s)
    for b in range(len(unique)):
        for c in range(len(unique)):
            a2 = a1[b,c]           
            matrix[c,b] = a2  
               
    S = []
    for f in range(len(Correct)):
        Val = Correct[f][2]
        S.append(Val[0])
        
    z = Counter(S)
    for g in range(len(unique)):    
        matrix[g,g] = z[g]
    
    Accur = []
    x = sum(matrix)
    for y in range(len(unique)):
        if x[y] > 0:
            Acc = (matrix[y,y]/x[y])*100
            Accur.append(Acc)
       
    ### Confusion Matrix    
    plt.figure()
    plot = sns.heatmap(matrix, annot=True, cmap = 'YlGnBu')
    plot.set_xlabel('Algorithm\'s Clusters')
    plot.set_ylabel('Labelled Clusters')
    
    ## IOU
    IOU = []
    for m in range(len(unique)):
        cluster = ds2.loc[(ds2['Cluster'] == m)]
        cluster1 = Label.loc[Label['Clust'] ==m ]
        
        interarea = []
        
        c = 0
        X1 = []
        Y1 = []
        X12 = []
        Y12 = []
        dist = []
        dist1 = []
        if len(cluster) > 0:
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
        else:
            cornerx = 0
            cornerx1 = 0
            cornery = 0
            cornery1 = 0
            # print(cornerx, cornery)
            # print(cornerx1, cornery1)
            
            
    
        if len(cluster1) > 0:
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
        else:
            cornerx2 = 0
            cornerx12 = 0
            cornery2 = 0
            cornery12 = 0
                    # print(cornerx2, cornery2)
                    # print(cornerx12, cornery12)
            
            
        dx = min(cornerx1, cornerx12) - max(cornerx, cornerx2)
        dy = min(cornery1, cornery12) - max(cornery, cornery2)
        if (dx>=0) and (dy>=0):
            interarea.append(dx*dy)
        else:
            interarea.append(0.0)
    
            
        area = (cornerx1 - cornerx)*(cornery1-cornery)
        area1 = (cornerx12 - cornerx2)*(cornery12-cornery2)
        union = (area+area1)-interarea[c]
        if union > 0:
            iou = (interarea[c]/union)
            IOU.append(iou)
        else:
            IOU.append(0.0)     
      
    # 3D plot
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
    for c in range(len(unique)):
        cluster = ds2.loc[(ds2['Cluster'] == c)]
        if len(cluster) > 0:
            R = cluster['R'].tolist()
            R1.append(R)
            G = cluster['G'].tolist()
            G1.append(G)
            B = cluster['B'].tolist()
            B1.append(B)
            e = 0
            dist = []
            while e in range(len(cluster)):
                Xdist = R[e] - Centers[c][0]
                Ydist = G[e] - Centers[c][1]
                Zdist = B[e] - Centers[c][2]
                part1 = math.sqrt((Xdist**2)+(Ydist**2))
                part2 = math.sqrt((part1**2)+(Zdist**2))
                dist.append(part2)
                e = e+1
            maxd = max(dist)
            max_dist.append(maxd)
            ax.scatter(R, G, B, color = colors[c])

    ### Other Metrics
    shs = metrics.silhouette_score(IRpix, labelsir, metric='euclidean')
    chi = metrics.calinski_harabasz_score(IRpix, labelsir)
    dbi = davies_bouldin_score(X, labels)
    
    SHS.append(shs)
    CHI.append(chi)
    DBI.append(dbi)
    GStrain.append(GS)
    MSIOU.append(sum(IOU)/len(IOU))
    Accuracytrain.append(Accuracy)    
    
    print('Bandwidth:', bandwidth)
    print("Number of estimated clusters : %d" % n_clusters_)
    print('Weighted Variance:', WVn)
    print('Moran\'s I:', moransn)
    print('Global Score:', GS)
    print('Mean-Shift IOU:', sum(IOU)/len(IOU))
    print('Accuracy:', Accuracy)
    
results = pd.DataFrame(data = GStrain, columns = (['GS']))
results['MSIOU'] = MSIOU
results['Accuracy'] = Accuracytrain
results['Time Taken'] = time_taken
results['Morans I'] = Moranstest
results['WV'] = WVtest
# results['N clusts'] = N_clusts
results['Silhouette Score'] = SHS
results['Calinski-Harabasz Index'] = CHI
results['Davies-Bouldin Index'] = DBI 
pd.DataFrame(results).to_csv(r'C:\Users\jorda\OneDrive\Documents\Thesis1.csv')
