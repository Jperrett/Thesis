import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from itertools import cycle
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from scipy.spatial import ConvexHull, convex_hull_plot_2d
pylab.rcParams['figure.figsize'] = 16, 12

### Loads in the image
image = Image.open('E:/Farmland/farm22.png')
im = Image.open('E:/Farmland/farm22.png')
width, height = im.size
image = np.array(image)
original_shape = image.shape

### Collecting the RGB and location of each pixel.
orig_pixel_map = im.load()
i,j = 0,0 
pixels =[]
locationx=[]
locationy=[]
for i in range(height):
    for j in range(width):
        pixel = orig_pixel_map[i, j]
        #print(i,j)
        #print(f"\n pixel: {pixel}")
        pixels.append(pixel)
        locationx.append(i)
        locationy.append(j)
        j = j+1
    i = i+1
arr = np.array(pixels)

### Flattening the image.
X = np.reshape(image, [-1, 3])
plt.imshow(image)

### Setting the Bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.05, n_samples=500) #0.05
print(bandwidth)

### Carrying out mean shift on the collected pixels.
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(arr)
cluster_centers = ms.cluster_centers_
print(cluster_centers.shape)
labels= ms.labels_

flat = ms.fit(X)
labels1 = ms.labels_
print(labels1.shape)

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

### Plotting the centers of the clusters
## Splitting the collected RGB values.
x = cluster_centers[:,0]
y = cluster_centers[:,1]
z = cluster_centers[:,2]
x1 = arr[:,0]
y1 = arr[:,1]
z1 = arr[:,2]

### Plotting a 2D graph of the clusters.
test = {}
center =[]
plt.figure()
plt.clf()
W = []
V = []
colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
for k, col in zip(range(n_clusters_), colors):
    print(k)
    my_members = labels == k
    cluster_center = cluster_centers[k]
    plt.plot(arr[my_members, 0], arr[my_members, 1], col + '.')
    plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
             markeredgecolor='k', markersize=14)
    ### Splitting the data into the different clusters.
    test[k] = arr[my_members]
    center.append(cluster_center)
    test1 = test[k]
    m =0
    distance ={}
    variance = []
    ### Calculating the Euclidean distance from the centroid and therefore 
    ### the Variance for each point.
    while m in range(len(test1)):
        dist = np.linalg.norm(test1[m]-center[k])
        variance.append(dist**2)
        distance[m] = dist
        m = m+1
        mean_value = sum(distance.values())/len(test1)
        cent = center[k]
    draw_circle = plt.Circle(cent[:2], mean_value,fill=False, color = col )
    plt.gcf().gca().add_artist(draw_circle)
    sigma = sum(variance)/len(test1)
    W.append(sigma)
    ### Calculating the area based on the mean distance.
    Area = 3.14*(mean_value**2)
    V.append(Area)
    
plt.title('Estimated number of clusters: %d' % n_clusters_)
plt.show()

### Calculating the WV score.
s = 0
t = 0
Echo = []
while s in range(len(W)):
    while t in range(len(V)): #Not entirely necessary (remove later)
        delta = W[s]*V[t]
        Echo.append(delta)
        s = s+1
        t = t+1
WV = sum(Echo)/sum(V)
print(WV)
    
### Plotting a 3D graph of the clusters.
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x1,y1,z1, color = 'r', alpha = 0.5)
ax.scatter(x,y,z,c = 'b')
plt.show()

### Code taken from:
    # http://www.chioka.in/meanshift-algorithm-for-the-rest-of-us-python/
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_mean_shift.html#sphx-glr-auto-examples-cluster-plot-mean-shift-py
    # Mostly used for the graphs.
    



