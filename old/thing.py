import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation

sns.set()
pal = sns.color_palette("Set2")
#the population
n=1
nod=250*n
#infection radius
radius=7.5
#size of area of population
k=10*n
ysize,xsize=k*50,k*50
colour=[]
time=[]
xdata,ydata=np.zeros(nod),np.zeros(nod)
susept=[]
infe=[]
recovered=[]
def dots():
    x=np.random.randint(-xsize,xsize)
    y=np.random.randint(-ysize,ysize)
    return x,y
for i in range(nod):
    x,y=dots()
    xdata[i]=x
    ydata[i]=y
colour.append(1)
colour.append(1)
colour.append(1)
for i in range(3,nod):
    colour.append(0)
cm=plt.get_cmap('plasma')
fig = plt.figure()
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
ax1.set_ylim(-xsize-10,xsize+10)
ax1.set_xlim(-xsize-10,xsize+10)
la=ax2.stackplot(time,infe,susept,recovered, labels=['infected','suseptable','recovered'],colors=pal)
ax2.legend(loc='upper left')
ln = ax1.scatter(xdata, ydata) 

def flumps(frames):
    for i in range(len(xdata)):
        x=xdata[i]+np.random.uniform(-6*k,6*k)
        y=ydata[i]+np.random.uniform(-6*k,6*k)
        if x>=xsize:
                
                xdata[i]=xsize-np.random.randint(0,5)
               
        if x<=-xsize: 
            
            xdata[i]=-xsize+np.random.randint(0,5)   
            
        else: 
            xdata[i]=x
        
        if y>=ysize :
           
            ydata[i]=ysize-np.random.randint(0,5)
            
        if y<=-ysize:
            
            ydata[i]=-ysize+np.random.randint(0,5)
        else:
            
            ydata[i]=y
    
    for i in range(len(xdata)):
        c,d=xdata[i],ydata[i]
        lem=np.array([c,d])
        for j in range(len(ydata)):
            if i!=j:
                lomy=np.array([xdata[j],ydata[j]])
                distance = np.linalg.norm(lem - lomy)
                if distance<=radius:
                    #infection probability
                    ana=np.random.randint(0,2)
                    if colour[j]==0 and colour[i]==0:
                        continue
                    if colour[j]>=300 or colour[i]>=300:
                        continue
                    
                    if colour[j]==0 and colour[i]>0:
                        if ana==1:
                            colour[j]=1
                    if colour[i]==0 and colour[j]>0:
                        if ana==1:
                            colour[i]=1
        ana=np.random.randint(0,2000)
        if colour[i]>0:
            if colour[i]<300:
                colour[i]=colour[i]+1
            if colour[i]==300:
                if ana==1:
                    colour[i]=1
    ln.set_color(cm(colour))
    data=np.array([xdata,ydata])
    data=data.transpose()
    ln.set_offsets(data)
    su=0
    infec=0
    recov=0
    for i in range(len(colour)):
        if colour[i]==0:
            su+=1
        if colour[i]>0 and colour[i]<300:
            infec+=1
        if colour[i]==300:
            recov+=1
    
    print(frames)
    time.append(frames/20)
    infe.append(infec)
    susept.append(su)
    recovered.append(recov)
    #data=np.array([time,infe,susept,recovered])
    
    #data=data.transpose()
    #la.set_offsets(data)
    ax2.stackplot(time,infe,susept,recovered, labels=['infected','suseptable','recovered'],colors=pal)
    
    
    return ln,la
    
def init():
    
    return ln,la





ani = FuncAnimation(fig, flumps, frames=np.arange(0,3000,1),
                     init_func=init)
plt.show()