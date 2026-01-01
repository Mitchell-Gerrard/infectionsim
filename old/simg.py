import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
from numba import njit

import time as ti
sns.set()
pal = sns.color_palette("Set1")
plt.rcParams['animation.ffmpeg_path']=r'/usr/bin/ffmpeg'
plt.rcParams['animation.convert_path']=r'/lib/ImageMagick-7.0.10/'
'''
#the population
nod=250
noot=nod
#infection radius
radius=3

#size of area of population

ysize,xsize=100,100

colour=[]
time=[]

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
colour.append(10)
for i in range(1,nod):
    colour.append(0)'''
cm=plt.get_cmap('plasma')
fig = plt.figure()
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
la=ax2.stackplot([],[],[],[], labels=['infected','suseptable','recovered'],colors=pal)
ax2.legend(loc='upper left')
ln = ax1.scatter([], []) 


''',xdata,ydata,xsize,ysize,colour,radius,time,infe,susept,recovered'''

      
def flumps(frames,xdata,ydata,xsize,ysize,colour,radius,time,infe,susept,recovered,L):
    print(frames)
    xdata=np.array(xdata)
    ydata=np.array(ydata)
    for i in range(len(xdata)):
        x=xdata[i]+np.random.randint(-50,50)
        y=ydata[i]+np.random.randint(-50,50)
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
                    if colour[j]>=100 or colour[i]>=100:
                        continue
                    
                    if colour[j]==0 and colour[i]>0:
                        if ana==1:
                            colour[j]=1
                    if colour[i]==0 and colour[j]>0:
                        if ana==1:
                            colour[i]=1
        if colour[i]>0:
            if colour[i]<100:
                colour[i]=colour[i]+1
    colour=colour/100
    ln.set_color(cm(colour))
    colour=colour*100
    data=np.array([xdata,ydata])
    data=data.transpose()
    ln.set_offsets(data)
    su=0
    infec=0
    recov=0
    for i in range(len(colour)):
        if colour[i]==0:
            su+=1
        if colour[i]>0 and colour[i]<100:
            infec+=1
        if colour[i]==100:
            recov+=1
    
    
    time.append(frames/6)
    infe.append(infec)
    susept.append(su)
    recovered.append(recov)
    #data=np.array([time,infe,susept,recovered])
    
    #data=data.transpose()
    #la.set_offsets(data)
    ax2.stackplot(time,infe,susept,recovered, labels=['infected','suseptable','recovered'],colors=pal)
    
    
    
    return ln,la

def dots(xsize,ysize):
        x=np.random.randint(-xsize,xsize)
        y=np.random.randint(-ysize,ysize)
        return x,y    
def init():
    
    return ln,la



def animations(name):
    start=ti.time()
    nod=np.random.randint(250,500)
    xdata,ydata=np.zeros(nod),np.zeros(nod)
    ysize,xsize=(nod),(nod)
    for i in range(nod):
        x,y=dots(xsize,ysize)
        xdata[i]=x
        ydata[i]=y
  

    radius=np.random.randint(1,15)

    

    

    colour=np.zeros(nod)
    time=[]
    
    susept=[]
    infe=[]
    recovered=[]
    ax1.set_xlim(-xsize,xsize)
    ax1.set_ylim(-ysize,ysize)
    L=0
    k=10
    for i in range (k):
        colour[i]=1
    for i in range(k,nod):
        colour[i]=0
    print(nod,radius)
    print(name)
    plt.title(f'{nod},{radius}')
    animat = FuncAnimation(fig, flumps, frames=np.arange(0,500,1),
                         init_func=init,repeat=False,fargs=(xdata,ydata,xsize,ysize,colour,radius,time,infe,susept,recovered,L))
   
    
    #plt.show()
    #Writer = ImageMagickFileWriter()
    name=f'n{nod}r{radius}size{xsize}{name}'
    animat.save(name,writer='imagegick',dpi=100)
    stop=ti.time()-start
   
    print(stop)
    return(name, recovered,infe,susept)
name=[]
k=100
for i in range(k):
    name.append(f'sim{i}.gif')
susepta=[]
infect=[]
recovere=[]
nade=[]
print(name)
if __name__ == '__main__':
    
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
         results = [executor.submit(animations,names) for names in name]
         for f in concurrent.futures.as_completed(results):
            name, recovered,infe,susept=f.result()
            print(name)
            nade.append(name)
            susepta.append(susept)
            infect.append(infe)
            recovere.append(recovered)
            
    print(len(infect))
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(k):
        ax.plot(np.divide(np.arange(len(recovere[i])),10),recovere[i],label=nade[i])
    ax.legend()
    plt.show()
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(k):
        ax.plot(np.divide(np.arange(len(susepta[i])),10),susepta[i],label=nade[i])
    ax.legend()
    plt.show()    
    fig, ax = plt.subplots(figsize=(10, 6))
    for i in range(k):
        ax.plot(np.divide(np.arange(len(infect[i])),10),infect[i],label=nade[i])
    ax.legend()
    plt.show()
