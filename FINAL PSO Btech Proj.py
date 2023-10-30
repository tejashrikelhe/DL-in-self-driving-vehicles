import random as rd
#import numpy as np
import matplotlib.pyplot as plt
#import math as m
import time


# In[70]:


v=float(input("cur vel of car from speedometeer in m/s = "))
d=float(input("curr dist btwn car n obj in meteres = "))
ttc=d/v
print("ttc=",ttc,"sec")


# In[71]:


W = 0.5
c1 = 0.8
c2 = 0.9

# In[82]:


n_iterations = 100
n_particles = 10
#target_error = 1e-6


# In[83]:




#INITILAISATION

p=[]
pvel=[]
pf=[]
pbest=[]
new_v=[]
new_p=[]
new_pf=[]
gbest_arr=[]
gpos_arr=[]
gvel_arr=[]
pbest_comp=[]
pf_comp=[]

def particle_position():
    y=rd.uniform(0, v)
    return y

def particle_vel():
    pv=rd.random()
    return pv


# INITILIZATION END

# ITERATION BEGINS for new vel

def fitness_vector(y):
    f=v-y
    return f
    
def pfi():
    for i in range(n_particles):
        pf.append(fitness_vector(p[i])) 
    #print("particle fitness values=", pf)

def gbest_update(pf):
    gbest=pf[0]
    gpar=0
    for g in range(len(pf)):
        if(pf[g]<gbest):
            gbest=pf[g]
            gpar=g
    return gpar


# In[84]:


def pbest_update(i,it):
    if (it==1):
        pbest.append(p[i])
        pbest_comp.append(pbest[i])
        pf_comp.append(pf[i])
    else:
        if (pf[i] < pf_comp[i]):
            pbest[i]=p[i]
            pbest_comp[i]=pbest[i]
            pf_comp[i]=pf[i]
        else:
            pbest[i]=pbest_comp[i]
    
    #print("for iteration =", it,"\n pbest for particle =",i, "        pbest=", pbest[i])
    #print("pbest_comp=",pbest_comp)
    #print("pf_comp=",pf_comp)
    
    

def update_particle(i,pos,vel,gpos,pbest):
    new_vel= W * (vel) + (c1*rd.random()*(pbest-pos)) + (c2*rd.random()*(gpos-pos))
    pvel[i]=new_vel
    new_pos= float(pos+new_vel)
    if (new_pos>=0 and new_pos<=v):
        p[i]=new_pos
    else:
        p[i]=pos


# In[85]:



it=1

start = time.time()

while (it <= n_iterations): 
    
    #posn,vel vectors
    if (it==1):
      for i in range(n_particles):
        p.append(particle_position())
        pvel.append(particle_vel())
      #print("Initial Particle positions=",p,"\n Initial Particle velocities=",pvel)
    
    else:
      #print("For iteration :", it)
      #print("Particle positions=",p,"\n Particle velocities=",pvel)
      print()
        
    #fittness vector 
    pfi()
    
    #gbest
    if(it==1):
        gpar=gbest_update(pf)
        gbest=pf[gpar]
        gpos_arr.append(p[gpar])
        gvel_arr.append(pvel[gpar])
        gbest_arr.append(gbest)
        gpos= gpos_arr[it-1]
        gvel= gvel_arr[it-1]
    else:
        gpar=gbest_update(pf)
        gbest=pf[gpar]
        gbest_arr.append(gbest)
        if( gbest_arr[it-1] < gbest_arr[it-2]):
            gbest= gbest_arr[it-1]
            gpos_arr.append(p[gpar])
            gvel_arr.append(pvel[gpar])
            gpos= gpos_arr[it-1]
            gvel= gvel_arr[it-1]
        else:
            gbest= gbest_arr[it-2]
            gpos=gpos_arr[it-2]
            gvel=gvel_arr[it-2]
            gpos_arr.append(gpos_arr[it-2])
            gvel_arr.append(gvel_arr[it-2])
        
    #print("global best fittness value=",gbest)
    #print("\n gbest particle number=",gpar+1,"\n gbest particle position=",gpos,"\n gbest partilce velocity=",gvel)
    #print("gbest_arr=",gbest_arr, "\n gpos_arr=", gpos_arr, "\n gvel_arr=", gvel_arr)
    
    
    # pbest
    for i in range(n_particles):
        pbest_update(i,it)
    
    # update particles
    for i in range(n_particles):
        update_particle(i,p[i],pvel[i],gpos,pbest[i])
        
    
    #print("updated P=",p, "\n updated pvel=",pvel)
    pf.clear()
    
    #print("\n *********************************************************************")
    
    if (it==n_iterations):
        print("Final solution .i.e. New velocity required for the car= ", gbest)
        
    #for k in range(n_particles):
       # plt.plot(p[k], 'ro')
    #plt.plot(gbest_arr[it-1], 'bo')
    #plt.show()
    
    it=it+1
end = time.time()
td=end-start
print('Time taken for PSO execution: ', td)

    


# In[86]:


vn=gbest
dn=d-(v*td)


# In[87]:


#  acc calc

pacc=[]
pvelacc=[]
pfacc=[]
pbestacc=[]
new_vacc=[]
new_pacc=[]
new_pfacc=[]
gbest_arracc=[]
gpos_arracc=[]
gvel_arracc=[]
pbest_compacc=[]
pf_compacc=[]

tmax=ttc-(td+1)
#1= rection time and other delays

amax=9.80665
#amax=3
tmin= -1 * ((vn-v)/amax)

#print("tmax=",tmax,"tmin=",tmin)

if(tmin < tmax):
    tdo=tmin
    tu=tmax
else:
    tdo=0
    tu=tmax
    
#print("td=",tdo, "tu=",tu)

def particle_position_acc():
    #z=time
    z=rd.uniform(tdo,tu)
    #if (z <= tmax):
     #   return z
    #else:
    #    return -1
    return z

        
def fitness_vector_acc(z):
    a= -1 * ((vn-v)/(z))
    #if(a <= amax):
     #   return a
    #else:
     #   return -1
    return a

def pfi_acc():
    for i in range(n_particles):
        pfacc.append(fitness_vector_acc(pacc[i])) 
    #print("acc particle fitness values=", pfacc)

def gbest_update_acc(pfacc):
    gbestacc=pfacc[0]
    gparacc=0
    for g in range(len(pfacc)):
        if(pfacc[g]<gbestacc):
            gbestacc=pfacc[g]
            gparacc=g
    return gparacc

def pbest_update_acc(i,it):
    if (it==1):
        pbestacc.append(pacc[i])
        pbest_compacc.append(pbestacc[i])
        pf_compacc.append(pfacc[i])
    else:
        #have to change this cpmarison as deaccelaration value possible cant exceed a certain value 
        if (pfacc[i] < pf_compacc[i]):
            pbestacc[i]=pacc[i]
            pbest_compacc[i]=pbestacc[i]
            pf_compacc[i]=pfacc[i]
        else:
            pbestacc[i]=pbest_compacc[i]
    
    #print("for iteration =", it,"\n pbest for particle =",i, "        pbest=", pbest[i])
    #print("pbest_comp=",pbest_comp)
    #print("pf_comp=",pf_comp)
    
    

def update_particle_acc(i,pos,vel,gpos,pbest):
    new_vel= W * (vel) + (c1*rd.random()*(pbest-pos)) + (c2*rd.random()*(gpos-pos))
    pvelacc[i]=new_vel
    new_pos= float(pos+new_vel)
    if (new_pos>=0 and new_pos<=tu):
        pacc[i]=new_pos
    else:
        pacc[i]=pos
        
        


# In[88]:



it=1
flag=0
start = time.time()

while (it <= n_iterations): 
    
    #posn,vel vectors
    if (it==1):
        for i in range(n_particles):
            #r= -1
            #while (r<0):
             #   r= particle_position_acc()
            pacc.append(particle_position_acc())
            pvelacc.append(particle_vel())
        #print("acc Initial Particle positions=",pacc,"\n acc Initial Particle velocities=",pvelacc)
    
    else:
      #print("For Acc calc iteration :", it)
      #print("Particle positions=",pacc,"\n Particle velocities=",pvelacc)
      print()
        
    #fittness vector 
    pfi_acc()
    
    #gbest
    if(it==1):
        gparacc=gbest_update_acc(pfacc)
        gbestacc=pfacc[gparacc]
        gpos_arracc.append(pacc[gparacc])
        gvel_arracc.append(pvelacc[gparacc])
        gbest_arracc.append(gbestacc)
        gposacc= gpos_arracc[it-1]
        gvelacc= gvel_arracc[it-1]
    else:
        gparacc=gbest_update_acc(pfacc)
        gbestacc=pfacc[gparacc]
        gbest_arracc.append(gbestacc)
        if( gbest_arracc[it-1] < gbest_arracc[it-2]):
            gbestacc= gbest_arracc[it-1]
            gpos_arracc.append(pacc[gparacc])
            gvel_arracc.append(pvelacc[gparacc])
            gposacc= gpos_arracc[it-1]
            gvelacc= gvel_arracc[it-1]
        else:
            gbestacc= gbest_arracc[it-2]
            gposacc=gpos_arracc[it-2]
            gvelacc=gvel_arracc[it-2]
            gpos_arracc.append(gpos_arracc[it-2])
            gvel_arracc.append(gvel_arracc[it-2])
        
    if(gbestacc <= (amax+0.1) and gbestacc >= 0 ):
        #print("Deacc required=",gbestacc,"m/s2"," in",gposacc,"seconds.")
        print()
    else:
        #print("calcultaed acc=",gbestacc,"m/s2","  which is not practically possible to achieve.")
        flag=1
    #print("\n gbest particle number=",gpar+1,"\n gbest particle position=",gpos,"\n gbest partilce velocity=",gvel)
    #print("gbest_arr=",gbest_arr, "\n gpos_arr=", gpos_arr, "\n gvel_arr=", gvel_arr)
    
    
    # pbest
    for i in range(n_particles):
        pbest_update_acc(i,it)
    
    # update particles
    for i in range(n_particles):
        update_particle_acc(i,pacc[i],pvelacc[i],gposacc,pbestacc[i])
        
    
    #print("updated P=",p, "\n updated pvel=",pvel)
    pfacc.clear()
    
    #print("\n *********************************************************************")
    
    if (it==n_iterations):
        if(gbestacc <= (amax+0.1) and gbestacc >= 0):
            print("Final solution .i.e. negative acceleration required for the car=",gbestacc,"m/s2"," in",gposacc,"seconds is POSSIBLE.")
        else:
            flag=1
            print("Final solution .i.e. negative acceleration required for the car= ", gbestacc,"m/s2 in ",gposacc,"seconds which is practically IMPOSSIBLE.")
        
    #for k in range(n_particles):
        #plt.plot(pacc[k], 'ro')
    #plt.plot(gbest_arracc[it-1], 'bo')
    #plt.show()
    
    it=it+1
    

end = time.time()
#e= time.time()
ta=end-start
print('Time taken for Acc PSO execution: ', ta)
#tf=s-e
time_best=gposacc


# In[89]:


a=gbestacc
if(flag==0):
    print("Calculated New velocity of the car=",vn,"m/s")
    print("negative acceleration required for the vehicle to reach the new velocity is= ",a," m/s2")
    print("This acceleration value is physically POSSIBLE to acheive.")
    print("This acceleration is achieved in time=", time_best, " s")
    rdist=(-1 * ((vn*vn)-(v*v))) / (2*a)
    print("stopping dist= ", rdist, "m")
    if(vn==0):
        ttc_new = float('inf')
    else:
        ttc_new= rdist/vn
    print("new ttc=", ttc_new, "sec")
    ttt=ttc+td+ta+3
    #3=New human drivers are taught to “leave 2 to 3 seconds’ worth of distance” between the car in front of them to provide the time and space to react.  
    if (ttc_new > ttt):
        print("collision can be avoided")
    else:
        print("collision cannot be avoided")
    
else:
    print("Calculated New velocity of the car=",vn,"m/s")
    print("Negative acceleration required for the vehicle to reach this new velocity= ",a," m/s2")
    print("This acceleration value is physically IMPOSSIBLE to acheive.")
    print("This acceleration is achieved in time =", time_best, " s.")
    rdist=(-1 * ((vn*vn)-(v*v))) / (2*a)
    print("stopping dist= ", rdist, "m")
    if(vn==0):
        ttc_new = float('inf')
    else:
        ttc_new= rdist/vn
    print("new ttc=", ttc_new, "sec")
    ttt=ttc+td+ta+3
    #3=New human drivers are taught to “leave 2 to 3 seconds’ worth of distance” between the car in front of them to provide the time and space to react.  
    #if (ttc_new > ttt):
       # print("collision can be avoided")
    #else:
    print("collision cannot be avoided")
    
 
    


# In[90]:


fig=plt.figure(figsize=(10,6))
# setting x and y axis range
plt.plot(gbest_arr, color='green',marker='o', markerfacecolor='red', markersize=5)
# naming the x axis
plt.ylabel('Value of New velocity')
# naming the y axis
plt.xlabel('Iteration Number')
  
# giving a title to my graph
plt.title('Optimized Global best Values for calculated New velocity ')
  
# function to show the plot
plt.show()


# In[91]:


fig=plt.figure(figsize=(10,6))

plt.plot(gbest_arracc, color='green',marker='o', markerfacecolor='blue', markersize=5)
# naming the x axis
plt.ylabel('negative acceleration')
# naming the y axis
plt.xlabel('Iteration Number')
  
# giving a title to my graph
plt.title('Optimized Global best Values for Negative Acceleration ')
  
# function to show the plot


plt.show()


# In[41]:


vel=[v,vn]
time=[0,time_best]
fig=plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.plot(time,vel, color='green',marker='o', markerfacecolor='pink', markersize=10)
plt.annotate("initial velocity",(0,v),(0.3,v))
plt.annotate("final velocity",(time_best,vn+1),)
# naming the x axis
plt.xlabel('Time(s)')
# naming the y axis
plt.ylabel('Velocity (m/s)')
# giving a title to my graph
plt.title(' Final velocity VS Initial velocity ')

for xy in zip(time, vel):                                       # <--
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--

plt.show()


# In[42]:


dis=[d,rdist]
time=[0,time_best]
fig=plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.annotate("Initial Distance from Object",(0,d),(0.3,d))
plt.annotate("Final Distance from Object",(time_best,rdist+2),)
plt.plot(time,dis, color='green',marker='o', markerfacecolor='pink', markersize=10)
# naming the x axis
plt.xlabel('Time (sec)')
# naming the y axis
plt.ylabel('Distance (m)')
# giving a title to my graph
plt.title('Distance between the self-driving car and the detected object')
for xy in zip(time, dis):                                       # <--
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--
plt.show()


# In[43]:


ttca=[ttc,float("inf")]
dists= [d,rdist]
fig=plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.annotate("Initial Time to collision",(0,d),(0.25,d))
plt.annotate("Final Time to collision",(float("inf"),rdist+3),)
plt.plot(ttca,dists, color='green',marker='o', markerfacecolor='pink', markersize=10)
# naming the x axis
plt.xlabel('Time to collision')
# naming the y axis
plt.ylabel('Distance from object')
plt.title('Time to collision VS Distance from object')
for xy in zip(ttca, dists):                                       # <--
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--
plt.show()


# In[44]:


acc=[0,-a]
time=[0,time_best]
fig=plt.figure(figsize=(10,6))
ax = fig.add_subplot(111)
plt.annotate("Initial Negative Accelaration",(0,0),(0.2,0))
plt.annotate("Final Negative Accelaration",(time_best-0.4,-a+0.5),)
plt.plot(time,acc, color='green',marker='o', markerfacecolor='pink', markersize=10)
# naming the x axis
plt.xlabel('Time')
# naming the y axis
plt.ylabel('Accelaration')
plt.title('Negative Accelaration VS Time')
for xy in zip(time, acc):                                       # <--
    ax.annotate('(%s, %s)' % xy, xy=xy, textcoords='data') # <--
plt.show()


# In[ ]:



# In[ ]:





# In[ ]:




