import numpy as np
import random
import argparse
import time
import threading

sum = 0
N = 10000
count=0
#count2=1

def timer(t):
    time.sleep(t)
    print 'time up,the number of calculation is', count, 'the number is', sum/count
    print "press ctrl+c to exit"

    exit(0)

def input(path):
    try:
        f=open(path,'r')
        txt=f.readlines()
        NodeNum=int(txt[0].split()[0])
        print "The number of nodes is",NodeNum
        EdgeNum=int(txt[0].split()[1])
        print "The number of edges is",EdgeNum
        AdjaceMatrix=np.zeros((NodeNum,NodeNum))
        for line in txt[1:len(txt)-1]:
            row=str.split(line)
            FormerNode=int(row[0])
            #print FormerNode
            NextNode=int(row[1])
            #print NextNode
            Probability=float(row[2])
            AdjaceMatrix[FormerNode-1][NextNode-1]=Probability
            #print AdjaceMatrix[FormerNode-1][NextNode-1]

        return AdjaceMatrix, NodeNum
    except IOError:
        print 'Error: file not found'
    finally:
        f.close()

"""This function gets all the neighbors of the seed"""

def find_all_neighbors(seed,AdjaceMat):
    neighbor=[]
    for i in range(0, Nodenum):
        if (AdjaceMat[seed - 1][i] != 0):
            newNeighbor = i + 1
            neighbor.append(newNeighbor)  # get the neighbor list of the seed
            # print neighbor
    return neighbor



"""This function take in one single seed, adjacementMatrix, ActivityState 
and find all the inactive neighbors
and store them into a list, 
also returns a list which stores the probability of the seed to active this neighbor"""
def find_inactive_neighbor(seed,AdjaceMat,ActivityState):
    neighbor=[]
    inactive_neighbor=[]
    prob_to_neighbor=[]# the probability to active this neighbor
    for i in range(0,Nodenum):
        if (AdjaceMat[seed-1][i]!=0):
               newNeighbor=i+1
               neighbor.append(newNeighbor) #get the neighbor list of the seed
    #print neighbor
    for i in range(0,len(neighbor)):
        if (ActivityState[neighbor[i]-1]==0):
            new_inactive_neighbor=neighbor[i]
            inactive_neighbor.append(new_inactive_neighbor) #get the inactive list of the seed
    #print inactive_neighbor
    for i in range(0,len(inactive_neighbor)):
        prob_to_neighbor.append(AdjaceMat[seed-1][inactive_neighbor[i]-1])
   # print prob_to_neighbor
    return inactive_neighbor,prob_to_neighbor #The number of nodes in this list is the real number of the nodes



"""
This function takes in a seed, the AdjacementMatrix, the ActivityState
and first, get the neighbors of the seed
then, get the active ones among these neighbors 
finally, calculate and returns the sum of weights
"""
def find_active_neighbor(seed,AdjaceMat,ActivityState):
    neighbor=[]
    active_neighbor=[]
    sum_of_weight=0
    for i in range(0, Nodenum):
        if (AdjaceMat[i][seed - 1] != 0):
            newNeighbor = i + 1
            neighbor.append(newNeighbor)  # get the neighbor list of the seed
    #print neighbor
    for i in range(0, len(neighbor)):
        if (ActivityState[neighbor[i] - 1] != 0):
            new_active_neighbor = neighbor[i]
            active_neighbor.append(new_active_neighbor)  # get the inactive list of the seed
    for i in range(0,len(active_neighbor)):
        sum_of_weight = sum_of_weight + AdjaceMat[active_neighbor[i] - 1][seed - 1]
    #print  sum_of_weight
    return sum_of_weight
            # print inactive_neighbor

"""
A model of DegreeDiscountIC sample
"""
def DegreeDiscountIC(AdjaceMat,k):
    S=[] #initialize the set of the output
    #dv=np.zeros(Nodenum) #initialize the set of degrees of each node
    dv=np.ones(Nodenum,dtype=np.int16)
    for i in range(0,Nodenum):
        for j in range(0,Nodenum):
               dv[i]=0   #set the dv to 0
    ddv=[Nodenum]
    tv=np.ones(Nodenum,dtype=np.int16)
    Nodelist=np.ones(Nodenum,dtype=np.int16)
    for i in range(0,Nodenum):
        Nodelist[i]=i+1
    #print Nodelist
    for i in range(0,Nodenum):
        for j in range(0,Nodenum):
           if(AdjaceMat[i][j]!=0):
               dv[i]=dv[i]+1
    #print dv  #the set of the degrees of each node
    ddv=dv
   # print ddv
    for i in range(0,k):
        Setu=[]  #The set of nodes which is in V and not in S
        for i in range(0,len(Nodelist)):
            if(Nodelist[i]) in S:
                continue
            else:
                Setu.append(Nodelist[i])#the index of nodes in Setu is the real index, when calculate please -1
       # print Setu
        tempddv=np.ones(len(Setu),dtype=np.int16)
        for i in range(0, len(tempddv)):
            tempddv[i] = 0

        for i in range(0,len(tempddv)):
            tempddv[i]=ddv[Setu[i]-1]
       # print tempddv
        u=np.argmax(tempddv)  #the index of tempddv is the index of the Setu which get the real index
       # print u
        #addu=Setu[tempddv[u]-1]
        addu=Setu[u]
        #print addu
        S.append(addu)
      #  print S
        neighbors_of_u=find_all_neighbors(addu,AdjaceMat)
        u_v_prob = []  # The propagation probability from u to v
        for i in range(0,len(neighbors_of_u)):
            if neighbors_of_u[i] in S:
                neighbors_of_u.remove(neighbors_of_u[i])
        for i in range(0,len(neighbors_of_u)):
            u_v_prob.append(AdjaceMat[u-1][neighbors_of_u[i-1]])
        for i in range(0,len(neighbors_of_u)):
            tv[i-1]=tv[i-1]+1
            ddv[i-1]=dv[i-1]-2*tv[i-1]-(dv[i-1]-tv[i-1])*tv[i-1]*u_v_prob[i]

    print S
    return S




"""
A one IC Sample
"""
def IC(AdjaceMat,SeedSet,Nodenum):

    ActivitySet = SeedSet
    ActivityState = np.zeros(Nodenum)  # Store the states of all the nodes
    for i in ActivitySet:
        temp = i
        ActivityState[temp - 1] = 1.0  # Set the state of the seed to active.
        # print ActivityState
        # find_inactive_neighbor(SeedSet[0],ActivitySet,AdjaceMat)
    count = ActivitySet.__len__()  # print count
    while (ActivitySet.__len__() != 0):
        newActivitySet = []
        for seed in ActivitySet:
            inactive_neighbor, prob_to_neighbor = find_inactive_neighbor(seed,AdjaceMat,ActivityState)
            for i in range(0, len(inactive_neighbor)):
                randomNum = random.random()
                if (randomNum < prob_to_neighbor[i]):
                    ActivityState[inactive_neighbor[i] - 1] = 1.0
                    newActivitySet.append(inactive_neighbor[i])

        count = count + newActivitySet.__len__()
        ActivitySet = newActivitySet

    #print count
    return count
"""
One LT Sample
"""
def LT(AdjaceMat,SeedSet,Nodenum):
    ActivitySet=SeedSet
    ActivityState = np.zeros(Nodenum)  # Store the states of all the nodes
    for i in ActivitySet:
        temp = i
        ActivityState[temp - 1] = 1.0  # Set the state of the seed to active.
        # print ActivityState
        # find_inactive_neighbor(SeedSet[0],ActivitySet,AdjaceMat)
    threshold = np.zeros(Nodenum)
    for i in range(0,len(threshold)):
        threshold[i]=random.random()
    #print threshold
    count = ActivitySet.__len__()
    while(ActivitySet.__len__() != 0):
        newActivitySet=[]
        for seed in ActivitySet:
            inactive_neighbor,prob_to_neighbor=find_inactive_neighbor(seed,AdjaceMat,ActivityState)
            #print inactive_neighbor
            for i in range(0,len(inactive_neighbor)):
                w_total=find_active_neighbor(inactive_neighbor[i],AdjaceMat,ActivityState)
               # print w_total
                temp=inactive_neighbor[i]-1
               # print temp
                w_single=threshold[temp]
                if(w_total>=w_single):
                    ActivityState[inactive_neighbor[i] - 1]=1.0
                    newActivitySet.append(inactive_neighbor[i])
        count = count + newActivitySet.__len__()
        ActivitySet=newActivitySet
    #print count
    return count




if __name__=='__main__':

    #input('network.txt')
  #  AdjaceMat,Nodenum = input('network.txt')
    #print AdjaceMat,Nodenum
   # SeedSet = DegreeDiscountIC(AdjaceMat, 4)
   # SeedSet=inputseed('seeds.txt')

    #IC(AdjaceMat,SeedSet)

    AP=argparse.ArgumentParser()
    AP.add_argument('-i',help='The path of the social network file', dest='path')
    AP.add_argument('-k',type=int,help='The path of the seed set file',dest='size')
    AP.add_argument('-m',help='The diffusion model',dest='model')
    AP.add_argument('-b',type=int,help='0 or 1. set to 0, then the termination condition is not changed, to 1 then the '
                              'maximal time budget specifies the termination condition.',dest='type')
    AP.add_argument('-t',type=float,help='How much time this algorithm can spend on',dest='timeout')
    AP.add_argument('-r',type=int,help='The random seed used in this run',dest='random_seed')

    args=AP.parse_args()
    path=args.path
    size=args.size
    model=args.model
    type=args.type
    #timeout=args.type
    timeout=args.timeout
    random_seed=args.random_seed
   #timer=threading.Timer(10,stop())
   # timer.start()
    AdjaceMat,Nodenum=input(path)
    #SeedSet=inputseed(seedpath)
    SeedSet=DegreeDiscountIC(AdjaceMat,size)
    if type==0:
      timer = threading.Thread(target=timer, args=(timeout ,))
      timer.start()
      if model=='LT' :
        while True:
            oneSample = LT(AdjaceMat, SeedSet,Nodenum)
            sum = sum + oneSample
            count=count+1
        #print "(LT)The average number of spread people is", sum1 / count1
        #timer.join()
      elif model=='IC':
        while True:
            oneSample = IC(AdjaceMat, SeedSet,Nodenum)
            sum = sum + oneSample
            count=count+1
        #print "(IC)The average number of spread people is", sum2 / count2
        #timer.join()
    elif type==1:
        if model == 'LT':
            for i in range(0, N):
                oneSample = LT(AdjaceMat, SeedSet,Nodenum)
                sum = sum + oneSample
            print "(LT)The average number of spread people is", sum / N
        elif model == 'IC':
            for i in range(0, N):
                oneSample = IC(AdjaceMat, SeedSet,Nodenum)
                sum = sum + oneSample
            print "(IC)The average number of spread people is", sum / N
