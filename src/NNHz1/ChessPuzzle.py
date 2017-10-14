import io
import numpy as np
import matplotlib.pyplot as plt

from src.Utils import now
from src.parameters import *
class ChessPuzzle(object):

    def __init__(self,size):
        self.size = size
        self.state = np.zeros((self.size, self.size))
        self.done = False
        self.placedQueens = 0

    def step(self, flat_action):
        action = [flat_action // self.size, flat_action%self.size]
        #print("Triyng move (%s,%s)"%(action[0],action[1]))
        if(self.state[action[0],action[1]]>0):
            #if here, we've made the wrong move. Game Over!!!
            self.done = True

            return self.state.flatten(), -self.state[action[0],action[1]], self.done, {},self.placedQueens
        #check diagonal dx
        b1 = action[1]-action[0]
        if(b1>=0):
            x1 = 0
            y1 = b1
        if(b1<0):
            x1=-b1
            y1= 0
        #check diagonal sx
        b2 = action[0]+action[1]
        if(b2<=self.size-1):
            x2 = b2
            y2 = 0
            limit2 = b2+1
        else:
            x2=self.size-1
            y2=b2-(self.size-1)
            limit2=self.size - (b2-self.size) -1

        for i in range(self.size):
            if(self.state[action[0],action[1]]>0):
                wrongMove = True
                break;
            #check row
            self.markTile(action[0],i,action)
            #check column
            self.markTile(i,action[1],action)
            #check diagonal dx
            if(i<self.size - abs(b1)):
                self.markTile(x1+i,y1+i,action)
            #check diagonal sx
            if(i<limit2):
                self.markTile(x2-i,y2+i,action)

        self.state[action[0],action[1]] = self.size
        self.placedQueens+=1

        # Determine reward
        reward = self.placedQueens
        #print('Good!!! Reward:%s Done=%s'%(reward,self.done))
        return self.state.flatten(), reward, self.done, {},self.placedQueens

    def markTile(self,x,y,action):
        #jump queen position
        if(not(action[0]==x and action[1]==y)):
            self.state[x,y] += 1.0

    def reset(self):
        self.state = np.zeros((self.size, self.size))
        self.placedQueens = 0
        self.done = False
        return self.state.flatten()

    def render(self):
        plt.imshow(self.state, interpolation='nearest')
        plt.title(now('%Y%m%d%H%M%S%f'))
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        return buf
