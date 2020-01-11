"""AI for robotics Assignment1 Localization"""
import numpy as np

def move(p, motion):
    """set the propablitiy after motion of robot"""
    #create new 2d list to not overwirte the old 
    q = np.zeros((len(p),len(p[0])))
    for i in range(len(p)):
        for j in range(len(p[0])):
            # assign probaility to each cell= 
            # probability of moving + probability of being in same palce
            # previous cell * p_move + same_cell * (1-p_move)
            q[i][j] = p[i-motion[0]][j-motion[1]] * p_move + p[i][j] * (1-p_move)
    
    return q

def sense(p, measurement):
    """set probabilities after sensing"""
    for i in range(len(colors)):
        for j in range(len(colors[0])):
            correct = (colors[i][j] == measurement)
            #print(correct)
            p[i][j] = p[i][j] * (correct * sensor_right + (1-correct)*(1-sensor_right))
    
    # Normalize    
    return p/np.sum(p)
    
def localize(colors,measurements,motions,sensor_right,p_move):
    # initializes p to a uniform distribution over a grid of the same dimensions as colors
    pinit = 1.0 / float(len(colors)) / float(len(colors[0]))
    p = [[pinit for row in range(len(colors[0]))] for col in range(len(colors))]
    #print("the matrix",p)
    for i in range(len(motions)):
        p = move(p, motions[i])
        p = sense(p, measurements[i])
    
    return p
    
def show(p):
    """display the array"""
    print(p)


colors = [['G', 'G', 'G'],
          ['G', 'R', 'R'],
          ['G', 'G', 'G']]
measurements = ['R', 'R']
motions = [[0,0], [0,1]]
sensor_right = 1.0
p_move = 0.5
p = localize(colors,measurements,motions,sensor_right,p_move)

show(p)