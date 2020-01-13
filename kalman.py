"""
kalman filter assignment from AI_for_Robotics course
this solution is not based on the given code in the course but it applies the 
same equations to solve the multidimension kalman filter challange
"""
import numpy as np 
#let's define some variables we will need

# We are taking discite measurement so this variable will be constant
delta_t = 0.1  # just an assumption
"""inital_state vector, we assumed that we start from co-ordinates(4,12) and 
velocity is 0 in the x and y directions"""
current_state = np.array([4, 12, 0, 0]) # initial values
x_pos_error = 0
y_pos_error = 0
x_vel_error = 1000
y_vel_error = 1000

x_pos_measure_error = 0
y_pos_measure_error = 0

U = 0   #external force

# let's start by defining the matricies
"""
State vector. we have four variables or states we want to estimate the position
and velocity at X-axis and at Y-axis. so the variables are 
x: position at X-axis
y: position at Y-axis
x_vel: velocity in X-axis
y_vel: velocity in Y-axis
The position in the X-axis is known using value of variables x and x_vel using 
this formula new_position = old_position + x_vel * delta_time
instead of doing this we can reprsent this relation in matrix form
"""
# State transmission matrix in the course is noted as F matrix
A = np.array([
            [1, 0, delta_t, 0], # position in X-axis
            [0, 1, 0, delta_t], # position in Y-axis
            [0, 0, 1, 0],       #velocity in X-axis
            [0, 0, 0, 1]        #velocity in Y-axis
            ])
"""
Input transmission matrix B. trasnform the input to the system(if we have 
external force affecting system) for our case U=0
so the values of B is not important, so can ignore it. this video explain how to 
design it  https://www.youtube.com/watch?v=NbRrLv_vX_U
"""
#B = np.zeros((4,1))

"""
Process covariance matix.
Defines the variance in the process and how they are related to each other
will be a 4x4 matrix. 
initial uncertainty: 0 for positions x and y, 1000 for the two velocities
"""
pk = np.array([
            [x_pos_error, 0, 0, 0],
            [0, y_pos_error, 0, 0],
            [0, 0, x_vel_error, 0],
            [0, 0, 0, y_vel_error]
             ])
"""
Measurement covariance matrix.
Defines covariance(same as variance but in multidimension we call it covariance)
of the variables you measure and how they affect each other. 
in our case: use 2x2 matrix with 0.1 as main diagonal
"""
R = np.diag([0.1, 0.1, 0.1, 0.1])

"""transmission matrix"""
H = np.identity(4)
#H = np.array([1,0,0,0],[0,1,0,0])
"""
measurement transmission matrix C
"""    
C = np.identity(4)
        
# Measurements are X and Y 
measurements = [[5., 10.], [6., 8.], [7., 6.], [8., 4.], [9., 2.], [10., 0.]]
        
np.seterr(divide='ignore', invalid='ignore') # to ignore division by zero

# Let's apply kalmanfilter it can be applied by repeating those six steps
for i in range(len(measurements)):
    """
    First step detrmine the current state 
    X = A*X + B*U + W, where W is noise in the process
    we have U=0, and will neglect the W
    """ 
    current_state = A.dot(current_state)
    """
    Second step predict the process covariance matrix 
    P(current) = A * P(previous) * A(transpose) + Q.
    where Q is the process noise covariance matrix: which keeps the state 
    covariance matrix from becomming to small or going to zero
    """
    #fi = np.dot(A, pk)
    #print(fi)
    pk = np.dot(A.dot(pk), A.transpose())
    #print("pk ",pk)
    # third step
    # calculate kalman gain
    # there is error because of dividing by zero 
    # suggested solution 
    #https://stackoverflow.com/questions/17514377/divide-by-arrays-containing-zeros-python
    k = np.divide(pk.dot(H.transpose()), (H.dot(pk).dot(H.transpose())+R))
    #print(k)
    # fourth step
    # new measurement value
    # Y = C*Y + Z
    Y = np.array([measurements[i][0], measurements[i][0], 0, 0])
    # fifth step
    #calculate the new state
    # assign it to initial state to be able to use it in the new iteration
    current_state = current_state + k.dot(Y - H.dot(current_state))
    # 6step
    # update the process covariance matrix
    pk = (np.identity(4) - k.dot(H)).dot(pk)


print(current_state)
# tutorials 
# https://www.youtube.com/playlist?list=PLX2gX-ftPVXU3oUFNATxGXY90AULiqnWT