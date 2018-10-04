import pdb, os, sys, glob, cv2
import numpy as np
import random
from matplotlib import pyplot as plt

epochs = 750
lr = 0.001
num_show = 25
################################ MAKE FAKE DATA ############################
# Define x and y range
range_ = [-50, 50]
deviation_range = 10

# Make fake data using an equation of a line
m_fake = random.randint(-100, 100)
m_fake /= 100.

c_fake = random.randint(-10, 10)

#print('m_fake and c_fake are: ' , m_fake , c_fake)

# Define the number of data points
num_points = 50

# Make fake data
x_list = []
y_list = []

for i in range(num_points):
    x = random.randint(range_[0] , range_[1])
    y = int(m_fake*x + c_fake)
    
    deviation_x = random.randint(-deviation_range , deviation_range)
    deviation_y = random.randint(-deviation_range , deviation_range)
    
    x_list.append(x + deviation_x)
    y_list.append(y + deviation_y)

#print('x_list: ' , x_list)
#print('y_list: ' , y_list)

x_min, y_min, x_max, y_max = min(x_list) , min(y_list) , max(x_list) , max(y_list)


#plt.plot([0,100],[0,200] , color = 'green')#, 'k-')
##############################################################################


# Initialize m and c

m = random.randint(-100 , 100)/100.
c = random.randint(-100 , 100)

print('INITIALIZED: ' , m , c)

#line_coord1 = [x_min , x_max]
#line_coord2 = [x_min*m + c , m*x_max + c]
#print(m,c,line_coord1,line_coord2)
# draw diagonal line from (70, 90) to (90, 200)
#plt.plot(line_coord1, line_coord2,color = 'red')#, 'k-')

#plt.show()
#exit()

def save_plt(m,c,epoch_id):
    
    
    # Plot
    plt.scatter(x_list, y_list, color = 'cyan')#, s=area, c=colors, alpha=0.5)
    plt.title('fake data')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    #plt.show()

    x_temp = x_max - x_min
    line_coord1 = [x_min , x_max ]
    line_coord2 = [x_min*m_fake + c_fake, m_fake*x_max + c_fake]
    plt.plot(line_coord1, line_coord2 , color = 'green')#, 'k-')    
    
    line_coord1 = [x_min , x_max]
    line_coord2 = [x_min*m + c , m*x_max + c]
    plt.plot(line_coord1, line_coord2 , color = 'pink')#, 'k-')
    plt.savefig('result/res{}.png'.format(epoch_id))
    #plt.show()    

def get_sum_squared_distance(m,c):
    sum_squared_distance = 0
    for x,y in zip(x_list,y_list):
        # Calculate y_
        y_ = m*x + c
        sum_squared_distance += (y - y_)**2
    return (sum_squared_distance)/float(2*num_points)

def update_parameters(m,c):
    
    dm = 0
    for x,y in zip(x_list , y_list):
        #pdb.set_trace()
        dm += 1/(float(2*num_points))*(m*x + c - y)*x
        #print('dm:',dm)
        #print('--------')
    #dm = dm/float(2*num_points) 
     
    dc = 0
    for x,y in zip(x_list , y_list):
        dc += (m*x + c - y)*1
        #print('dc:' , dc)
        #print('*************')
    dm = dm/float(2*num_points)
    
    #print('dm and dc' , dm,dc)
    #print('before updating',m,c)
    #pdb.set_trace()
    # update parameters
    m -= lr*dm
    c -= lr*dc
    #print('after updating',m,c)
    #pdb.set_trace()
    return m , c
    

# Iterate for specified number of epochs
for i in range(epochs):
    #print('Epoch #', i)
    
    m_current = m
    c_current = c
    
    #print(m_current , c_current)
    loss = get_sum_squared_distance(m_current , c_current)

    # Update parameters
    # TODO
    m,c = update_parameters(m_current , c_current)
    #print(m,c)
    #print('-------------------')

    if i % num_show == 0:
        print('loss: ' , loss)
        save_plt(m,c,i)
        

        

print('PREDICTED  : ',m,c)
print('REAL       : ' , m_fake , c_fake)