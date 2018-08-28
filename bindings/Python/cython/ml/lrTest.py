from numpy import *


def compute_error_for_line_given_points(inital_b, inital_m, data):
    #initialize at 0
    totalError = 0
    #for every point
    for i in range(0, len(data)):
        x= data[i,0]
        y = data[i,1]
        #get the difference square it, add it to the total
        totalError+= (y -(inital_m*x+inital_b))**2
    return (totalError/len(data))


def step_gradient(current_b, current_m, points, learning_rate):
    #starting points for our gradients
    b_gradient = 0
    m_gradient = 0
    N = len(points)
    for i in range(0, len(points)):
        x= points[i,0]
        y=points[i,1]
        #direction with respect to b and m
        #computing partial derivatives of our error function
        b_gradient+= -(2/N)*(y-((current_m*x)+current_b))
        m_gradient+= (2/N)* x*(y-((current_m*x)+current_b))
    #update our b and m values using our partial derivatives
    new_b = current_b -(learning_rate*b_gradient)
    new_m = current_m -(learning_rate*m_gradient)
    return[new_b, new_m]


def gradient_descent_runner(points, inital_b, inital_m, learning_rate, number_iterations):
    #gradient descent
    b=inital_b
    m=inital_m

    for i in range(number_iterations):
        #update b and m with the new more accurate b and m by performing
        #this gradient step
        b, m = step_gradient(b,m, array(points), learning_rate)
    return [b,m]


def run():
    #step 1: collect our data
    data = genfromtxt('data.csv', delimiter=',', skip_header=1)
    #print(data)

    #step 2: define our parameters
    #how fast should our model converge?
    learning_rate = 0.0001
    #y=mx+b
    inital_m=0
    inital_b=0
    number_iterations = 2000
    #step3: train our model
    print('starting gradient descent at b={0}, m = {1}, error = {2}'.format(inital_b,inital_m, compute_error_for_line_given_points(inital_b,inital_m, data)))
    [b,m]= gradient_descent_runner(data, inital_b, inital_m, learning_rate, number_iterations)
    print('ending point at b={1}, m = {2}, error = {3}'.format(number_iterations, b,m,
                                                                            compute_error_for_line_given_points(
                                                                                b, m, data)))

    #print(data)




if __name__ == '__main__':
    run()