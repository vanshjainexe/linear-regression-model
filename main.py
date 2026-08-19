# this is a basic implmenetation of a linear regression model using raw python 
# we will use gradient descent as the optmizer
# the model however is linear regression 
# lets goo :)
# the function which we are trying to make the machine learn is y=4x+3, so the model should figure out the values of w and b such that it can predict y given x.
# we know that the the equation of line is y=wx+b, so we will try to find the values of w and b such that the loss function is minimized (that is guess y=4x+3 as close as possible (exactly precisely in this case))
# to simpliy this, we will make a machine learn that the function is without telling it the function, how cool is that :)
# you can follow the same steps for any other function you want to implement (it should be linear though)
# ps : you need to know derivates, partial derviates, basic idea of what gradient descent is and how it work (not too complex) and how to calculate the loss function (mean squared error in this case) and how to calculate the gradients of the loss function with respect to the parameters (w and b in this case)



#following are sample inputs
#our main goal is to make the model figure out the relationship between x and y
#if you are beginning with machine learning, I would suggest you to try this out and then move on to more complex models like neural networks and deep learning.
#this guide/code is specifically made for people who are new to machine learning and want to understand how the actual hell goes 0s and 1s know what to do and how to respond to pure language.

x=[1,2,3,4,5,6,7,8,9,10]
y = [7, 11, 15, 19, 23, 27, 31, 35, 39, 43]



#setting up the intial parameters for the mode, we will start updating this in the training loop, and the models goal is to find the values of w and b such that the loss function is minimized (that is guess y=4x+3 as close as possible (exactly precisely in this case))
w=0.0
b=0.0
learning_rate=0.01


#this is the actual learning loop we are going to run for about 10k times to get insane accuracy.
# Each epoch means one complete pass through all training examples.
for epoch in range(10000):
    errors=[]

    for i in range(len(x)):
        prediction=w*x[i]+b
        error=prediction-y[i]
        errors.append(error)


    #this is the mean sqaured function (mse) which goes like 1/n summation from i=1 to n of error^2

    loss=sum(error**2 for error in errors)/len(x)
    dw=0
    db=0

    #this is us calculation the partial derivatives,I have wrote the simplified version of the chain rule of directly, try using the chain rule. you will get the same thing.
    for i in range(len(x)):
        dw+=errors[i]*x[i]*2
        db+=2*errors[i]

    #averaging the gradients:)
    dw=dw/len(x)
    db=db/len(x)

    # updates values of w and b using the learning rate and the gradients, this is the actual gradient descent, mathematically this is wnew=wold-learningrate*partialderivativeofw and bnew=bold-learningrate*partialderivativeofb and like wise for b.

    w=w-learning_rate*dw
    b=b-learning_rate*db

#printing the prediction values of w and x (which will be exact in this case) and the loss function value (which is near 0 in this case)
#note:if you see a value of something like 1e-10 or 1e-12, it is basically 0, so dont worry about that, it is just a very small number which is very close to 0. and also 4.00000000001 is 4 and 2.99999999999937 is also 3, so dont worry about that too. it is just a very small number which is very close to 4 and 3 respectively.
#it is recommended to change the code by yourself to learn more deeply

print(w,b)
print("the function is : y = {}x + {}".format(w, b)) 