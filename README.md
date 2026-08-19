# Linear regression model

this is meant to be a fun project of implementing the most basic model in all of machine learning. this repo is ideally meant for people looking to get started in machine learning.

if you even have the knowledge of high school mathematics, you should be able to follow along well (you just need to know the theory of gradient descent tough).

by this you will get a grasp of how machines learn (do not get fooled by the ease of this, this is just the tip of the iceberg) but a must to learn when beginning.

we know that the equation of a straight line is $y = mx + b$

we are just going to use that.

in this code we have two lists, one X and the other Y.

X maps to Y, our human brain can figure out the relation almost instantly because for the learning purpose the list is simple enough for us to follow.

but how should we get the machine to learn (or predict) the relation between x and y?

well what we are going to do is quite literally guess the values w and b randomly from the beginning, yea it sounds stupid but it is actually how it's done. not in the way you think.

we guess the values of w and b and then we calculate how wrong we are and we make it better the next time.

thats it.

for this we will use partial dervs and gradient descent (basically move in the direction of the slope until you reach the lowest point)

enough theory i guess, lets delve into the mathematics.

after reading this, you should be competent enough to go through the code and understand it.

try writing it on your own for the function y=2x, y=3x or y=0.5x on your own.

# formulas

first we set w=0 and b=0 (because the machine does not know their values yet)

we set learning rate (α) something like 0.01 (it is your choice personally, but keep it small. larger learning rates will not yield results because with smaller learning rates there is more scope for iteration, with larger learning rates you will take huge leaps, in most cases be terribly wrong and eventually never reach the right prediction)

we give 2 lists x and y (they must follow the relation you are essentially trying to get the computer guess)

then we make a prediction, we take each x at a time and use the formula:

$$
\hat{y} = wx + b
$$

then we calculate the error (note the error, not the loss), it is simply:

$$
error = (\hat{y} - y)^2
$$

(we square it to not have to deal with negatives while having many inputs)

loss calculated using the mse function:

$$
L = \frac{1}{n}\sum_{i=1}^{n}(wx_i + b - y_i)^2
$$

we divide by n to find the average loss

we set the partial dervs 0 initially

now here's the part that actually makes it "learn"

we need to find out how much w and b are responsible for the error, so we take the partial derivative of the loss function w.r.t w and b:

$$
\frac{\partial L}{\partial w} = \frac{2}{n}\sum_{i=1}^{n}(wx_i + b - y_i) \cdot x_i
$$

$$
\frac{\partial L}{\partial b} = \frac{2}{n}\sum_{i=1}^{n}(wx_i + b - y_i)
$$

dont be scared of these, all we did was differentiate the mse function w.r.t w and b respectively (basic calculus, chain rule)

now that we know the slope (gradient) of the loss function at our current w and b, we know which direction is "downhill"

so we update our guesses:

$$
w = w - \alpha \frac{\partial L}{\partial w}
$$

$$
b = b - \alpha \frac{\partial L}{\partial b}
$$

we subtract because the gradient points in the direction of steepest increase, and we want to go the other way (towards the minimum loss)

# putting it all together

so the full loop looks something like:

1. start with w=0, b=0
2. make predictions for all x using $\hat{y} = wx + b$
3. calculate the error and loss (mse)
4. calculate $\partial L/\partial w$ and $\partial L/\partial b$
5. update w and b using the formulas above
6. repeat steps 2-5 for a bunch of iterations (epochs)

each time you do this, the loss should get smaller and smaller, and your w and b should get closer to the actual relation between x and y

thats literally it, thats the whole "learning" part. no magic, just calculus and repetition

# how to run

```bash
python main.py
```

play around with the learning rate and number of epochs and see what happens (try making the learning rate too big, watch it blow up, thats a good lesson too)

# try it yourself

before looking at the code, try writing your own version for:
- y = 2x
- y = 3x
- y = 0.5x

just to get the hang of it, once you can do these by hand (well, by code) you basically understand the core of how a machine "learns" a pattern

# note

this is obviously an extremely simplified version of what actual ml training looks like, no libraries, no matrix ops, no vectorization, just plain loops and lists so its easy to follow. once you're comfortable with this, go check out how numpy/pytorch does this under the hood, its the same idea just way faster and scaled up

# visual representation

here is how it actually looks when graphed:

<img width="691" height="411" alt="Linear-regression" src="https://github.com/user-attachments/assets/36197a10-93ff-4d16-a98f-242b78f819f0" />

ps: ignore the values in this image, focus on the plotted line and points
