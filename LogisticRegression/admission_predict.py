import pandas as pd
import numpy as np

# Read data
# Change to data path on your computer
data = pd.read_csv('Admission_Predict.csv')

# Show the description of data
data.describe()

sn = data['Serial No.'].tolist()
gre = data['GRE Score'].tolist()
X1 = np.asarray(gre)
tfl = data['TOEFL Score'].tolist()
X2 = np.asarray(tfl)
unirt = data['University Rating'].tolist()
X3 = np.asarray(unirt)
sop = data['SOP'].tolist()
X4 = np.asarray(sop)
lor1 = data['LOR '].tolist()
X5 = np.asarray(lor1)
cgpa1 = data['CGPA'].tolist()
X6 = np.asarray(cgpa1)
research_exp = data['Research'].tolist()
X7 = np.asarray(research_exp)
prob_Admit = data['Chance of Admit'].tolist()
Yt = np.asarray(prob_Admit)
Yt = np.where(Yt >= 0.75, 1, 0)
# printing list data
X = np.asarray([X1, X2, X3, X4, X5, X6, X7])
y = np.asarray([Yt]).T
x_train = X[:350]
y_train = y[:350]
x_test = X[350:len(X)]
y_test = y[350:len(X)]

Xbar = np.concatenate((np.ones((1, X.shape[1])), X), axis=0)
Xbar = Xbar.T
print(y)
def sigmoid(s):
    if np.any(s > 709):
        s = 708
    elif np.any(s < -708):
        s = -707
    return 1 / (1 + np.exp(-s))


def logistic_sigmoid_regression(X, y, w_init, eta, tol=1e-4, max_count=10000):
    # method to calculate model logistic regression by Stochastic Gradient Descent method
    # eta: learning rate; tol: tolerance; max_count: maximum iterates
    w = [w_init]
    it = 0
    N = X.shape[1]
    d = X.shape[0]
    count = 0
    check_w_after = 20
    # loop of stochastic gradient descent
    while count < max_count:
        # shuffle the order of data (for stochastic gradient descent).
        # and put into mix_id
        mix_id = np.random.permutation(N)
        for i in mix_id:
            xi = X[:, i].reshape(d, 1)
            yi = y[i]
            zi = sigmoid(np.dot(w[-1].T, xi))
            w_new = w[-1] + eta * (yi - zi) * xi
            count += 1
            # stopping criteria
            if count % check_w_after == 0:
                if np.linalg.norm(w_new - w[-check_w_after]) < tol:
                    return w
            w.append(w_new)
    return w

eta = .05
d = x_train.shape[0]
w_init = np.random.randn(d, 1)
w = logistic_sigmoid_regression(x_train, y_train, w_init, eta)
#print(w[-1])
