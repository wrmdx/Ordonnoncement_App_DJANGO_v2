import numpy as np
from scipy.optimize import minimize

# Define the objective function to be minimized
def objective(c):
    return c[3,5]  # This is a placeholder since you're not specifying the objective in your LINGO code

# Define the matrix dimensions
M = 4
JOB = 6
p=[[3,5,2,5,7,3],
[4,3,6,5,7,6],
[6,8,4,4,2,6],
[6,8,5,3,6,8]]
# Decision variables
x = np.zeros((JOB, JOB),dtype=int)
c=np.zeros((4,6))
print(x.shape)
# Define the constraints based on your LINGO code
constraints = []
def const1(x,K):
    return np.sum(x[:, K]) - 1
def const2(x,J):
    return np.sum(x[J, :]) - 1

def const3(c,x,p,J):
    return c[0, J] - np.sum(x[:, J] * p[0][:])

def const4(c,x,p,I,J):
    return c[I, J] - c[I-1, J] - np.sum(x[:, J] * p[I][:])


# Constraint 1: Sum of columns must be 1 for each JOB
for K in range(JOB):
    constraint = {
        'type': 'eq',
        'fun': const1
    }
    constraints.append(constraint)

# Constraint 2: Sum of rows must be 1 for each POS
for J in range(M):
    constraint = {
        'type': 'eq',
        'fun': const2
    }
    constraints.append(constraint)

# Constraint 3: c(1,J) >= SUM(x(K,J) * p(1,K))
for J in range(JOB):
    constraint = {
        'type': 'ineq',
        'fun': const3
    }
    constraints.append(constraint)

# Constraint 4: c(I,J) >= c(I-1,J) + SUM(x(K,J) * p(I,K))
for I in range(1, M):
    for J in range(JOB):
        constraint = {
            'type': 'ineq',
            'fun': const4
        }
        constraints.append(constraint)

# Initial guess (you can provide a better initial guess if available)
initial_guess = np.zeros((JOB, JOB), dtype=int)

# Bounds for decision variables (binary variables)
bounds = [(0, 1)] * (JOB * JOB)  # 0 and 1 represent binary values

# Solve the optimization problem
result = minimize(objective(c), initial_guess.flatten(), constraints=constraints, bounds=bounds)

# Extract and reshape the solution
x_solution = result.x
x_solution = x_solution.reshape(M, JOB)

# Print the solution
print("Solution:")
print(x_solution)