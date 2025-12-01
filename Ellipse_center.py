import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import networkx as nx



def points_to_distance_matrix(starting_points: np.ndarray) -> np.ndarray:
    
    starting_points_w_zero = np.vstack((np.zeros((1,2)), starting_points))
    total_points = starting_points_w_zero.shape[0]
    distance_matrix = np.zeros((total_points, total_points))

    for i in range(total_points):
        for j in range(total_points):
            if i != j:
                distance_matrix[i,j] = np.linalg.norm(starting_points_w_zero[i] - starting_points_w_zero[j])
    print(distance_matrix)
    return distance_matrix


# returns C, X, u
def get_variables(starting_points : np.ndarray) -> tuple[np.ndarray, cp.Variable, cp.Variable]:
    C = points_to_distance_matrix(starting_points)
    X = cp.Variable(C.shape, boolean=True)
    u = cp.Variable(C.shape[0])
    return C, X, u


# returns the cvxpy objective function for the MTZ formulation of TSP
def get_objective_function_MTZ(C : np.ndarray, X : cp.Variable) -> cp.Expression:
    return cp.Minimize(cp.sum(cp.multiply(C, X)))

def form_constraints(num_points : int, X : cp.Variable, u : cp.Variable) -> list[cp.Constraint]:
    ones_vector = np.ones((num_points+1, 1))
    constraint_set = []
    constraint_set.append(cp.diag(X) == 0)
    constraint_set.append(X @ ones_vector == ones_vector)
    constraint_set.append(X.T @ ones_vector == ones_vector)
    constraint_set.append(u[1:] <= num_points+1)
    constraint_set.append(u[1:] >= 2)
    constraint_set.append(u[0] == 1)

    # now the subtour elimination constraints:
    for i in range(1, num_points+1):
        for j in range(1, num_points+1):
            if i != j:
                constraint_set.append(u[i]- u[j]+ 1 <= (num_points) * (1-X[i,j]))
    return constraint_set

if __name__ == '__main__':
    starting_points = np.array([
        [5,1],
        [3,3],
        [1.5,1],
        [1,3.8]
    ])
    C, X, u = get_variables(starting_points)
    # print(X.value)
    objective = get_objective_function_MTZ(C, X)
    constraints = form_constraints(4, X, u)
    program = cp.Problem(objective, constraints)
    program.solve('MOSEK', verbose=True)
    # print(X.value)
    print(np.sum(X.value @ C))


