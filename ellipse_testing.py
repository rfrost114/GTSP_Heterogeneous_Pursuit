import cvxpy as cp
import numpy as np
from gcsopt import GraphOfConvexSets

directed = False # Both directed and undirected work.
graph = GraphOfConvexSets(directed)


mu = 0.5
T = 0.8
alpha = 0.9
rad = alpha * T
centers = np.array([
    [5,1],
    [3,3],
    [1.5,1],
    [1,3.8]
])
centers[:,0] = (centers[:,0] + mu*T) / (1-(mu**2))
centers[:,1] = (centers[:,1]) / (np.sqrt(1-(mu**2)))

major = rad / (1-(mu**2))
minor = rad / (np.sqrt(1-(mu**2)))



v1 = graph.add_vertex(1)
x1 = v1.add_variable(2)
c1 = np.array([centers[0,0], centers[0,1]]) # center of the set
D1 = np.diag([1/major, 1/minor]) # scaling matrix
v1.add_constraint(cp.norm2(D1 @ (x1 - c1)) <= 1)

v2 = graph.add_vertex(2)
x2 = v2.add_variable(2)
c2 = np.array([centers[1,0], centers[1,1]])
D2 = np.diag([1/major, 1/minor]) 
v2.add_constraint(cp.norm2(D2 @ (x2 - c2)) <= 1)

v3 = graph.add_vertex(3)
x3 = v3.add_variable(2)
c3 = np.array([centers[2,0], centers[2,1]])
D3 = np.diag([1/major, 1/minor]) 
v3.add_constraint(cp.norm2(D3 @ (x3 - c3)) <= 1)

v4 = graph.add_vertex(4)
x4 = v4.add_variable(2)
c4 = np.array([centers[3,0], centers[3,1]])
D4 = np.diag([1/major, 1/minor]) 
v4.add_constraint(cp.norm2(D4 @ (x4 - c4)) <= 1)

v5 = graph.add_vertex(5)
x5 = v5.add_variable(2)
v5.add_constraint(cp.norm2(x5) <= 0)

for i, tail in enumerate(graph.vertices):
    heads = graph.vertices[i + 1:]
    if directed:
        heads += graph.vertices[:i]
    for head in heads:
        if tail != head:
            if directed or tail.name < head.name:
                edge = graph.add_edge(tail, head)

                # Edge cost is Euclidean distance.
                x_tail = tail.variables[0]
                x_head = head.variables[0]
                edge.add_cost(cp.norm2(x_head - x_tail))


import matplotlib.pyplot as plt
# plt.figure()
# plt.axis("equal")
# graph.plot_2d()
# plt.show()

graph.solve_traveling_salesman(subtour_elimination=True)
print("Problem status:", graph.status)
print("Optimal value:", graph.value)
print(graph.solver_stats)
plt.figure()
plt.axis("equal")
# graph.plot_2d()
graph.plot_2d_solution()
plt.show()