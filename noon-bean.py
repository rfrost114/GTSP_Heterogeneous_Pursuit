import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import copy
import cvxpy as cp
import itertools
import random
from collections.abc import Callable

# deterimines the center and radius of the capture circles for the intercept problem
# Parameters: 
#   starting points: a numpy arry of size nx2 representing the starting locations of the n targets
#   alpha: The velocity of the interceptor 0 <= alpha <= 1
#   T: Time required by interceptor to hit target
#   mu: The velocity of of the target, mu < alpha
# Returns:
#   circle centers: A numpy array of size nx2 representing the centers of the capture circles at time 0
#   radius: the radius of the capture circles (uniform for all targets)
def get_init_capture_circles(starting_points: np.ndarray, alpha: float, T: float, mu: float) -> tuple[np.ndarray, float]:
    radius = alpha * T
    circle_centers = np.zeros(starting_points.shape)
    circle_centers[:, 0] = starting_points[:,0] + (mu * T)
    circle_centers[:, 1] = starting_points[:,1]
    return circle_centers, radius

def visualize_circles(circle_centers : np.ndarray, radius : float, show=True, data=None) -> None:
    fig , ax = plt.subplots()
    for i in range(circle_centers.shape[0]):
        c = patches.Circle(circle_centers[i,:], radius, color='m', fill=False)
        ax.add_patch(c)
        
    ax.set_aspect('equal', adjustable='box')
    x_max, y_max = np.max(circle_centers[:,0])+radius+1, np.max(circle_centers[:,1])+radius+1
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)    
    ax.set_xticks(range(int(x_max)+1))
    ax.set_yticks(range(int(y_max)+1))
    if data is not None:
        for data_set in data:
            plt.scatter(data_set[:,0], data_set[:,1], c='b')
    if show:
        plt.title('Intial Circle Locations')
        plt.show()

def visualize_ellipses(starting_points: np.ndarray, alpha: float, T: float, mu: float, show=True, data=None) -> None:
    major_axis = (alpha*T) / (1- (mu**2))
    minor_axis = (alpha*T) / np.sqrt((1- (mu**2)))
    ellipse_centers = np.zeros(starting_points.shape)
    ellipse_centers[:,0] = (starting_points[:,0] + (mu*T)) / (1- (mu**2))
    ellipse_centers[:,1] = starting_points[:,1] / np.sqrt((1- (mu**2)))

    fig , ax = plt.subplots()
    for i in range(ellipse_centers.shape[0]):
        el = patches.Ellipse(ellipse_centers[i,:], width=2*major_axis, height=2*minor_axis, color='m', fill=False)
        ax.add_patch(el)
    ax.set_aspect('equal', adjustable='box')   
    x_max, y_max = np.max(ellipse_centers[:,0])+major_axis+1, np.max(ellipse_centers[:,1])+minor_axis+1
    ax.set_xlim(0, x_max)
    ax.set_ylim(0, y_max)    
    ax.set_xticks(range(int(x_max)+1))
    ax.set_yticks(range(int(y_max)+1))
    if data is not None:
        for data_set in data:
            plt.scatter(data_set[:,0], data_set[:,1], c='b')
    if show:
        plt.title('Static Frame Ellipses')
        plt.show()


# samples n points uniformly from each of the circles 
# set seed to an integer for standardization
def circle_sample(centers: np.ndarray, radius : float, n : int, seed=None) -> list[np.ndarray]:

    if seed is not None:
        np.random.seed(seed)

    samples = []
    for circle in range(centers.shape[0]):
        circle_points = []
        for _ in range(n):
            # create a random point in polar coordinates 
            r = radius * np.sqrt(np.random.random())
            theta = 2 * np.pi * np.random.random()

            # now return to cartesian coords 
            center_x, center_y = centers[circle, 0], centers[circle, 1]
            x = center_x + (r*np.cos(theta))
            y = center_y + (r*np.sin(theta))
            circle_points.append([x,y])
        samples.append(np.array(circle_points))
    return samples

# transforms samples to static frame
def static_frame_transform(samples: list[np.ndarray], mu: float) -> list[np.ndarray]:
    static_samples = []
    for sample in samples:
        t = np.zeros(sample.shape)
        t[:,0] = sample[:,0] / (1-(mu)**2)
        t[:,1] = sample[:,1] / np.sqrt(1-(mu)**2)
        static_samples.append(t)
    return static_samples

def generate_digraph_representation(samples : list[np.ndarray]) -> tuple[nx.DiGraph, list[list[int]]]:
    # an empty graph object
    G = nx.DiGraph()

    # first calculate the number of nodes, adding an extra node for the vehicles starting location
    graph_size = (len(samples) * samples[0].shape[0]) + 1

    # now add nodes to the graph, storing their position as an attribute for later access
    G.add_node(0, position=np.array([0,0])) # node zero represents the origin
    label = 1 # labeling of other nodes begins at 1
    node_clusters = [[0]] # we will keep track of which nodes are in what circle to make adding the edges easier
    for sample in samples:
        cluster = []
        for i in range(sample.shape[0]):
            cluster.append(label)
            G.add_node(label, position=sample[i,:])
            label += 1
        node_clusters.append(cluster)
    
    # now we will add the edges to the graph, the graph is "almost" complete, all arcs are presnt save for those within the same cluster
    for i, cluster in enumerate(node_clusters):
        non_cluster_nodes = []
        for j in range(len(node_clusters)):
            if i != j:
                non_cluster_nodes += node_clusters[j]
        
        for c_node in cluster: 
            for node in non_cluster_nodes:
                # use the euclidean distance as the weight
                distance = np.linalg.norm(G.nodes[c_node]['position'] - G.nodes[node]['position'])
                G.add_edge(c_node, node, weight=distance)

    return G, node_clusters


def visualize_graph(graph : nx.DiGraph, node_labels=True, edge_weights=True) -> None:
    positioning = {}
    for node in graph.nodes:
        positioning[node] = tuple(graph.nodes[node]['position'])
    nx.draw(graph, pos=positioning)

    if node_labels:
        nx.draw_networkx_labels(graph, pos=positioning)
    if edge_weights:
        e_labs = {(u, v) : w for u, v, w in graph.edges(data="weight")}
        nx.draw_networkx_edge_labels(graph, pos=positioning, edge_labels=e_labs)


def noon_bean_transformation(graph : nx.DiGraph, node_clusters : list[list[int]]) -> nx.DiGraph:
    clustered_tsp_graph = to_clustered_tsp(graph, node_clusters)
    # visualize_graph(clustered_tsp_graph)
    # plt.show()
    total_edge_weight = 0
    for _, _, w in clustered_tsp_graph.edges(data="weight"):
        total_edge_weight += w
    total_edge_weight += 10
    
    cluster_dict ={}
    for i in range(len(node_clusters)):
        for node in node_clusters[i]:
            cluster_dict[node] = i

    for u, v in clustered_tsp_graph.edges:
        if cluster_dict[u] != cluster_dict[v]:
            clustered_tsp_graph.edges[u,v]['weight'] += total_edge_weight

    # visualize_graph(clustered_tsp_graph)
    # plt.show()
    
    return clustered_tsp_graph
    



def to_clustered_tsp(graph : nx.DiGraph, node_clusters : list[list[int]]) -> nx.DiGraph:
    c_tsp_graph = nx.DiGraph()
    for node, pos in graph.nodes(data="position"):
        c_tsp_graph.add_node(node, position=pos)
    
    for cluster in node_clusters:
        # if the cluster has exactly one node, just copy its outgoing edges
        if len(cluster) == 1:
            arcs = list(graph.out_edges(cluster[0], data='weight'))
            
            for u , v , w in arcs:
                c_tsp_graph.add_edge(u,v, weight=w)
        # other wise adjust the arc so they eminate from the 'previous vertex' 
        else:
            max_node = max(cluster)
            for n_index, node in enumerate(cluster):
                if n_index == 0:
                    new_root = max_node
                else:
                    new_root = cluster[n_index-1]
                arcs = list(graph.out_edges(node, data='weight'))
                for u, v, w in arcs:
                    c_tsp_graph.add_edge(new_root, v, weight=w)
        
        # after this is done add a cycle of arcs in the cluster with weight 0
        
        if len(cluster) > 1:
            for i in range(len(cluster)):
                if i == len(cluster)-1:
                    c_tsp_graph.add_edge(cluster[i], cluster[0] , weight=np.float64(0))
                else:
                    c_tsp_graph.add_edge(cluster[i], cluster[i+1] , weight=np.float64(0))
    return c_tsp_graph

def recover_path(nb_edges: np.ndarray, node_clusters: list[list[int]]) -> list[tuple[int, int]]:
    backup = copy.deepcopy(nb_edges)
    nb_edges = [list(i) for i in nb_edges]
    
    ordered_path = []


    while len(nb_edges):
        start_length = len(nb_edges)
        if len(ordered_path) == 0:
            search_node = 0
        else:
            search_node = ordered_path[-1][1]
        
        for i in range(len(nb_edges)):
            if nb_edges[i][0] == search_node:
                ordered_path.append(nb_edges.pop(i))
                break
        end_length = len(nb_edges)
        # print(nb_edges)
        # print(ordered_path)
        if start_length == end_length:
            print(f'no valid tour {backup}')
            print(f'didnt find edge starting with {search_node}')
            return None
    
    true_edges = []
    for edge in ordered_path:
        u, v = int(edge[0]), int(edge[1])
        valid_edge = True
        for cluster in node_clusters:
            if u in cluster and v in cluster:
                valid_edge = False
                break
        if valid_edge:
            true_edges.append(edge)
    
    tsp_path = [true_edges[0][0], true_edges[0][1]]
    tsp_path = [true_edges[0][0], true_edges[0][1]]
    for edge in true_edges[1:]:
        target_node = int(tsp_path[-1])
        target_cluster = None
        for cluster in node_clusters:
            if target_node in cluster:
                target_cluster = cluster 
                break
        
        u, v = int(edge[0]), int(edge[1])

        if u not in target_cluster and v not in target_cluster:
            tsp_path += [u,v]
        else:
            tsp_path.append(u if u not in target_cluster else v)

    tsp_path = [int(i) for i in tsp_path]

    edge_view = []
    for i in range(len(tsp_path)-1):
        edge_view.append((tsp_path[i], tsp_path[i+1]))
    

    return edge_view
    
def to_adjacency_matrix(graph : nx.DiGraph) -> np.ndarray:


    sum_weight = 0
    for _, _, w in graph.edges(data="weight"):
        sum_weight += w
    sum_weight += 10
    adj = nx.to_numpy_array(a_tsp, a_tsp.nodes, nonedge=sum_weight)

    return adj


def visualize_path(digraph : nx.Graph, path: list[list[int]], starting_points: np.ndarray, alpha: float, T: float, mu: float) -> None:
    visualize_ellipses(starting_points, alpha, T, mu, show=False)
    visualize_graph(digraph, True, False)
    positioning = {}
    for node in digraph.nodes:
        positioning[node] = tuple(digraph.nodes[node]['position'])
    nx.draw_networkx_edges(digraph, pos=positioning, edgelist=path, width=5, edge_color='g', alpha=0.5)  
    plt.show()



    


if __name__ == '__main__':
    start = np.array([
        [5,1],
        [3,3],
        [1.5,1],
        [1,3.8]
    ])
    alpha = 0.9
    T = 0.5
    mu = 0.5
    centers, radius = get_init_capture_circles(start, alpha, T, mu)
    # visualize_circles(centers, radius, True)

    # transform to ellipse 
    # visualize_ellipses(start, alpha, T, mu)

    # sample uniformly in the circles
    samples = circle_sample(centers, radius, 4, seed=1127)
    
    # show the sampled points
    # visualize_circles(centers, radius, True, data=samples)

    # transform into elliptical frame and visualize within the ellipses 
    samples = static_frame_transform(samples, mu)
    # visualize_ellipses(start, alpha, T, mu, data=samples)


    dg, n_clusters = generate_digraph_representation(samples)
    # visualize_ellipses(start, alpha, T, mu, show=False, data=samples)
    # visualize_graph(dg, edge_weights=False)
    # plt.show()

    # Noon Bean transformation of GTSP to ATSP
    a_tsp = noon_bean_transformation(dg, n_clusters )
    # visualize_ellipses(start, alpha, T, mu, show=False, data=samples)
    # visualize_graph(a_tsp, edge_weights=False)
    
    # take the final Graph to the adjancency Matrix
    # C = to_adjacency_matrix(a_tsp)

    # # MTZ ILP Formulation
    # X = cp.Variable(C.shape, boolean=True)
    # u = cp.Variable(C.shape[0])
    # objective = cp.Minimize(cp.sum(cp.multiply(C, X)))
    # n = C.shape[0]
    # constraint_set = []
    # ones_vector = np.ones((n, 1))
    # constraint_set.append(cp.diag(X) == 0)
    # constraint_set.append(X @ ones_vector == ones_vector)
    # constraint_set.append(X.T @ ones_vector == ones_vector)
    # constraint_set.append(u[1:] <= n)
    # constraint_set.append(u[1:] >= 2)
    # constraint_set.append(u[0] == 1)
    # for i in range(1, n):
    #     for j in range(1, n):
    #         if i != j:
    #             constraint_set.append(u[i]- u[j]+ 1 <= (n-1) * (1-X[i,j]))
    # program = cp.Problem(objective, constraint_set)
    # program.solve('MOSEK', verbose=False)
    # # print(program.status)
    # X_sol = np.argwhere(X.value >= 0.5)

    # # # recover the path in the original frame (pre noon bean transformation)
    # digraph_tsp_path = recover_path(X_sol, n_clusters)
    # if digraph_tsp_path is not None:
    #     plt.close('all')
    #     print(digraph_tsp_path)
    #     total_cost = 0
    #     for e in digraph_tsp_path:
    #         u, v = e[0], e[1]

    #         total_cost += dg.edges[u, v]['weight']
    #     print(total_cost)
    #     # visualize_path(dg, digraph_tsp_path, start, alpha, T, mu)
    # else:
    #     print(X_sol)
    #     print(X.value.astype(np.int32))
    #     print(X.value)
    
    

# Simulated annealing testing 

def complete_atsp(atsp_graph: nx.DiGraph)-> nx.DiGraph:
    complete_graph = copy.deepcopy(atsp_graph)
    total_weight = 0
    for _, _, w in atsp_graph.edges(data="weight"):
        total_weight += (w+1)
    
    for u, v in nx.complement(atsp_graph).edges:
        complete_graph.add_edge(u, v, weight=total_weight)
    
    return complete_graph
    
    
# simmulated annealing testing
if __name__=='__main__':
    catsp = complete_atsp(a_tsp)
    sim_solution = nx.algorithms.approximation.simulated_annealing_tsp(catsp, init_cycle='greedy', N_inner=500, temp=4000)
    print(sim_solution)
    
    # put the simmulated annealing solution in the form we want it to use our old function
    edge_view_path = [tuple(x) for x in itertools.pairwise(sim_solution)]
    # for i in range(len(sim_solution)-1):
    #     edge_view_path.append([sim_solution[i], sim_solution[i+1]])
    # print(edge_view_path)
    
    # print(list(itertools.pairwise(sim_solution)))
    sim_a_path = recover_path(np.array(edge_view_path), n_clusters)

    print(sim_a_path)
    total_cost = 0
    for e in sim_a_path:
        u, v = e[0], e[1]

        total_cost += dg.edges[u, v]['weight']
    print(total_cost) 

# simmulated annealing algorithm for tsp


# intial path function
# Graph : the graph to extract a route for
# seed : an integer random seed
# -> Path a list of nodes that represents the order of the tsp path
def inital_path(graph : nx.DiGraph, seed=None) -> list[int]:
    if seed is not None:
        random.seed(seed)
    nodes = list(graph.nodes)
    random.shuffle(nodes)
    return nodes

# randomly transposes two nodes from a path list 
def transpose_path_nodes(path : list[int]) -> list[int]:
    i, j = random.sample(list(range(len(path))), 2)
    path[i], path[j] = path[j], path[i]
    return path

def path_length(graph: nx.graph , path: list[int]) -> float:
    return sum(graph[u][v].get('weight', 1) for u, v in itertools.pairwise(path))

def simulated_annealing_TSP(graph : nx.digraph, N_iter : int, init_temperature : float, cooling_function: Callable, alpha: float, seed=None, value_list = True) -> list[int]:
    # at time step zero our best solution is a random path through the graph
    best_path = inital_path(graph, seed)
    # the cost is given by the sum over the edge lengths
    best_cost = path_length(graph, best_path)
    current_iter = 0

    if value_list:
        values = []
    
    
    while current_iter <= N_iter:
        current_iter += 1
        
        # create a new proposed path
        proposed_path = transpose_path_nodes(copy.copy(best_path))
        proposed_cost = path_length(graph, proposed_path)
        if value_list:
            values.append(proposed_cost)
        cost_difference = proposed_cost - best_cost
        # always accept a path that has a lower cost 
        temp = cooling_function(init_temperature, current_iter, alpha)
        if temp <= 0:
            break
        if cost_difference <= 0:
            best_cost = proposed_cost
            best_path = proposed_path.copy()
        # a worse solution is accepted with some probability 
        else:
            prob = min(1, np.exp(-cost_difference / temp))
            if prob >= random.random():
                best_cost = proposed_cost
                best_path = proposed_path.copy()
        
        # print(best_path)
    print(f'{best_cost=}')
    if value_list:
        return best_path + [best_path[0]], values
    
    return best_path + [best_path[0]]

## some cooling functions
# def linear_cooling(t, i, a):
#     return t - a * i

def exponential_cooling(t, i, a):
    return t * (a**i)

def fast_cooling(t, i, a):
    return t/i

def log_cooling(t, i, a):
    return t * (np.log(2)/np.log(i+2))

if __name__ == '__main__':

    # get an estimate of what the inital temperature should be using a random sample of 50 paths from the graph
    temp_estimate = np.average([path_length(catsp, inital_path(catsp)) for _ in range(50)])
    print(temp_estimate)
    path, values = simulated_annealing_TSP(catsp, 10000, temp_estimate, exponential_cooling, 0.9, seed=1127)
    p = recover_path(np.array([tuple(x) for x in itertools.pairwise(path)]), n_clusters)
    print(p)
    print(path_length(dg, [h[0] for h in p]+[p[0][0]]))
    plt.plot(values)
    plt.show()

    path, values = simulated_annealing_TSP(catsp, 10000, temp_estimate, fast_cooling, 0.9, seed=1127)
    p = recover_path(np.array([tuple(x) for x in itertools.pairwise(path)]), n_clusters)
    print(p)
    print(path_length(dg, [h[0] for h in p]+[p[0][0]]))
    plt.plot(values)
    plt.show()

    path, values = simulated_annealing_TSP(catsp, 10000, temp_estimate, log_cooling, 0.9, seed=1127)
    p = recover_path(np.array([tuple(x) for x in itertools.pairwise(path)]), n_clusters)
    print(p)
    print(path_length(dg, [h[0] for h in p]+[p[0][0]]))
    plt.plot(values)
    plt.show()