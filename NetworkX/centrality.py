import networkx as nx
import numpy as np
from networkx.linalg import laplacian_matrix
from scipy.sparse import csr_matrix
import pandas as pd


def DC(G):
    """
    Computes the Degree Centrality for all nodes in the graph G using NetworkX's built-in function.
    
    Parameters:
        G (networkx.Graph): An undirected or directed graph.
    
    Returns:
        np.array: A numpy array where each element corresponds to the degree centrality of a node.
    """
    dc_dict = nx.degree_centrality(G)  # Use built-in function
    dc_vector = np.array(list(dc_dict.values()))  # Convert to numpy array
    
    return dc_vector

def influence_centrality(G):
    
    """
    Compute Influence Centrality for all nodes in the graph.aaa

    Parameters:
    G (networkx.Graph): An undirected networkx graph.

    Returns:
    np.array: A vector where the i-th element is the influence centrality of node i.
    """

    N = G.number_of_nodes()
    influence = np.zeros(N)  # Initialize a vector of zeros
    degrees = np.array([d for _, d in sorted(G.degree())])  # Get degrees in node order

    for i, node in enumerate(sorted(G.nodes())):
        d_i = degrees[i]
        if d_i == 0:
            continue  # Influence remains 0 for isolated nodes

        sum_neighbors = sum((d_i - degrees[j]) / (d_i + degrees[j]) for j in G.neighbors(node))
        influence[i] = (1 / d_i) * sum_neighbors

    return influence


def  BC(G):
    """
    Compute Betweenness Centrality for all nodes in the graph.

    Parameters:
    G (networkx.Graph): An undirected, connected networkx graph.

    Returns:
    np.array: A vector where the i-th element is the betweenness centrality of node i.
    """
    betweenness_dict = nx.betweenness_centrality(G, normalized=True)  # Compute centrality
    betweenness_vector = np.array([betweenness_dict[node] for node in sorted(G.nodes())])  # Convert to NumPy array
    
    return betweenness_vector


def random_walk_betweenness(G):
    """
    Compute Random Walk Betweenness Centrality for all nodes in the graph.

    Parameters:
    G (networkx.Graph): An undirected, connected networkx graph.

    Returns:
    np.array: A vector where the i-th element is the random walk betweenness centrality of node i.
    """
    rw_betweenness_dict = nx.current_flow_betweenness_centrality(G, normalized=True)  # Compute centrality
    rw_betweenness_vector = np.array([rw_betweenness_dict[node] for node in sorted(G.nodes())])  # Convert to NumPy array
    
    return rw_betweenness_vector


def bridge_centrality(G):
    """
    Computes the bridge centrality for all nodes in the graph G.
    
    Parameters:
        G (networkx.Graph): An undirected graph.
    
    Returns:
        dict: A dictionary where keys are nodes and values are their bridge centrality.
    """
    # Compute betweenness centrality
    betweenness = nx.betweenness_centrality(G)
    
    # Compute Bc_i for each node
    bc_values = {}
    for i in G.nodes():
        d_i = G.degree(i)
        if d_i == 0:
            bc_values[i] = 0  # If isolated node, set Bc to 0
            continue
        
        sum_inv_deg = sum(1 / G.degree(j) for j in G.neighbors(i))
        bc_values[i] = (1 / d_i) / sum_inv_deg if sum_inv_deg > 0 else 0
    
    # Compute BRC_i
    bridge_centrality_values = np.array([betweenness[i] * bc_values[i] for i in G.nodes()])

    return bridge_centrality_values

def CC(G):
    """
    Computes the closeness centrality for all nodes in graph G.
    Uses Dijkstra’s algorithm for weighted graphs and BFS for unweighted graphs.
    
    :param G: NetworkX graph (weighted or unweighted)
    :return: A NumPy array containing the closeness centrality of each node
    """
    cc_dict = nx.closeness_centrality(G, distance="weight")  # Fast built-in method
    cc_vector = np.array([cc_dict[node] for node in G.nodes()])  # Convert to NumPy array
    return cc_vector


def closeness_centrality_mfpt(G):
    """
    Computes the Random Walk Closeness Centrality for all nodes in the graph G.
    
    Parameters:
        G (networkx.Graph): An undirected graph.
    
    Returns:
        np.array: A numpy array where each element corresponds to the RWCC of a node.
    """
    # Compute transition probability matrix P
    A = nx.to_numpy_array(G)
    D = np.diag(A.sum(axis=1))
    P = np.linalg.inv(D) @ A  # Transition probability matrix
    
    # Compute steady-state vector w
    eigvals, eigvecs = np.linalg.eig(P.T)
    steady_state = np.real(eigvecs[:, np.isclose(eigvals, 1)]).flatten()
    steady_state /= steady_state.sum()
    W = np.outer(steady_state, np.ones(len(G.nodes())))
    
    # Compute fundamental matrix Z
    I = np.eye(len(G.nodes()))
    Z = np.linalg.inv(I - P + W)
    
    N = len(G.nodes())
    rwcc_values = np.zeros(N)
    
    # Compute Mean First-Passage Time (MFPT)
    MFPT = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                MFPT[i, j] = (Z[j, j] - Z[i, j]) / steady_state[j]
    
    # Compute RWCC for each node
    for i in range(N):
        sum_mfpt = np.sum(MFPT[i, :])  # Sum over all destinations
        if sum_mfpt > 0:
            rwcc_values[i] = (N - 1) / sum_mfpt
        else:
            rwcc_values[i] = 0  # If isolated, set RWCC to 0
    
    return rwcc_values
#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# def information_centrality(G):
#     """
#     Computes the Information Centrality for all nodes in the graph G.
    
#     Parameters:
#         G (networkx.Graph): An undirected graph.
    
#     Returns:
#         np.array: A numpy array where each element corresponds to the information centrality of a node.
#     """
#     # Compute the Laplacian matrix (convert sparse matrix to dense)
#     L = laplacian_matrix(G).toarray()

   
#     #L = L.toarray()  # Convert to dense format using .toarray() if sparse
# # Convert to dense format using .toarray()
#     N = len(G.nodes())
    
#     # Create the all-ones matrix J
#     J = np.ones((N, N))
    
#     # Compute matrix C
#     C = np.linalg.inv(L + J)
    
#     # Compute Information Centrality for each node
#     IC_values = np.zeros(N)
#     trace_C = np.trace(C)
    
#     for i in range(N):
#         sum_Cjj = np.sum(np.diag(C))
#         sum_Cij = np.sum(C[i, :])
#         IC_values[i] = 1 / (C[i, i] - ((sum_Cjj - 2 * sum_Cij) / N))
    
#     return IC_values
# #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def information_centrality(G):
    """
    Computes the Information Centrality for all nodes in a connected undirected graph G.
    
    This version uses the Moore–Penrose pseudoinverse of the Laplacian matrix.
    
    The Information Centrality (IC) of node i is defined as:
    
        IC(i) = 1 / ( (1/n) * sum_{j=1}^{n} (l^+_{ii} + l^+_{jj} - 2 * l^+_{ij}) )
    
    where L^+ is the pseudoinverse of the Laplacian L of G, and n is the number of nodes.
    
    Parameters:
        G (networkx.Graph): A connected, undirected graph.
    
    Returns:
        np.array: A numpy array of information centrality values for each node.
    """
    # Number of nodes
    n = len(G.nodes())
    
    # Compute the Laplacian matrix of G as a dense array.
    # The Laplacian L is defined as D - A where D is the degree matrix and A is the adjacency matrix.
    L = nx.laplacian_matrix(G).toarray()
    
    # Compute the Moore–Penrose pseudoinverse of L.
    # This is more stable than directly inverting (L + J), and it correctly handles the singularity of L.
    L_pinv = np.linalg.pinv(L)
    
    # Precompute the diagonal of L_pinv
    diag = np.diag(L_pinv)
    
    # Initialize the information centrality values array.
    IC_values = np.zeros(n)
    
    # For each node i, compute the denominator:
    #   denom_i = (1/n) * sum_{j=1}^{n} (L_pinv[i,i] + L_pinv[j,j] - 2 * L_pinv[i,j])
    # Then, set IC(i) = 1 / denom_i.
    for i in range(n):
        sum_terms = 0.0
        for j in range(n):
            sum_terms += diag[i] + diag[j] - 2 * L_pinv[i, j]
        denom = sum_terms / n
        
        # If (due to numerical issues) denom is nonpositive, set information centrality to 0.
        if denom <= 0:
            IC_values[i] = 0
        else:
            IC_values[i] = 1 / denom
    
    return IC_values

# eigenvector_centrality

def EC(G, tol=1e-6, max_iter=100):
    """
    Computes the eigenvector centrality for all nodes in graph G using power iteration.
    
    :param G: NetworkX graph (weighted or unweighted)
    :param tol: Tolerance for convergence (default: 1e-6)
    :param max_iter: Maximum iterations before stopping (default: 100)
    :return: A NumPy array containing the eigenvector centrality of each node
    """
    ec_dict = nx.eigenvector_centrality_numpy(G, weight="weight")  # Uses NumPy for fast computation
    ec_vector = np.array([ec_dict[node] for node in G.nodes()])  # Convert to NumPy array
    return ec_vector

# Katz Centrality

def KC(G, alpha=None, beta=1.0):
    """
    Computes the Katz Centrality for all nodes in the graph G.
    
    Parameters:
        G (networkx.Graph): An undirected graph.
        alpha (float, optional): Attenuation factor, must be less than 1/lambda_max.
        beta (float): Exogenous factor (default is 1.0).
    
    Returns:
        np.array: A numpy array where each element corresponds to the Katz centrality of a node.
    """
    # Compute adjacency matrix
    A = nx.to_numpy_array(G)
    N = len(G.nodes())
    
    # Compute largest eigenvalue
    eigvals = np.linalg.eigvals(A)
    lambda_max = max(abs(eigvals))
    
    # Set alpha if not provided (default to 0.85 / lambda_max for convergence)
    if alpha is None:
        alpha = 0.85 / lambda_max
    
    # Compute Katz centrality
    I = np.eye(N)
    ones_vector = np.ones(N) * beta
    x = np.linalg.inv(I - alpha * A) @ ones_vector
    
    return x



def PC(G, alpha=0.85, beta=None):
    """
    Computes the PageRank Centrality for all nodes in the graph G using the matrix formulation.
    
    Parameters:
        G (networkx.Graph): A directed or undirected graph.
        alpha (float): Damping factor (default is 0.85).
        beta (np.array, optional): Custom teleportation vector. If None, defaults to uniform.
    
    Returns:
        np.array: A numpy array where each element corresponds to the PageRank centrality of a node.
    """
    # Compute adjacency matrix
    A = nx.to_numpy_array(G)
    N = len(G.nodes())
    
    # Compute degree matrix D
    D = np.diag(A.sum(axis=1))
    
    # Compute inverse of D
    D_inv = np.linalg.inv(D)
    
    # Compute the transition matrix
    M = np.eye(N) - alpha * D_inv @ A
    
    # Teleportation vector beta (uniform if not provided)
    if beta is None:
        beta = np.ones(N) / N

    # Solve for PageRank vector
    #PR =  np.linalg.inv(M) @ (I- alpha)*beta 
    PR = np.linalg.inv(M) @ ((1 - alpha) * beta)

    return PR


# def PC_pagerank(G, alpha=0.85, beta=None, tol=1e-6, max_iter=100):
#     """
#     Computes PageRank centrality using NetworkX's power iteration method.
    
#     Parameters:
#         G (networkx.Graph): A directed or undirected graph.
#         alpha (float): Damping factor (default = 0.85).
#         beta (np.array or None): Optional personalization/teleportation vector.
#                                  If None, uniform teleportation is used.
#         tol (float): Convergence tolerance (default = 1e-6).
#         max_iter (int): Maximum number of iterations (default = 100).
    
#     Returns:
#         np.array: NumPy array of PageRank values in node order.
#     """
#     if beta is not None:
#         # Convert array to dictionary {node: value}
#         personalization = {node: float(beta[i]) for i, node in enumerate(G.nodes())}
#     else:
#         personalization = None

#     # Compute PageRank with power iteration
#     pr_dict = nx.pagerank(G, alpha=alpha, personalization=personalization,
#                           tol=tol, max_iter=max_iter)

#     # Convert result to a NumPy array in node order
#     pr_vector = np.array([pr_dict[node] for node in G.nodes()])
    
#     return pr_vector



# Function to compute centralities
# this Fucntion compute all centralities for graph G !!!

def compute_centralities_safe(G):
    centralities = {
        "DE": DC(G),
        "EI": EC(G),
        "BE": BC(G),
        "CL": CC(G),
        "IN": influence_centrality(G),
        "RW BE": random_walk_betweenness(G),
        "BR": bridge_centrality(G),
        "CC MFPT": closeness_centrality_mfpt(G),
        "INFO": information_centrality(G),
        "KA": KC(G),
        "PA": PC(G),
    }

    # Compute Katz centrality with a safer approach
    try:
        alpha_safe = 0.9 / max(abs(np.linalg.eigvals(nx.to_numpy_array(G))))
        centralities["KA"] = nx.katz_centrality(G, alpha=alpha_safe, beta=1.0, max_iter=1000, tol=1e-6)
    except (nx.PowerIterationFailedConvergence, np.linalg.LinAlgError):
        print("Katz centrality failed for this graph, skipping it.")
        centralities["KA"] = {node: np.nan for node in G.nodes()}
    centralities = {key: pd.Series(value) for key, value in centralities.items()}
    
    return pd.DataFrame(centralities)


def  compute_centralities(G):
    A = nx.to_numpy_array(G)
    centralities = {
        "DE": DC(G),
        "EI": EC(G),
        "BE": BC(G),
        "CL": CC(G),
        "IN": influence_centrality(G),
        "RW BE": random_walk_betweenness(G),
        "BR": bridge_centrality(G),
        "CC MFPT": closeness_centrality_mfpt(G),
        "INFO": information_centrality(G),
        "KA": KC(G),
        "PA": PC(G),
    }

    # Compute Katz centrality with a safer approach
    try:
        alpha_safe = 0.9 / max(abs(np.linalg.eigvals(A)))
        centralities["KA"] = nx.katz_centrality(G, alpha=alpha_safe, beta=1.0, max_iter=1000, tol=1e-6)
    except (nx.PowerIterationFailedConvergence, np.linalg.LinAlgError):
        print("Katz centrality failed for this graph, skipping it.")
        centralities["KA"] = {i: np.nan for i in range(A.shape[0])}
    
    centralities_matrix = np.column_stack([list(value) for value in centralities.values()])
    return centralities_matrix.T
