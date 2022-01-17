from matplotlib import pyplot as plt
import numpy as np
from scipy import linalg as la
from matplotlib.patches import Circle
from scipy.spatial.distance import pdist,squareform
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.stats import multivariate_normal
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from umap import UMAP
import pandas as pd
import numpy as np
import networkx as nx

class Node:
    """
    Node in a tree with its associated with a normal distribution.

    Parameters
    ----------
    mean : (M,) ndarray
        Mean of the normal distribution associated with the node
    stdev : float
        Standard deviation of the normal distribution associated with
        the node
    N : int
        number of data points to generate from the normal distribution
        associated with the node
    parent : Node default=None
        Parent of the node within a tree structure
    seed : int
        Random seed used to generate the normal distribution associated with
        the node. Random distributions are only seeded for root nodes.
    pseudotime : float
        Distance in time between this node and the root node.

    Attributes
    ----------
    mean : (M,) ndarray
        Mean of the normal distribution associated with the node
    stdev : float
        Standard deviation of the normal distribution associated with
        the node
    N : int
        number of data points associated with the node; generated from the
        associated normal distribution
    parent : Node
        Parent node within tree structure
    children : list of Nodes
        Children of the node within tree structure
    level : int
        Number of generations between this node and the root node.
    id : int
        Identification number of the node
    dim : int
        Dimension M of the data associated with the node. Implied from the shape
        of `mean`
    data : (N,M) ndarray
        N data points in M dimensions associated with the node; generated from
        the associated normal distribution
    reduced_data : dictionary
        Dictionary which maps a method type (e.g. 'pca', 'tsne', 'umap') to the
        reduced data associated with that node. Because the reduction depends on
        the entire tree and not just an individual node, `reduced_data` is
        initially empty until the a Tree is created from the root node
    """
    #initialize class variables: colormap and id counter
    _maxid = 0

    def __init__(self,mean,stdev,N,pseudotime=0,parent=None,seed=124):
        #set mean and stdev
        self.mean = mean
        self.dim = len(mean)
        self.stdev = stdev
        if pseudotime==0 and parent is not None:
            raise ValueError("Pseudotime is only 0 at the root node")
        else:
            self.pseudotime = pseudotime
        #generate data associated with the node
        self.N = N
        self.data = np.random.normal(loc=mean,scale=stdev,size=(N,self.dim))
        self.reduced_data = {}
        #initialize relationships
        self.parent = parent
        self.children = []
        #node counters: node id and level down the tree
        if parent is None:
            np.random.seed(seed)
            self.level = 0
            Node._maxid = 0
        else:
            self.level = parent.level+1
        self.id = Node._maxid
        Node._maxid += 1

    def add_child(self,delta_t,N=None):
        """
        Randomly adds a child to a node using brownian motion from the previous node's mean.

        Parameters
        ----------
        delta_t : float
            "Time difference" between parent and child nodes. In practice, this
            means that the displacement between the means of the parent and child
            nodes is distributed as a normal distribution with mean 0 and standard
            deviation `sqrt(delta_t)`

        """
        #find displacement between parent and child
        displacement = np.random.randn(self.dim)*np.sqrt(delta_t)
        if N is None:
            N = self.N
        #new node location
        new_mean = self.mean+displacement
        #create node
        child = Node(parent=self,mean=new_mean,stdev=self.stdev,N=N,
                        pseudotime=self.pseudotime+delta_t)
        self.children.append(child)

    def __str__(self):
        """
        Gets a string representation of the node
        """
        tabs = '-'*self.level
        return tabs + f"Node {self.id}: N({self.mean},{self.stdev})"

    def print_posterity(self):
        """
        Prints the descendancy of a node.
        """
        #recursively print posterity
        print(self)
        for child in self.children:
            child.print_posterity()

    def plot(self,ax,tree,method='pca'):
        """
        Plots the data associated with a node on a particular set of axes.

        Parameters
        ----------
        ax : matplotlib axes
            Axes on which to plot the data
        method : {None, 'pca', 'tsne', 'umap'} default=None
            Method used to reduce the data for plotting purposes. Must have
            already created a Tree from the root data, which will assigned values
            to the `reduced_data` attribute under the hood. If method is `None`,
            the plot will plot the first two coordinates of the raw, unreduced data.
        """
        #find the color
        color = tree._colors[self.id]
        if method is None:
            data = self.data
        else:
            data = self.reduced_data[method]
        #plot the data
        ax.plot(data[:,0],data[:,1],
                color=color,
                markersize=3,
                alpha=.4,
                marker = '.',
                linestyle = '',
                label=self.id)

class Tree:
    """
    Initializes a tree from its root node

    Parameters
    ----------
    root : Node
        Root node of tree.
    methods : list of strings
        Which methods of dimensionality reduction to perform on the tree data.
        Options include ['pca','pca-34','pca-56','tsne','umap',
                        'Laplacian eigenmaps','Laplacian eigenmaps-34',
                        'Laplacian eigenmaps-56']
        PCA-34 returns the 3rd and 4th principal components, same for 56 and for
        Laplacian eigenmaps.


    Attributes
    ----------
    root : Node
        Root node
    nodelist : List of nodes
        List of nodes in the tree, including the root and all of its descendants.
        Ordered according to the id numbers of the nodes
    data : ndarray
        Array of all the data associated with the tree.
    labels : list
        List of labels which assign each data point to the id of the node that generated it
    leaf_mask : ndarray of bools
        Boolean array which is True for data points associated with leaf nodes
        and False for data points associated with interior nodes
    reduced_data : dictionary
        Maps the reduction method to an array of the data reduced via that method.
    methods : list of strings
        Methods used to reduce the data
    """
    def __init__(self,root,methods=['pca','tsne']):
        self.root=root
        #make a list of all the nodes using BFS
        queue = [root]
        self.nodelist = [root]
        while len(queue) > 0:
            curr = queue.pop()
            self.nodelist.extend(curr.children)
            queue.extend(curr.children)
        #reorder nodelist
        order = [node.id for node in self.nodelist]
        self.nodelist = [self.nodelist[id] for id in np.argsort(order)]
        #reduce data
        self.data = [node.data for node in self.nodelist]
        self.labels = [[node.id]*node.N for node in self.nodelist]
        self.data = np.concatenate(self.data)
        self.labels = np.concatenate(self.labels)
        srt = np.argsort(self.labels)
        self.labels = self.labels[srt]
        self.data = self.data[srt]
        #leaf mask and reductions
        self.leaf_mask = np.ravel([[len(node.children)==0]*node.N for node in self.nodelist])
        self.reduced_data = {}
        self.methods = methods
        for method in methods:
            self._reduce(method,newdim=2)
        #make colormap
        _cm = plt.get_cmap('rainbow')
        unique_labels = [node.id for node in self.nodelist]
        self._colors = {label:_cm(label/len(unique_labels)) for label in unique_labels}
        #get true tree matrix
        numnodes = len(self.nodelist)
        self.tree_matrix = np.zeros([numnodes]*2,dtype=np.bool)
        indices = np.triu_indices(numnodes)
        for node in self.nodelist:
            for child in node.children:
                self.tree_matrix[node.id,child.id] = True
                self.tree_matrix[child.id,node.id] = True


    def savecsv(self,filename):
        """
        Saves the data and reduced data from a tree as csv files. Columns will appear as
        either "raw" or the name of the reduction method, followed by the component number

        Parameters
        ----------
        filename : str
            Name of the file, with the '.csv' file extension.
        """
        #put in raw data
        df = pd.DataFrame(self.data,columns=[f'Raw{num}'for num in range(self.data.shape[1])])
        #put in columns for datalabels and whether nodes are leaf nodes or not
        tmp_df = pd.DataFrame(self.labels,columns=['Labels'])
        df = tmp_df.join(df)
        tmp_df = pd.DataFrame(self.leaf_mask,columns=['IsLeaf'])
        df = tmp_df.join(df)

        #save reduced data
        for method in self.methods:
            tmp_df = pd.DataFrame(self.reduced_data[method],columns=[method+str(num)
                                    for num in range(self.reduced_data[method].shape[1])])
            df = df.join(tmp_df)

        #rename indices to Row{idx]}
        def make_row_name(idx):
            return f'Row{idx}'
        df = df.rename(index = make_row_name)

        #save to csv
        df.to_csv(filename,index=True,index_label='Indices')

    def _reduce(self,method,newdim=2):
        #get the reducer object
        print(method,'start')
        if method=='pca':
            reducer = PCA(n_components=newdim)
        elif method=='pca-34':
            reducer = PCA(n_components=4)
            #save the reduced data to the tree
            self.reduced_data[method] = reducer.fit_transform(self.data)[:,2:4]
        elif method=='pca-56':
            reducer = PCA(n_components=6)
            #save the reduced data to the tree
            self.reduced_data[method] = reducer.fit_transform(self.data)[:,4:6]
        elif method=='tsne':
            reducer = TSNE(n_components=newdim,n_iter=500,init='pca',learning_rate=1000)
        elif method=='umap':
            reducer = UMAP(n_components=newdim)
        elif method=='Laplacian eigenmaps':
            reducer = SpectralEmbedding(n_components=newdim)
        elif method=='Laplacian eigenmaps-34':
            reducer = SpectralEmbedding(n_components=4)
            #save the reduced data to the tree
            self.reduced_data[method] = reducer.fit_transform(self.data)[:,2:4]
        elif method=='Laplacian eigenmaps-56':
            reducer = SpectralEmbedding(n_components=6)
            #save the reduced data to the tree
            self.reduced_data[method] = reducer.fit_transform(self.data)[:,4:6]
        else:
            raise ValueError('method must be one of {pca,tsne,umap}')
        #save reduced data to the tree
        if method not in {'pca-34','pca-56','Laplacian eigenmaps-34','Laplacian eigenmaps-56'}:
            self.reduced_data[method] = reducer.fit_transform(self.data)
        #save the reduced data for that node to the node
        for node in self.nodelist:
            node.reduced_data[method] = self.reduced_data[method][self.labels==node.id]
        print(method,'done')
        return self.reduced_data[method]

    def plot_summary(self):
        """
        Plots the tree and minimum spanning trees recovered before and after
        dimensionality reduction. Plot appears as a `len(methods)` by 3 grid.
        Each row uses a different dimensionality reduction technique to visualize
        the data. The first column displays the true underlying tree structure.
        The second column displays the minimum spanning tree recovered using
        distances between nodes as computed before dimensionality reduction. The
        third column displays the minimum spanning tree recovered using
        distances between nodes as computed after dimensionality reduction.
        """
        num_methods=len(self.methods)
        fig = plt.figure(figsize=(15,5*num_methods))
        plotnum = 1
        for method in self.methods:
            #true tree
            ax1 = fig.add_subplot(num_methods, 3, plotnum)
            self.plot_tree(ax1,method,'True Tree')
            plotnum += 1
            ax1.set_ylabel(method.upper())
            #distances before reduction
            ax2 = fig.add_subplot(num_methods, 3, plotnum)
            mst,nodes = get_mst(self.data,self.labels)
            nodes = np.array([node.reduced_data[method].mean(axis=0) for node in self.nodelist])
            self.plot_tree(ax2,method,'Recovered\nDistances Before Reduction',mst,nodes)
            plotnum += 1
            #distances after reduction
            ax3 = fig.add_subplot(num_methods, 3, plotnum)
            mst,nodes = get_mst(self.reduced_data[method],self.labels)
            self.plot_tree(ax3,method,'Recovered\nDistances After Reduction',mst,nodes)
            plotnum += 1
        plt.show()

    def plot_tree(self,ax,visualization_method,title,tree_matrix=None,nodes=None):
        """
        Plots a tree on a two-dimensional reduction of the data.

        Parameters
        ----------
        ax : matplotlib axis
            Axis on which to plot the tree
        visualization_method : string
            Which methods of dimensionality reduction to perform on the tree data
            to reduce to two dimensions.
            Options include ['pca','pca-34','pca-56','tsne','umap',
                            'Laplacian eigenmaps','Laplacian eigenmaps-34',
                            'Laplacian eigenmaps-56']
            PCA-34 returns the 3rd and 4th principal components, same for 56 and for
            Laplacian eigenmaps.
        title : string
            Title of plot
        tree_matrix : boolean ndarray, default=None
            If `None`, plots the true tree that generated the data. Otherwise, matrix
            should be True when there is an edge between nodes (i,j) and False otherwise.
        nodes : (N,2) ndarray
            The locations of the nodes in the tree. Used only if using the tree_matrix
            parameter.
        """
        for node in self.nodelist:
            node.plot(ax,tree=self,method=visualization_method)
        if tree_matrix is None:
            tree_matrix = self.tree_matrix
        if nodes is None:
            nodes = np.array([node.reduced_data[visualization_method].mean(axis=0)
                                for node in self.nodelist])
        plot_tree_from_matrix(tree_matrix,nodes,ax,title=title)

    def show_tree(self,tree_matrix=None,labels=None,title=None):
        """
        Shows a graphical representation of the tree.

        Parameters
        ----------
        tree_matrix : boolean ndarray, default=None
            If `None`, plots the true tree that generated the data. Otherwise, matrix
            should be True when there is an edge between nodes (i,j) and False otherwise.
        labels : list, default=None
            Labels of the nodes represented in the tree_matrix. If tree_matrix=None,
            the labels will be the node ids.
        title : string, default=None
            Title of plot
        """
        if tree_matrix is None:
            tree_matrix = self.tree_matrix
        if labels is None:
            labels = {i:node.id for i,node in enumerate(self.nodelist)}
        else:
            labels = {i:label for i,label in enumerate(labels)}
        colors = [self._colors[label] for label in labels]
        G = nx.convert_matrix.from_numpy_matrix(tree_matrix)
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G,pos,node_color=colors)
        nx.draw_networkx_edges(G,pos,edge_color='k')
        nx.draw_networkx_labels(G,pos,labels=labels)
        plt.title(title)
        plt.show()

def plot_tree_from_matrix(matrix,nodes,ax,title='Tree From Matrix'):
    """
    Plots a tree structure from a binary adjacency matrix and the locations of
    the nodes

    Parameters
    ----------
    matrix : (N,N) ndarray of type np.bool
        The binary adjacency matrix of the tree.
    nodes : (N,2) ndarray
        The locations of the nodes in the tree.
    ax : matplotlib axes
        Axes on which to plot the tree
    title : string
        title of plot

    Raises
    ------
    ValueError
        if the input matrix is not boolean or if the nodes are not a
        two-dimensional array.
    """
    if matrix.dtype != np.bool:
        raise ValueError('must have a boolean matrix')
    if nodes.shape[1] != 2:
        raise ValueError('nodes must be 2 dimensional')
    #iterating thru pairs of nodes is equivalent to iterating thru triu indices
    indices = np.triu_indices(len(nodes))
    for spot in zip(indices[0],indices[1]):
        #if there is an edge between nodes, plot a line between them
        if matrix[spot]:
            fromnode = nodes[spot[0]]
            tonode = nodes[spot[1]]
            ax.plot([fromnode[0],tonode[0]],
                    [fromnode[1],tonode[1]],
                         color='k',linewidth=1)
    ax.axis('equal')
    ax.set_title(title)

def get_mst(data,labels,weights='euclidean'):
    """
    Finds a minimum spanning tree of data which has already been clustered to
    particular nodes.

    Parameters
    ----------
    data : (N,M) np.array
        Data array of N points in M-dimensional space.
    labels : (N,) np.array
        Array of size N which contains the labels of each datapoint, i.e. the
        assignment to a particular node. Labels should start from 0 and go to K,
        the number of distinct labels.
    weights : string or function, default='euclidean'
        If 'eucilidean', the distances are the euclidean distances between the
        means of the data with each distinct label. Otherwise, must be a function
        such that weights(data,labels) returns a matrix where the (i,j) and (j,i)
        entries are the distances between the node labeled i and the node labeled
        j.

    Returns
    ------
    matrix : (N,M) np.array
        Data array of N points in M-dimensional space.
    nodes : (N,M) np.array
        Locations of the means of the mean of each cluster
    """
    #get list of unique label
    unique_labels = np.unique(labels)
    #locate the center of each cluster
    means = np.array([data[labels==group].mean(axis=0) for group in unique_labels])
    #pariwise distances between clusters
    if weights == 'euclidean':
        dists = squareform(pdist(means))
    else:
        dists = weights(data,labels)
    #get tree as a boolean array
    mst = minimum_spanning_tree(dists).toarray()
    mst = (mst+mst.T) > 0
    return mst, means

def get_posterity_means(node):
    """
    Returns the means of the node and all of its children

    Parameters
    ----------
    node : Node object
        Node we are interested in finding the mean of and its children.
    """
    means = [node.mean]
    for child in node.children:
        means.extend(get_posterity_means(child))
    return means

def _greedy_PCA_tree_recovery(data,center_tol,purity_tol):
    """
    Helper function for greedy_PCA_tree_recovery.
    """
    transformer = PCA(n_components=1)
    PC = transformer.fit_transform(data)[:,0]
    center = np.abs(PC) < center_tol
    node = np.mean(data[center],axis=0)
    left = PC > 0
    right = (~left)*(~center)
    left = left * (~center)
    if left.sum() < purity_tol and right.sum() < purity_tol:
        return np.mean(data,axis=0)
    else:
        return node, _greedy_PCA_tree_recovery(data[left],center_tol,purity_tol),\
                     _greedy_PCA_tree_recovery(data[right],center_tol,purity_tol)

def greedy_PCA_tree_recovery(data,center_tol=10,purity_tol=50):
    """
    Attempts to recover the tree-structure from data by recursively using the
    first prinicipal component to separate the data into three groups: data that
    whose PC is near zero and is associated with the current node, data whose
    PC is sufficiently positive to be classified to a right child, and data
    whose PC is sufficient negative to be classified to a left child.

    Parameters
    ----------
    data : ndarray
        Data with tree-structure.
    center_tol : float, default=10
        Distance from zero that defines the "near zero" group at each step.
    purity_tol : int, default=50
        Number of datapoints that constitute a "small enough" leaf node. If the
        number of datapoints in a node is less than purity_tol, it will not be split
        further.
    """
    res = _greedy_PCA_tree_recovery(data,center_tol,purity_tol)
    #extract nodes
    nodes = []
    queue = list(res)
    while len(queue) > 0:
        node = queue.pop(0)
        if isinstance(node,tuple):
            nodes.append(node[0])
            queue.extend([node[1],node[2]])
        else:
            nodes.append(node)
    #form matrix
    num_nodes = len(nodes)
    levels = int(np.log2(num_nodes+1))
    matrix = np.zeros([num_nodes]*2)
    child_num = 1
    for node in range(num_nodes):
        if child_num+2 > num_nodes:
            break
        else:
            matrix[node,child_num] = 1
            child_num += 1
            matrix[node,child_num] = 1
            child_num += 1
    matrix += matrix.T
    return matrix > 0,np.array(nodes)

def plot_greedy_PCA_tree_recovery(tree):
    """
    Plots the tree recovered using greedy_PCA_tree_recovery from the data. Uses
    tsne to visualize the data.

    Parameters
    ----------
    tree : Tree object
        Tree with associated data that we will attempt to use to recover the
        true structure.
    """
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    tree.plot_tree(ax1,'tsne','True Tree')
    ax2 = fig.add_subplot(122)
    matrix,nodes = greedy_PCA_tree_recovery(tree.data)
    nodeidx = np.array([np.argmin(la.norm(tree.data-node,axis=1)) for node in nodes])
    nodes = tree.reduced_data['tsne'][nodeidx]
    tree.plot_tree(ax2,'tsne','Greedy PCA Recovered Tree',matrix,nodes)
    plt.show()

def show_greedy_PCA_tree_recovery(tree):
    """
    Shows a graphical representation of the tree recovered using
    greedy_PCA_tree_recovery from the data.

    Parameters
    ----------
    tree : Tree object
        Tree with associated data that we will attempt to use to recover the
        true structure.
    """
    tree.show_tree(title='True Tree')
    matrix,nodes = greedy_PCA_tree_recovery(tree.data)
    nodeidx = np.array([np.argmin(la.norm(tree.data-node,axis=1)) for node in nodes])
    labels = tree.labels[nodeidx]
    tree.show_tree(tree_matrix=matrix,labels=labels,title='Greedy PCA Recovered Tree')

#todo update documentation
def gen_tree(treenum,levels,stdev,seed=0,N=100,dim=500,delta_t=1,start="zeros",
                plots=False,
                methods=['pca','tsne'],save=True,filename=None):
    """
    Generates a balanced, binary tree and associated data, and then saves the data
    to a csv file. If multiple seeds are provided, several datasets will be created.

    Parameters
    ----------
    levels : int
        The number of levels in the binary tree. For example, a binary tree with
        three nodes will have one level.
    seeds : iterable of ints
        Random seeds used to generate the data. If multiple seeds are provided,
        several datasets will be created.
    stdev : float
        The standard deviation used to generat the normal distributions around each node
    N : int
        The number of samples to draw from each node
    dim : int
        The dimensionality of the data
    delta_t : The variance of the normal distribution used to generate the displacement
        between parent and child nodes.
    start : "zeros", "ones", or ndarray of shape (dim,)
        The location of the root node. If "zeros" or "ones", the root node will be
        generated as np.zeros(dim) or np.ones(dim), respectively.
    """
    if start == "zeros":
        start_mean = np.zeros(dim)
    elif start == "ones":
        start_mean = np.ones(dim)
    else:
        start_mean = start_mean
    root = Node(start_mean,stdev=stdev,N=N,seed=seed)
    children = [root]
    for level in range(levels):
        new_children = []
        for child in children:
            child.add_child(delta_t,N=N)
            child.add_child(delta_t,N=N)
            new_children.extend(child.children)
        children = new_children
    t = Tree(root,methods=methods)

    #plots
    if plots:
        t.plot_summary()
        plot_greedy_PCA_tree_recovery(t)
        show_greedy_PCA_tree_recovery(t)

        m,n = t.data.shape
        print(m,n)
        u,s,vh = la.svd(t.data/np.sqrt(n))
        plt.figure(figsize=(10,5))
        plt.hist(s,bins=500,density=False)
        lambdamax = stdev*(1+np.sqrt(m/n))
        plt.vlines(lambdamax,0,25)
        plt.show()
        print(lambdamax,sum(s>lambdamax))
    t.savecsv(filename)
    return t

if __name__ == "__main__":
    N = 100
    dim = 500
    delta_t = 1
    seeds = [1]
    treenum = 0
    methods = ['pca','tsne']
    for seed in seeds:
        for levels in [2,3,4]:
            for stdev in [1,4]:
                for start in ["zeros","ones"]:
                    if start == "ones" and stdev == 4:
                        continue
                    else:
                        treenum += 1
                        print(treenum)
                        filename = f"NodeTree{treenum}"
                        gen_tree(treenum,
                                    levels=levels,
                                    stdev=stdev,
                                    seed=seed,
                                    N=N,
                                    dim=dim,
                                    delta_t=delta_t,
                                    start=start,
                                    plots=False,
                                    methods=methods,
                                    save=True,
                                    filename=f"{filename}/{filename}.csv")
                        with open(f"{filename}/info.txt",'w') as f:
                            f.write(f"levels {levels}\nstdev {stdev}\nseed {seed}\nN {N}\ndim {dim}\ndeltat {delta_t}\nstart {start} vector\nmethods {methods}")
