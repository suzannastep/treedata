import numpy as np
import networkx as nx
import pandas as pd
from matplotlib import pyplot as plt
from itertools import combinations,combinations_with_replacement
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import SpectralEmbedding
from umap import UMAP

#todo update documentation

class Node:
    """
    Node/branch point in abstract tree structure that can be used to generate data.

    Parameters
    ----------
    delta_t : float
        The amount of pseodotime between a node and its parent. For a root node,
        this is the amount of "pseodotime" between the root node and some hypothetical,
        original drift when a true root node was at the origin.
    parent : Node default=None
        Parent node of the current node. Defaults to None for a root node.


    Attributes
    ----------
    delta_t : float
        The amount of pseodotime between a node and its parent. For a root node,
        this is the amount of "pseodotime" between the root node and some hypothetical,
        original drift when a true root node was at the origin.
    parent : Node default=None
        Parent node of the current node. Defaults to None for a root node.
    children : list of Nodes
        Children of the current node.
    level: int
        The number of nodes in the path between the current node and the root node.
    id : int
        An id number associated to a node. Each time a new root node is created
        (i.e. parent is None), the id counter resets to zero.
    pseudotime : float
        The amount of pseodotime between a node and the origin. Specifically,
        the sum of the delta_t values of a node and all of its ancestor nodes.
    numdatapoints : int
        The number of data points along the edge between the node and its parent.
        Only defined after a Tree object is created and generate_covariance is run.
    delta_ts : (numdatapoints,) ndarray of floats
        The amount of pseodotime between the parent node and the data points
        along the edge between the node and its parent. Only defined after a
        Tree object is created and generate_covariance is run.
    pseudotimes : (numdatapoints,) ndarray of floats
        The amount of pseodotime between associtaed data points and the origin.
        Only defined after a Tree object is created and generate_covariance is run.
    start_idx : int
        The index into the data associated with a tree of when the data associated
        with a particular node begins. Only defined after a Tree object is created
        and generate_covariance is run.
    end_idx : int
        The index into the data associated with a tree of when the data associated
        with a particular node ends. Only defined after a Tree object is created
        and generate_covariance is run.
    """
    _numnodes = 0
    def __init__(self,delta_t=0,parent=None):
        self.delta_t = delta_t
        self.parent = parent
        self.children = []
        if self.parent is None:
            self.level = 0
            Node._numnodes = 0
            self.id = 0
            self.pseudotime = delta_t
        else:
            self.level = self.parent.level + 1
            self.id = Node._numnodes
            self.pseudotime = self.parent.pseudotime + delta_t
        Node._numnodes += 1

    def add_child(self,delta_t):
        """
        Add a child to a node.

        Parameters
        ----------
        delta_t : float
            The amount of pseodotime between the node and the newly created child.
        """
        child = Node(delta_t,parent=self)
        self.children.append(child)

    def __str__(self):
        return '-'*self.level + 'Node ' + str(self.id)

    def posterity_string(self):
        """
        Returns string that, when printed, reports the posterity of a node.
        The format is as follows. Each node is a new line, and the number of `-`
        preceeding a node's id number is its level. A child node will be printed
        below its parents. For example, a simple binary tree with two levels would
        be printed as follows

        Node 0
        -Node 1
        --Node 3
        --Node 4
        -Node 2
        --Node 5
        --Node 6

        Parameters
        ----------
        delta_t : float
            The amount of pseodotime between the node and the newly created child.
        """
        str = self.__str__()
        for child in self.children:
            str += '\n' + child.posterity_string()
        return str

class Tree:
    """
    Abstract tree object that can be used to generate data.

    Parameters
    ----------
    root : Node
        Root node of the tree, with children already added. A tree should not
        be added until all of the descendants have been added to via the 'add_child'
        function of the Node object.

    Attributes
    ----------
    root : Node
        Root node of the tree.
    nodelist : list of Node objects
        A list of nodes in the tree, ordered by id number
    matrix : ndarray (N,N) where N is the number of nodes
        Adjacency matrix of the tree. Weights between nodes is the psuedotime
        elapsed between them.
    """
    def __init__(self,root):
        self.root = root
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
        #get true tree matrix
        self._numnodes = len(self.nodelist)
        self.matrix = np.zeros([self._numnodes]*2)
        indices = np.triu_indices(self._numnodes)
        for node in self.nodelist:
            for child in node.children:
                self.matrix[node.id,child.id] = child.delta_t
                self.matrix[child.id,node.id] = child.delta_t
        #colors and labels for graph plotting
        _cm = plt.get_cmap('rainbow')
        plt.set_cmap('rainbow')
        self._colors = [_cm(node.id/self._numnodes)
                                for node in self.nodelist]
        self._ids = {node.id:node.id
                                for node in self.nodelist}
        #initialize covariance and data as Nonetypes
        self._cov = None
        self._data = None
        self._reduced_data = {}

    def __str__(self):
        return self.root.posterity_string()

    def get_covariance(self):
        """
        Returns the covariance matrix of the data associated with the tree, if it
        has already been generated
        """
        if self._cov is None:
            raise ValueError("Covariance is not yet defined. Must generate covariance first.")
        return self._cov

    def get_data(self):
        """
        Returns the data associated with the tree, if it
        has already been generated
        """
        if self._data is None:
            raise ValueError("Data is not yet defined. Must generate data first.")
        return self._data

    def get_reduced_data(self,method):
        """
        Returns the dimensionality-reduced data associated with the tree, if it
        has already been reduced using the specified method.

        Parameters
        ----------
        method : {'pca', 'tsne', 'umap', 'Laplacian eigenmaps'}
            Method used to reduce the data.
        """
        if not method in self._reduced_data.keys():
            raise ValueError(f"Data has not yet been reduced using method '{method}' yet. Must reduce data first.")
        return self._reduced_data[method]

    def generate_covariance(self,N,seed):
        """
        Generates and returns the covariance matrix associated with the data from
        the tree.

        Parameters
        ----------
        N : int
            Number of desired datapoints to generate. To ensure equal-spacing of points
            along the pseudotime, slightly fewer points may actually be generated.
        seed : int
            Random seed used to generate the covariance matrix.
        """
        np.random.seed(seed)
        #generate pseudotimes
        total_pseudotime = sum([node.delta_t for node in self.nodelist])
        points_per_unit_time = N/total_pseudotime
        idx = 0
        for node in self.nodelist:
            node.numdatapoints = int(points_per_unit_time*node.delta_t)
            node.delta_ts = np.random.uniform(0,node.delta_t,node.numdatapoints)
            node.delta_ts = np.sort(node.delta_ts)
            if node.parent is None:
                node.pseudotimes = node.delta_ts
            else:
                node.pseudotimes = node.parent.pseudotime + node.delta_ts
            node.start_idx = idx
            idx +=  node.numdatapoints
            node.end_idx = idx
        N = sum([node.numdatapoints for node in self.nodelist])
        #form covariance matrix
        cov = np.zeros((N,N))
        ## form diagonal first
        for node in self.nodelist:
            for i,j in combinations_with_replacement(range(node.numdatapoints),2):
                cov[node.start_idx + i,node.start_idx + j] = min(node.pseudotimes[i],node.pseudotimes[j])
        for node1,node2 in combinations(self.nodelist,2):
            #case 1: node1 *is* an ancestor of node2
            path2 = []
            curr = node2.parent
            while curr is not None:
                path2.append(curr)
                curr = curr.parent
            if node1 in path2:
                cov[node2.start_idx:node2.end_idx,node1.start_idx:node1.end_idx] = node1.pseudotimes
            else:
                #case 2: node2 *is* the ancestor of node1
                path1 = []
                curr = node1.parent
                while curr is not None:
                    path1.append(curr)
                    curr = curr.parent
                if node2 in path1:
                    cov[node1.start_idx:node1.end_idx,node2.start_idx:node2.end_idx] = node2.pseudotimes
                else:
                    #case 3: still need to find common ancestors
                    for ancestor1,ancestor2 in zip(path1[::-1],path2[::-1]):
                        if ancestor1 != ancestor2:
                            common_ancestor = ancestor1.parent
                            break
                    else:
                        common_ancestor = ancestor1

                    cov[node1.start_idx:node1.end_idx,node2.start_idx:node2.end_idx]  = common_ancestor.pseudotime

        cov += cov.T
        cov[np.diag_indices(N)] /= 2
        self._cov = cov
        return cov

    def generate_data(self,dim,seed):
        """
        Generates and returns data associated with the tree. Must create a covariance matrix
        first using generate_covariance.

        Parameters
        ----------
        dim : int
            Dimensionality of the data. Must be at least 2.
        seed : int
            Random seed used to generate the data.
        """
        if self._cov is None:
            raise ValueError("Covariance must be generated first")
        np.random.seed(seed)
        N = self._cov.shape[0]
        self._data = np.random.multivariate_normal(np.zeros(N),
                                                    cov=self._cov,
                                                    size=dim)
        self._data = self._data.T
        self._labels = np.zeros(N)
        for node in self.nodelist:
            self._labels[node.start_idx:node.end_idx] = node.id
        return self._data

    def reduce_data(self,newdim=2,method='pca'):
        """
        Reduces the dimensionality of the data associated with the tree for visualization
        and algorithmic purposes. Must create a covariance matrix and data using
        first using `generate_covariance` and `generate_data`.

        Parameters
        ----------
        dim : int
            Dimensionality of the data. Must be at least 2.
        seed : int
            Random seed used to generate the data.
        """
        if self._data is None:
            raise ValueError("Data must be generated first")
        print(method,'start')
        if method=='pca':
            reducer = PCA(n_components=newdim)
        elif method=='pca-34':
            reducer = PCA(n_components=4)
            #save the reduced data to the tree
            self._reduced_data[method] = reducer.fit_transform(self._data)[:,2:4]
        elif method=='pca-56':
            reducer = PCA(n_components=6)
            #save the reduced data to the tree
            self._reduced_data[method] = reducer.fit_transform(self._data)[:,4:6]
        elif method=='tsne':
            reducer = TSNE(n_components=newdim,n_iter=500,init='pca',learning_rate=1000)
        elif method=='umap':
            reducer = UMAP(n_components=newdim)
        elif method=='Laplacian eigenmaps':
            reducer = SpectralEmbedding(n_components=newdim)
        elif method=='Laplacian eigenmaps-34':
            reducer = SpectralEmbedding(n_components=4)
            #save the reduced data to the tree
            self._reduced_data[method] = reducer.fit_transform(self._data)[:,2:4]
        elif method=='Laplacian eigenmaps-56':
            reducer = SpectralEmbedding(n_components=6)
            #save the reduced data to the tree
            self._reduced_data[method] = reducer.fit_transform(self._data)[:,4:6]
        else:
            raise ValueError('method must be one of {pca,tsne,umap}')
        #save reduced data to the tree
        if method not in {'pca-34','pca-56','Laplacian eigenmaps-34','Laplacian eigenmaps-56'}:
            self._reduced_data[method] = reducer.fit_transform(self._data)
        print(method,'done')
        return self._reduced_data[method]

    def save_data(self,filename):
        """
        Saves the data and reduced data from a tree as csv files. Columns will appear as
        either "raw" or the name of the reduction method, followed by the component number

        Parameters
        ----------
        filename : str
            Name of the file, without the csv file extension.
        """
        if self._data is None:
            raise ValueError("Data must be generated first")
        df = pd.DataFrame(self._data,columns=[f'Raw{num}'for num in range(self._data.shape[1])],
                            index = [f'Row{rownum}' for rownum in np.arange(self._data.shape[0])])
        tmp_df = pd.DataFrame(self._labels,columns=['Labels'],
                            index = [f'Row{rownum}' for rownum in np.arange(self._data.shape[0])])
        df = tmp_df.join(df)

        #save reduced data
        for method in self._reduced_data.keys():
            tmp_df = pd.DataFrame(self._reduced_data[method],columns=[method+str(num)
                                    for num in range(self._reduced_data[method].shape[1])],
                                index = [f'Row{rownum}' for rownum in np.arange(self._data.shape[0])])
            df = df.join(tmp_df)
        df.to_csv(f'data/ContinuousTrees/{filename}/{filename}.csv',index=True)

    def show(self,title=None):
        """
        Shows the graphical structure of the tree using the `networkx` package.

        Parameters
        ----------
        title : string
            Title to add to the plot
        """
        G = nx.convert_matrix.from_numpy_matrix(self.matrix)
        pos = nx.spring_layout(G)
        nx.draw_networkx_nodes(G,pos,node_color=self._colors)
        nx.draw_networkx_edges(G,pos,edge_color='k')
        nx.draw_networkx_labels(G,pos,labels=self._ids)
        plt.title(title)

    def plot_reduced_data(self,title,method,ax,colors=None,cmap=None):
        """
        Plots the reduced data from a tree using all available visualization method.

        Parameters
        ----------
        filename : str
            Name of the file, without the csv file extension.
        """
        if method not in self._reduced_data.keys():
            raise ValueError(f"Must reduce using {method} first.")
        data = self._reduced_data[method]
        if colors is None:
            colors = self._labels
        if cmap is not None:
            plt.set_cmap(cmap)
        _max = np.max(np.abs(colors))
        ax.scatter(data[:,0],data[:,1],
                    c = colors,
                    s=3,
                    alpha=.4,
                    vmin = -_max,
                    vmax = _max,
                    marker = '.')
        ax.set_xlabel('component 1')
        ax.set_ylabel('component 2')
        ax.set_title(title)
        ax.axis('equal')

    def show_summary(self,num,method=None):
        #print basic structure of tree
        print(self)
        print('Num Nodes',self._numnodes)
        print('Tree Matrix\n',self.matrix)
        print('\nid\tdeltat\tpseudotime')
        for node in self.nodelist:
            print(node.id,'\t',node.delta_t,'\t',node.pseudotime)
        #show tree graph
        plt.subplot(131)
        self.show(f'tree {num}')
        #plot covariances
        plt.subplot(132)
        plt.imshow(self.get_covariance())
        plt.title(f'tree {num} covariance')
        plt.colorbar()
        # plt.imshow(np.cov(self.get_data()))
        # plt.colorbar()
        # plt.title(f'tree {num} numerical covariance')
        ax = plt.subplot(133)
        self.plot_reduced_data(f'tree {num} '+method,method=method,ax=ax)
        plt.tight_layout()
        plt.show()

def save_tree1():
    #tree 1
    N = 700
    dim = 500
    #create nodes
    root = Node(0)
    root.add_child(1)
    root.add_child(1)
    for child in root.children:
        child.add_child(1)
        child.add_child(1)
    #form, save and return tree
    tree = Tree(root)
    #generate data
    tree.generate_covariance(N=N,seed=142)
    tree.generate_data(dim=dim,seed=67)
    #reduce data
    tree.reduce_data(method='pca')
    tree.reduce_data(method='tsne')
    # tree.reduce_data(method='umap')
    # tree.reduce_data(method='Laplacian eigenmaps')
    tree.save_data('tree1')
    return tree

def save_tree2():
    #tree 2
    N = 700
    dim = 500
    #create nodes
    root = Node(1)
    root.add_child(1)
    root.add_child(1)
    for child in root.children:
        child.add_child(1)
        child.add_child(1)
    #form, save and return tree
    tree = Tree(root)
    #generate data
    tree.generate_covariance(N=N,seed=142)
    tree.generate_data(dim=dim,seed=67)
    #reduce data
    tree.reduce_data(method='pca')
    tree.reduce_data(method='tsne')
    # tree.reduce_data(method='umap')
    # tree.reduce_data(method='Laplacian eigenmaps')
    tree.save_data('tree2')
    return tree

def save_tree3():
    #tree 3
    N = 700
    dim = 500
    #create nodes
    root = Node(1)
    root.add_child(1)
    root.add_child(1)
    for child in root.children:
        child.add_child(1)
        child.add_child(1)
        for grandchild in child.children:
            grandchild.add_child(1)
            grandchild.add_child(1)
    #form, save and return tree
    tree = Tree(root)
    #generate data
    tree.generate_covariance(N=N,seed=142)
    tree.generate_data(dim=dim,seed=67)
    #reduce data
    tree.reduce_data(method='pca')
    tree.reduce_data(method='tsne')
    # tree.reduce_data(method='umap')
    # tree.reduce_data(method='Laplacian eigenmaps')
    tree.save_data('tree3')
    return tree

def save_tree4():
    #tree 4
    N = 700
    dim = 500
    #create nodes
    root = Node(.37)
    root.add_child(.11)
    root.add_child(.79)
    root.children[0].add_child(.31)
    root.children[0].add_child(.59)
    root.children[1].add_child(.38)
    root.children[1].add_child(.19)
    #form, save and return tree
    tree = Tree(root)
    #generate data
    tree.generate_covariance(N=N,seed=142)
    tree.generate_data(dim=dim,seed=67)
    #reduce data
    tree.reduce_data(method='pca')
    tree.reduce_data(method='tsne')
    # tree.reduce_data(method='umap')
    # tree.reduce_data(method='Laplacian eigenmaps')
    tree.save_data('tree4')
    return tree

if __name__ is "__main__":
    tree1 = save_tree1()
    tree2 = save_tree2()
    tree3 = save_tree3()
    tree4 = save_tree4()
