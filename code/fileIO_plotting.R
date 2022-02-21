# module for reading and formating the data
# and plotting the dimnensionality reduced data with certain colors
library(RColorBrewer)
library(scales)

#' Forms a tree object from a csv file
#'
#' @param filename string containing the path to the csv file
#' @returns a tree vector with the following attributes.
#' csv: the data from the csv file
#' raw: selects columns Raw0:Raw499 in order to avoid the dimensionality-reduced data
#' matrix: the raw data cast to a matrix
#' dimred: the tsne dimensionality reducition from the csv file
#' counts: the "raw counts" matrix, which is computed as min(0,round(2**(tree$matrix)-1))
#' dataset: counts and matrix wraped for dynverse functions, with prior information that Row0 is the starting cell
#' trajectory: an empty vector for saving trajectory results
form_tree_from_file <- function(filename){
    tree <- vector(mode="list")
    tree$csv <- read.csv(filename,row.names=1)
    tree$labels <- tree$csv$Labels
    tree$matrix <- tree$csv %>%
        select(Raw0:Raw499)
    tree$matrix <- as.matrix(tree$matrix)
    tree$dimred <- tree$csv %>%
        select(tsne0:tsne1)
    #As input, dynwrap requires raw counts and normalized (log2) expression data.
    tree$counts <- round(2**(tree$matrix)-1)
    tree$counts[tree$counts<0] <- 0
    tree$dataset <- wrap_expression(
        expression = tree$matrix,
        counts = as.matrix(tree$counts)
    )
    tree$dataset <- add_prior_information(
        tree$dataset,
        start_id = "Row0"
    )
    tree$trajectory <- vector(mode="list")
    return(tree)
}

#' Plots the dimensionality reduction of a nodetree along with arrows along the branches.
#'
#' @param tree tree object, like the output of form_tree_from_file
#' @param color vector of colors for the data points
#' @param palette color pallete to use for the plot
plot_nodetree <- function(tree,color=1+tree$labels,palette=c("#0000D2","#D2D2D2","#D20000")){
    #create colormap
    ## create color palette function
    pal <- colorRamp(palette)
    ## rescale color to be between 0 and 1, and 0.5 being the "zero" point
    newcolor <- color/max(abs(color)) / 2 + 0.5
    ## use pal and alpha to get rgb value for color
    newcolor <- alpha(rgb(pal(newcolor)/255),0.4)
    ## make colored scatter plot
    plot(tree$dimred,col=newcolor,pch=20)

    # draw arrows between parents and children
    ## get all the different labels of the nodes
    labels <- unique(tree$labels)
    ## get the nodes that are parents
    parents <- labels[1:(length(labels)/2)]
    ## initialize index of child and num children per node
    childidx <- 2
    children_per_node <- 2
    ## draw arrows
    for (parent in parents){
        for (childnum in 1:children_per_node){
            child <- labels[childidx]
            arrows(mean(tree$dimred$tsne0[tree$labels == parent]),
                   mean(tree$dimred$tsne1[tree$labels == parent]),
                   mean(tree$dimred$tsne0[tree$labels == child]),
                   mean(tree$dimred$tsne1[tree$labels == child]),
                   length=0.05)
            childidx <- childidx + 1
        }
    }
}

#' Plots the dimensionality reduction of a continuous tree along with arrows along the branches.
#'
#' @param tree tree object, like the output of form_tree_from_file
#' @param color vector of colors for the data points
#' @param palette color pallete to use for the plot
plot_continuoustree <- function(tree,color=tree$labels,palette=c("#0000D2","#D2D2D2","#D20000")){
    #create colormap
    ## create color palette function
    pal <- colorRamp(palette)
    ## rescale color to be between 0 and 1, and 0.5 being the "zero" point
    newcolor <- color/max(abs(color)) / 2 + 0.5
    ## use pal and alpha to get rgb value for color
    newcolor <- alpha(rgb(pal(newcolor)/255),0.4)
    ## make colored scatter plot
    plot(tree$dimred,col=newcolor,pch=20)

    # draw arrows along branches
    ## get all the different labels of the nodes
    labels <- unique(tree$labels)
    ## draw arrows
    for (label in labels){
        group <- tree$dimred[tree$labels == label,]
        first <- head(group,1)
        last <- tail(group,1)
        arrows(first$tsne0,first$tsne1,
               last$tsne0,last$tsne1,
               length=0.05)
    }
}

#' For each loading, plots the dimensionality reduction of a nodetree
#' along with arrows along the branches, with data points colored according
#' to their value for the loading
#'
#' Also creates a second plot which shows the values of each loading,
#' with the data colored according to the grouping
#'
#' @param tree tree object, like the output of form_tree_from_file
#' @param loading posterior mean of loadings, used to color the tree
#' @param color colors for each data grouping to visualize the
plot_nodetree_loadings <- function(nodetree,loadings,color){
    numloadings <- dim(loadings)[2]
    #dim reduction colored according to loading
    par(mfrow=c(ceiling(numloadings/3), 3),mar=c(1,1,1,1))
    for (loadingnum in 1:numloadings){
        #uses function from fileIO_plotting.R to produce plot
        # with arrows between groups
        plot_nodetree(nodetree,color=loadings[,loadingnum])
    }
    #values of the loadings
    par(mfrow=c(ceiling(numloadings/3), 3),mar=c(1,1,1,1))
    for (loadingnum in 1:numloadings){
        plot(loadings[,loadingnum],col=color)
    }
}
