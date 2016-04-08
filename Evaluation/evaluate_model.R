suppressPackageStartupMessages(library("randomForest"))
library("Rtsne")

# ## Function definitions
# 
# sgm <- function(x){
#   # Sigmoid function
#     return(1/(1+exp(-x)))
# }
# 
# get_activations <- function(exp_data, w, b){
#   # Propagate inputs through to the hidden layer
#   # Linear transform
#   lin <- t(w) %*% as.matrix(exp_data)
#   # Add bias (a bit ugly)
#   bia <- lin
#   for(i in 1:nrow(lin)){
#     bia[i,] <- lin[i,] + b[i,] 
#   }
#   act <- t(sgm(bia))
#   return(act)
# }

node.act.per.type <- function(act, node, m){
    lev <- levels(coi)
    boxes <- NULL
    for (ctype in lev){
        box <- t(act[which(m==ctype), node])
        boxes[[ctype]] <- box
    }
    boxplot(boxes, las=2, main=paste("Node", node))
}

type.act.per.node <- function(act, m, filename){
    par(mfrow=c(4,1))
    for(cell in levels(coi)){
        boxplot(act[which(coi==cell),],main=cell,las=2,names=paste0("Node",1:ncol(act)))
    }
    par(mfrow=c(1,1))
}

# Propagate activity through the network
# Activation of visible layer is the actual expression data
# act <- t(exp_data)
# for(i in 1:numLayers){
#   # Read weights and bias for the layer in question
#   print(paste("Reading weights for layer", i))
#   w <- read.delim(args[2*i+1],header=FALSE)
#   print(paste("Reading biases for layer", i))
#   b <- read.delim(args[2*i+2],header=FALSE)
#   act <- get_activations(t(act), w, b)
# }

# # Define colors and such for the metadata
def_colors <- function(meta){
    # Make the 1st column index column (rownames in R), and remove it 
    rownames(meta) <- meta[, colnames(meta)[1]] 
    meta[, colnames(meta)[1]] <- NULL
    
    # Now 1st column is the former 2nd column. So we use this to take tha names
    typeNames <- levels(meta[, colnames(meta)[1]])
    typeColors <<- rainbow(length(typeNames))
    names(typeColors) <<- typeNames
    
    # Take the column of interest (coi) and assign example names to the labels
    coi <<- meta[, colnames(meta)[1]]
    names(coi) <<- rownames(meta)
}

# # Handle several analysis functions 
do_analysis <- function(act, outfile_pref){
    col_name <- colnames(act)[1]
    rownames(act) <- act[,col_name]
    act[, col_name] <- NULL
    
    nondup <- act[which(!duplicated(act)),]
    colrs <<- typeColors[coi[rownames(nondup)]]
    
    plot_pca(nondup, outfile_pref)
    plot_tsne(nondup, outfile_pref)
    node_profiles(act, outfile_pref)
    cell_profiles(act, outfile_pref)
    calc_rf(act)
}

# # PCA on activations
plot_pca <- function(act, outfile_pref){
    pcafile <- paste(outfile_pref, "PCA.pdf", sep="_")

    p <- prcomp(act)

    pdf(file=pcafile, paper="a4r")
    par(mfrow=c(1,2))
    plot(p$x, col=colrs, pch=20)
    plot(p$x[,2:3], col=colrs,pch=20)
    dev.off()
}

# # Rtsne
plot_tsne <- function(act, outfile_pref){
    tsnefile <- paste(outfile_pref, "tSNE.pdf", sep="_")
    
#     nondup <- act[which(!duplicated(act)),]
    r <- Rtsne(act, perplexity=10)

    pdf(file=tsnefile, paper="a4")
    plot(r$Y, pch=20, col=colrs)
    dev.off()
}

# # Look at the nodes in order of decreasing standard deviation
node_profiles <- function(act, outfile_pref){
    filename <- paste(outfile_pref, "node_profiles.pdf", sep="_")

    pdf(filename, paper="a4")
    par(mfrow=c(2,4))
    for(node in order(apply(act, 2, sd),decreasing=TRUE)){
        node.act.per.type(act, node, coi)
    }
    dev.off()
}

# # Or per cell type
cell_profiles <- function(act, outfile_pref){
    filename <- paste(outfile_pref, "cell_profiles.pdf", sep="_")

    pdf(filename, paper="a4")
    type.act.per.node(act, coi)
    dev.off()
}

# # Check predictivity
calc_rf <- function(act){
    rf <- randomForest(x=act, y=as.factor(coi), importance=TRUE)
    print(paste("RF estimated error rate", tail(rf$err.rate, n=1)[,1], sep=":"))
}
