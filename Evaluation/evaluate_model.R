suppressPackageStartupMessages(library("randomForest"))
library("Rtsne")

# # Function definitions

sgm <- function(x){
    # Sigmoid function
    return(1/(1+exp(-x)))
}

get_activations <- function(exp_data, w, b){
    # Propagate inputs through to the hidden layer
    # Linear transform
    print(dim(w))
    print(dim(exp_data))
    lin <- t(w) %*% as.matrix(exp_data)
    # Add bias (a bit ugly)
    bia <- lin
    for(i in 1:nrow(lin)){
        bia[i,] <- lin[i,] + b[[i]]
    }
    act <- t(sgm(bia))
    return(act)
}

node.act.per.type <- function(act, node, m){
    lev <- levels(coi)
    boxes <- NULL
    for (ctype in lev){
        box <- t(act[which(m==ctype), node])
        boxes[[ctype]] <- box
    }
    boxplot(boxes, las=2, main=paste("Node", node), ylim=c(0,1))
}

type.act.per.node <- function(act, m, filename){
    par(mfcol=c(3,1))
    for(cell in levels(coi)){
        boxplot(act[which(coi==cell),], main=cell, las=2, names=paste0("Node",1:ncol(act)), ylim=c(0,1))
    }
    par(mfrow=c(1,1))
}

# # Define colors and such for the metadata
def_colors <- function(meta){
#     print(meta)
    # Now 1st column is the former 2nd column. So we use this to take tha names
    typeNames <<- levels(meta[, colnames(meta)[1]])
#     print(typeNames)

## COLORS :  red=552, blue=26, black=24, green=254, yellow=652 --> change-to yellow2=654
## COLORS :  orange=498 --> change-to darkorange1=91, brown=32 --> change-to chocolate4=56
## COLORS :  purple=547, grey39=300, violetred=641, darkgreen=81, cyan=68, magenta=450
## COLORS :  goldenrod4=151, hotpink=367, darkolivegreen2=87, midnightblue=477, lightcoral=404
## COLORS :  darkslategrey=113, 

    distinct_color_pool <- c("red","blue","black","green","yellow2","darkorange1",
                            "chocolate4","purple","grey39","violetred","darkgreen",
                            "cyan","magenta","goldenrod4","hotpink","darkolivegreen2",
                            "midnightblue","midnightblue","darkslategrey")
#     typeColors <<- rainbow(length(typeNames))
    typeColors <<- distinct_color_pool[1:length(typeNames)]
#     print(typeColors)
    names(typeColors) <<- typeNames
#     print(typeColors)
    
    # Take the column of interest (coi) and assign example names to the labels
    coi <<- meta[, colnames(meta)[1]]
#     print(coi)
    names(coi) <<- 1:nrow(meta)
#     print(coi)
}

# # Handle several analysis functions
do_analysis <- function(act, w, b, outfile_pref, bias_node=FALSE){
    for(i in 1:length(w)){
        if(bias_node == TRUE){
            act <- cbind(rep(1, nrow(act)), act)
        }
        act <- get_activations(t(act), w[[i]], b[[i]])
    #     print(act)
        nondup <- act[which(!duplicated(act)),]
        print(dim(act))
        print(dim(nondup))
        
        colrs <- typeColors[coi[1:nrow(act)]]
        plot_pca(nondup, colrs, paste(outfile_pref, i, sep='_'))
        
        colrs <- typeColors[coi[1:nrow(nondup)]]
        plot_tsne(nondup, colrs, paste(outfile_pref, i, sep='_'))

        node_profiles(act, paste(outfile_pref, i, sep='_'))
        cell_profiles(act, paste(outfile_pref, i, sep='_'))
        calc_rf(act)
    }
}

# # PCA on activations
plot_pca <- function(act, colrs, outfile_pref){
    pcafile <- paste(outfile_pref, "PCA.pdf", sep="_")

    p <- prcomp(act)

    pdf(file=pcafile, paper="a4r")
    # par mar(Bottom, Left, Top, Right)
    layout(matrix(c(1,2,3,3), ncol=2, byrow=TRUE), heights=c(4, 1))
    plot(p$x, col=colrs, pch=20)
    plot(p$x[,2:3], col=colrs, pch=20)
    par(mai=c(0,0,0,0))
    plot.new()
    legend("center", bty="n", legend=names(typeColors), col=typeColors, pch=rep(20,length(typeColors)), ncol=as.integer((length(typeColors)/10)+0.5), cex=0.8, pt.cex=0.8)
    dev.off()
}

# # Rtsne
plot_tsne <- function(act, colrs, outfile_pref){
    tsnefile <- paste(outfile_pref, "tSNE.pdf", sep="_")
    
#     nondup <- act[which(!duplicated(act)),]
    r <- Rtsne(act, perplexity=10)

    pdf(file=tsnefile, paper="a4r")
    layout(matrix(c(1,2), ncol=1), heights=c(4, 1))
    plot(r$Y, pch=20, col=colrs, xlab="", ylab="")
    par(mai=c(0,0,0,0))
    plot.new()
    legend("center", bty="n", legend=names(typeColors), col=typeColors, pch=rep(20,length(typeColors)), ncol=as.integer((length(typeColors)/10)+0.5), cex=0.7, pt.cex=0.7)
    dev.off()
}

# # Look at the nodes in order of decreasing standard deviation
node_profiles <- function(act, outfile_pref){
    filename <- paste(outfile_pref, "node_profiles.pdf", sep="_")

    pdf(filename, paper="a4")
    layout(matrix(c(1,2,3), nrow=1, ncol=3,byrow=TRUE))
    par(mar=c(15.0, 2.3, 2.6, 2.1))

    for(node in order(apply(act, 2, sd),decreasing=TRUE)){
        node.act.per.type(act, node, coi)
    }
    dev.off()
}

# # Or per cell type
cell_profiles <- function(act, outfile_pref){
    filename <- paste(outfile_pref, "cell_profiles.pdf", sep="_")

    pdf(filename, paper="a4")
    par(mar=c(4.5, 2.3, 1.7, 0.1))
    type.act.per.node(act, coi)
    dev.off()
}

# # Check predictivity
calc_rf <- function(act){
    rf <- randomForest(x=act, y=as.factor(coi), importance=TRUE)
    print(paste("RF estimated error rate", tail(rf$err.rate, n=1)[,1], sep=":"))
}
