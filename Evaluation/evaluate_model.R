suppressPackageStartupMessages(library("randomForest"))
library("Rtsne")

## Function definitions

sgm <- function(x){
  # Sigmoid function
    return(1/(1+exp(-x)))
}

get_activations <- function(exp_data, w, b){
  # Propagate inputs through to the hidden layer
  # Linear transform
  lin <- t(w) %*% as.matrix(exp_data)
  # Add bias (a bit ugly)
  bia <- lin
  for(i in 1:nrow(lin)){
    bia[i,] <- lin[i,] + b[i,] 
  }
  act <- t(sgm(bia))
  return(act)
}

node.act.per.type <- function(act, node, m){
  shortNames <- c("Astro","Endo","GABA","Glut","Microglia","Oligo","OligoPC","Uncl")
  boxplot(act[which(m=="Astrocyte"),node], act[which(m=="Endothelial Cell"),node],
          act[which(m=="GABA-ergic Neuron"),node],act[which(m=="Glutamatergic Neuron"),node],
          act[which(m=="Microglia"),node], act[which(m=="Oligodendrocyte"),node],
          act[which(m=="Oligodendrocyte Precursor Cell"),node], act[which(m=="Unclassified"),node],
          names=shortNames,main=paste("Node",node),las=2,cex=0.5)
}

type.act.per.node <- function(act, m){
  par(mfrow=c(4,2))
  for(cell in levels(btype)){
    boxplot(act[which(btype==cell),],main=cell,las=2,names=paste0("Node",1:ncol(act)))
  }
  par(mfrow=c(1,1))
}


args <- commandArgs(trailingOnly = TRUE)
numLayers <- (length(args) - 2)/2
print(paste("Number of layers:", numLayers))

# Read expression data. (Currently used only to get gene names.)
print("Reading expression data...")
exp_data <- read.delim(args[1],check.names=FALSE,row.names=1)
# Read metadata (clustering results)
print("Reading metadata...")
meta <- read.delim(args[2],check.names=FALSE,row.names=1)
# Check for same ordering
stopifnot(identical(colnames(exp_data), rownames(meta)))

# Propagate activity through the network
# Activation of visible layer is the actual expression data
act <- t(exp_data)
for(i in 1:numLayers){
  # Read weights and bias for the layer in question
  print(paste("Reading weights for layer", i))
  w <- read.delim(args[2*i+1],header=FALSE)
  print(paste("Reading biases for layer", i))
  b <- read.delim(args[2*i+2],header=FALSE)
  act <- get_activations(t(act), w, b)
}

# Define colors and such for the metadata
typeNames <- levels(meta$broad_type)
typeCols <- c("red","blue","black","green","yellow","orange","brown","purple")
names(typeCols) <- typeNames
btype <- meta$broad_type
names(btype) <- rownames(meta)

outfile_pref <- strsplit(basename(args[1]),"\\.")[[1]][1]
print(outfile_pref)

# PCA on activations
pcafile <- paste(outfile_pref, "PCA.pdf", sep="_")
pdf(pcafile)
par(mfrow=c(1,2))
p <- prcomp(act)
plot(p$x,col=typeCols[btype[rownames(act)]],pch=20)
plot(p$x[,2:3],col=typeCols[btype[rownames(act)]],pch=20)
dev.off()

# Rtsne
#nondup <- act[which(!duplicated(act)),]
#tsnefile <- paste(outfile_pref, "tSNE.pdf", sep="_")
#pdf(tsnefile)
#r <- Rtsne(nondup)
#plot(r$Y, col=typeCols[btype[rownames(act)]],pch=20)
#dev.off()

nondup <- act[which(!duplicated(act)),]
tsnefile <- paste(outfile_pref, "tSNE.pdf", sep="_")
pdf(tsnefile)
r <- Rtsne(nondup, perplexity=10)
plot(r$Y, col=typeCols[btype[rownames(nondup)]],pch=20)
dev.off()


# Look at the nodes in order of decreasing standard deviation
pdf(paste(outfile_pref, "node_profiles.pdf", sep="_"))
par(mfrow=c(2,4))
for(node in order(apply(act, 2, sd),decreasing=TRUE)){
  node.act.per.type(act, node, btype)
}
dev.off()

# Or per cell type
pdf(paste(outfile_pref, "cell_profiles.pdf", sep="_"))
type.act.per.node(act, btype)
dev.off()

# Check predictivity
rf <- randomForest(x=act, y=as.factor(btype), importance=TRUE)
print(paste("RF estimated error rate", tail(rf$err.rate, n=1)[,1], sep=":"))
