# Function to calculate hidden layer from data
# 
# @keyword internal
#
# Function for calculating hidden layer:
VisToHid <- function(vis, weights, y, y.weights, cons = F) {
  # Function for calculating a hidden layer.
  #
  # Args:
  #   vis: Visual layer, or hidden layer from previous layer in DBN
  #   weights: Trained weights including the bias terms (use RBM)
  #   y: Label vector if only when training an RBM for classification
  #   y.weights: Label weights and bias matrix, only neccessary when training a RBM for classification
  #
  # Returns:
  #   Returns a hidden layer calculated with the trained RBM weights and bias terms.
  #
  # Initialize the visual, or i-1 layer
  V0 <- vis
  # if ( is.null(dim(V0))) {
  #   # If visual is a vector create matrix
  #   V0 <- matrix(V0, nrow= length(V0))
  # }
  if(missing(y) & missing(y.weights)) {
    # Calculate the hidden layer with the trained weights and bias
    if (cons){
      if (length(V0[[1]]) == 1){
        V0_1 <- c()
        V0_1[[1]] <- V0[, c(1:(ceiling(ncol(V0)/2)))]
        V0_1[[2]] <- V0[, c(1, (ceiling(ncol(V0)/2)+1):ncol(V0))]
        V0 <- V0_1
      }
      H <- 1/(1 + exp(- ( cbind(V0[[1]] %*% weights[[1]], (V0[[2]] %*% weights[[2]])[,-1 , drop = F])))) 
    } else {
      H <- 1/(1 + exp(-( V0 %*% weights))) 
    }
  } else {
    if (cons){
      if (length(V0[[1]]) == 1){
        V0_1 <- c()
        V0_1[[1]] <- V0[, c(1:(ceiling(ncol(V0)/2)))]
        V0_1[[2]] <- V0[, c(1, (ceiling(ncol(V0)/2)+1):ncol(V0))]
        V0 <- V0_1
      }
      Y0 <- y
      H <- 1/(1 + exp(- ( cbind(as.matrix(V0[[1]] %*% weights[[1]]), as.matrix(V0[[2]] %*% weights[[2]][, -1])) + Y0 %*% y.weights))) 
    } else {
      Y0 <- y
      H <- 1/(1 + exp(- ( V0 %*% weights + Y0 %*% y.weights))) 
    }
  }
  return(H)
}

# Function for reconstructing data from a hidden layer
# 
# @keyword internal
# Function for reconstructing visible layer:
HidToVis <- function(inv, weights, y.weights, cons = F, n.h) {
  # Function for reconstructing a visible layer.
  #
  # Args:
  #   inv: Invisible layer
  #   vis.bias: Trained visible layer bias (use RBM)
  #   weights: Trained weights (use RBM)
  #   y.weights: Label weights, only nessecessary when training a classification RBM.
  #
  # Returns:
  #   Returns a vector with reconstructed visible layer or reconstructed labels.
  #
  if(missing(y.weights)) {
    # Reconstruct only the visible layer when y.weights is missing
    if (cons){
      V <- 1/(1 + exp(-( cbind(inv[, c(1:((n.h/2)+1))] %*% t(weights[[1]]), 
                      as.matrix(inv[, c(1, ((n.h/2)+2):(n.h+1))] %*% t(weights[[2]])[, -1])) )))
    } else {
      V <- 1/(1 + exp(-(   inv %*% t(weights)) ))
    }
    return(V)
  } else {
    # Reconstruct visible and labels if y.weights
    Y <- 1/(1 + exp(-( inv %*% t(y.weights)))) 
    return(Y)
  }
}

# Logistic function
# 
# @keyword internal
# Logistic function
logistic <- function(x) {
  1/(1+exp(-x))
}

# Function to calculate the energy of a RBM 
# 
# @keyword internal
# Function for calculating the energy of the machine:
Energy <- Energy <- function(vis, inv, weights, y, y.weights, constr = F) {
  # Function for calculating the energy of a trained RBM
  #
  # Args:
  #   vis: visible layer
  #   weights: the weights matrix including the bias terms
  #   inv: invisible layer
  #   y: label vector (binary)
  #   y.weights: trained label weights (use RBM), including bias terms
  #
  # Returns:
  #   The energy of the RBM machine for label y
  #
  # Calculate the energy if supervised
  if(!missing(y) & !missing(y.weights)){
    E <- -(vis %*% weights %*% t(inv)) - (y %*% y.weights %*% t(inv))
  } else {
    # Calculate the energy if unsupervised
    E <- -(vis %*% weights %*% t(inv))
  }
  # Return the energy:
  return(E)
  
}

# Function for doing contrastive divergence CD
#
# @keyword internal
CD <- function(vis, weights, y, y.weights, constr = F, n.hid) {
  # Function for doing k=1 contrastive divergence
  # 
  # Args:
  #   vis: visible layer values vector of shape n_features * 1
  #   weights: weights vector of shape n_features * n_hidden
  #   vis.bias: bias of the visible layer
  #   inv.bias: bias of the invisible layer
  #   y: labels, only used when provided
  #   y.weigths: label weights of shape n_labels * n_hidden, only used when provided
  #   y.bias: bias term for the labels of shape n_features * 1, only used when provided
  #
  # Returns:
  #   A list with all gradients for the bias and weights; adds label bias and weights if y is provided
  #
  # Start positive phase
  if (missing(y) & missing(y.weights)) {
    # Calculate hidden layer
    H0 <- VisToHid(vis, weights, cons = constr)
    # H0[,1] <- 1 #Turned off because this was done different in other implementation
  } else {
    # Add a layer with labels if y is provided
    H0 <- VisToHid(vis, weights, y, y.weights, cons = constr)
    #H0[,1] <- 1 #Turned off because this was done different in other implementation
  }
  if (!is.matrix(H0)){
    H0 <- as.matrix(H0)
  }
  # Binarize the hidden layer:
  unif  <- runif(nrow(H0) * (ncol(H0)))
  H0.states <- H0 > matrix(unif, nrow=nrow(H0), ncol= ncol(H0))
  
  # Calculate positive phase, we always use the probabilities for this
  if (constr){
    pos.phase <- c()
    pos.phase[[1]] <- t(vis[[1]]) %*% H0[, c(1:(ceiling(n.hid/2)+1))]
    pos.phase[[2]] <- t(vis[[2]]) %*% H0[, c(1, (ceiling(n.hid/2)+2):(n.hid+1))]
  } else {
    pos.phase <- t(vis) %*% H0
  }
  if (!missing(y)) {
    pos.phase.y <- t(y) %*% H0
  }
  # Start negative  phase
  # Reconstruct visible layer
  V1 <- HidToVis(H0.states, weights, cons = constr, n.h = n.hid)
  # Set the bias unit to 1
  V1[,1] <- 1
  
  if (missing(y) & missing(y.weights) ) {
    # Reconstruct hidden layer unsupervised, no need to fix the bias anymore
    H1 <- VisToHid(V1, weights, cons = constr)
  } else {
    # Reconstruct labels if y is provided
    Y1 <- HidToVis(H0, weights,  y.weights, cons = constr, n.h = n.hid)
    # Set the bias unit to 1
    Y1[,1] <- 1
    # Reconstruct hidden layer supervised, no need to fix the bias anymore
    H1 <- VisToHid(V1, weights, Y1, y.weights, cons = constr)
  }
  # Calculate negative associations, we alway use the probabilities for this:
  if (constr){
    if(typeof(V1) != "matrix"){
      V1 <- as.matrix(V1)
      H1 <- as.matrix(H1)
    }
    neg.phase <- c()
    neg.phase[[1]] <- t(V1[, c(1:(ceiling((ncol(V1)-1)/2)+1)), drop = FALSE]) %*% H1[, c(1:(ceiling(n.hid/2)+1)), drop = FALSE]
    neg.phase[[2]] <- t(V1[, c(1, (ceiling((ncol(V1)-1)/2)+2):ncol(V1)), drop = FALSE]) %*% H1[, c(1, (ceiling(n.hid/2)+2):(n.hid+1)), drop = FALSE]
  } else {
    neg.phase <- t(V1) %*% H1
  }

  if (!missing(y) & !missing(y.weights)) {
    # Calculate negative phase y
    neg.phase.y <- t(Y1) %*% H1
  }
  ## Calculate the gradients
  # Calculate gradients for the weights:
  if (constr){
    grad.weights <- c()
    grad.weights[[1]] <- pos.phase[[1]] - neg.phase[[1]]
    grad.weights[[2]] <- pos.phase[[2]] - neg.phase[[2]]
  } else {
    grad.weights <- pos.phase - neg.phase
  }
  
  if (!missing(y) & !missing(y.weights)) {
    # Calculate gradients for y.weigths
    grad.y.weights <- pos.phase.y - neg.phase.y
    
    # Return list with  gradients supervised
    return(list('grad.weights' = grad.weights,'grad.y.weights' = grad.y.weights))
  } else {
    # Return list with gradients unsupervised
    return(list('grad.weights' = grad.weights  ))
  }
}

# Function for binarizing label data
# 
# TODO: Replace loop by C++ loop (rcpp?)
# @keyword internal
# Function for binarizing labels:
LabelBinarizer <- function(labels) {
  # This function takes as input the labels of the trainset.
  # Args:
  #   Labels: has to be numerical data vector from 1 to 9.
  #
  # Returns:
  #   Matrix with binarized vectors for the labels that can be used in the RBM function
  #
  # Initialize matrix to save label vectors:
  y <- matrix(0, length(labels), length(unique(labels)))
  for (i in 1:length(labels)) {
    # Put a one on position of the number in vector:
    y[i, labels[i] + 1] <- 1
  }
  return(y)
}


# Adjust weight matrix 

adj_wm <- function(input, n.hid, bias = F){
  if (bias){
    m_zero_1 <- matrix(c(input[[2]][1, ][-1], rep(0, nrow(input[[2]])*n.hid/2-length(input[[2]][1, ][-1]))),
                       nrow = nrow(input[[2]]), ncol = n.hid/2, byrow = T)
    m_zero_2 <- matrix(c(input[[2]][, 1][-1], rep(0, (nrow(input[[2]])-1)*((n.hid/2)+1)-length(input[[2]][, 1][-1]))),
                       nrow = (nrow(input[[2]])-1), ncol = ((n.hid/2)+1), byrow = F)
    
    m_out_1 <- cbind(input[[1]], m_zero_1)
    m_out_2 <- cbind(m_zero_2, input[[2]][-1, -1])
    out <- rbind(m_out_1, m_out_2)
  } else {
    wts_out1 <- cbind(input[[1]], matrix(0, nrow = dim(input[[1]])[1], 
                                                  ncol = dim(input[[1]])[2]))
    wts_out2 <- cbind(matrix(0, nrow = dim(input[[2]])[1], 
                             ncol = dim(input[[2]])[2]), 
                      input[[2]])
    out <- rbind(wts_out1, wts_out2)
  }
  return(out)
}

get_energy_means <- function(outputs){
  lists <- list("1" = numeric(),
                "2" = numeric(),
                "3" = numeric(),
                "4" = numeric(),
                "1_false" = numeric(),
                "2_false" = numeric(),
                "3_false" = numeric(),
                "4_false" = numeric(),
                "all" = numeric(),
                "false" = numeric())
  
  for (j in c(1:length(lists))){
    lists[names(lists[j])] <- outputs[[1]]$energy[names(lists[j])]
  }
  for (i in c(2:length(outputs))){
    for (j in c(1:length(lists))){
      lists[names(lists[j])] <- list(rbind(lists[names(lists[j])][[1]], outputs[[i]]$energy[names(lists[j])][[1]]))
    }
  }
  
  averages <- list("energy" = list("1" = colMeans(lists$`1`, na.rm = T), 
                                   "2" = colMeans(lists$`2`, na.rm = T),
                                   "3" = colMeans(lists$`3`, na.rm = T),
                                   "4" = colMeans(lists$`4`, na.rm = T), 
                                   "1_false" = colMeans(lists$`1_false`, na.rm = T), 
                                   "2_false" = colMeans(lists$`2_false`, na.rm = T),
                                   "3_false" = colMeans(lists$`3_false`, na.rm = T),
                                   "4_false" = colMeans(lists$`4_false`, na.rm = T), 
                                   "all" = colMeans(lists$`all`), 
                                   "false" = colMeans(lists$`false`)))
  
  return(averages)
}

# counts after how many iterations a model did a classification x times correct in a row
classification_counter <- function(model, min_true = 10, thresh_abs = 0){
  counter <- 0
  diff <- model$energy$all - model$energy$false + thresh_abs
  for (i in c(1: length(diff))){
    if (diff[i] < 0){
      counter <- counter +1
    } else {
      counter <- 0
    }
    if (counter >= min_true){
      break
    }
  }
  return(i)
}

counter <- function(model, min_true = 10, thresh_abs = 0){
  if (names(model)[length(names(model))] == "energy"){
    out <- classification_counter(model = model, min_true = min_true, thresh_abs = thresh_abs)
  } else {
    out <- list("pre.shift" = c(), 
                "rev.shift" = c(), 
                "nonrev.shift" = c())
    for (i in model$pre.shift){
      out$pre.shift <- c(out$pre.shift, classification_counter(i))
    }
    for (i in model$rev.shift){
      out$rev.shift <- c(out$rev.shift, classification_counter(i))
    }
    for (i in model$nonrev.shift){
      out$nonrev.shift <- c(out$nonrev.shift, classification_counter(i))
    }
  }
  return(out)
}





