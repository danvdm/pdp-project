# little neural network

sigmoid <- function(x) { #sigmoid activation
  return(1.0 / (1.0 + exp(-x)))
}
d_sigmoid <- function(x) { # derivative of sigmoid 
  return(x * (1.0 - x))
}

feedforward <- function(input, wts_mmtms, batch, activation_function, constrained = F) {
  weights <- wts_mmtms[[1]]
  out <- list()
  if (constrained){
    for (i in c(1:length(weights))){
      if (typeof(weights[[i]]) == "list"){
        if (i == 1){
          o1 <- activation_function(input[[1]][batch, , drop = F] %*% weights[[i]][[1]])
          o2 <- activation_function(input[[2]][batch, , drop = F] %*% weights[[i]][[2]])
          out[[paste("layer", i, sep = "")]] <- cbind(o1, o2)
        } else if (i == length(weights)){
          o1 <- sigmoid(out[[i-1]][ , 1:nrow(weights[[i]][[1]]), drop = F] %*% weights[[i]][[1]])
          o2 <- sigmoid(out[[i-1]][ , (nrow(weights[[i]][[1]])+1):ncol(out[[i-1]]), drop = F] %*% weights[[i]][[2]])
          out[[paste("layer", i, sep = "")]] <- cbind(o1, o2)
        } else {
          o1 <- activation_function(out[[i-1]][ , 1:nrow(weights[[i]][[1]]), drop = F] %*% weights[[i]][[1]])
          o2 <- activation_function(out[[i-1]][ , (nrow(weights[[i]][[1]])+1):ncol(out[[i-1]]), drop = F] %*% weights[[i]][[2]])
          out[[paste("layer", i, sep = "")]] <- cbind(o1, o2)
        }
      } else {
        if (i == 1){
          out[[paste("layer", i, sep = "")]] <- activation_function(input[batch, , drop = F] %*% weights[[i]])
        } else if (i == length(weights)) {
          out[[paste("layer", i, sep = "")]] <- sigmoid(out[[i-1]] %*% weights[[i]])
        } else {
          out[[paste("layer", i, sep = "")]] <- activation_function(out[[i-1]] %*% weights[[i]])
        }
      }
    }
  } else {
    for (i in c(1:length(weights))){
      if (i == 1){
        out[[paste("layer", i, sep = "")]] <- activation_function(input[batch, , drop = F] %*% weights[[i]])
      } else if (i == length(weights)){
        out[[paste("layer", i, sep = "")]] <- sigmoid(out[[i-1]] %*% weights[[i]])
      } else {
        out[[paste("layer", i, sep = "")]] <- activation_function(out[[i-1]] %*% weights[[i]])
      }
    }
  }
  return(out)
}

backprop <- function(input, layers, wts_mmtms, y, batch, activation_function, 
                     learning_rates = 1, momentum = 0.5, momentums = mimentums, constrained = F, 
                     derivative_activation, lambda = 0.001, train_layers) {
  # application of the chain rule to find derivative of the loss function with
  # respect to weights2 and weights1
  weights <- wts_mmtms[[1]]
  momentums <- wts_mmtms[[2]]
  d_wts <- list()
  grad <- list()
  grad[[length(weights)]] <- 2 * (y[batch, , drop = F] - layers[[length(weights)]]) * d_sigmoid(layers[[length(weights)]])
  if (length(weights)==1){
    if (constrained){
      print("Constrained perceptron not yet implemented")
      break
    } else {
      d_wts[[length(weights)]] <- t(input[batch, , drop = F]) %*% grad[[length(weights)]]
      momentums[[1]] <- momentum * momentums[[1]] + ((d_wts[[1]] - (lambda * weights[[1]])))
      weights[[1]] <- weights[[1]] + (learning_rates[1] *  momentums[[1]])
    }
  } else {
    d_wts[[length(weights)]] <- t(layers[[length(weights)-1]]) %*% grad[[length(weights)]]
    if (constrained){
      for (i in c((length(weights)-1):1)){
        if (typeof(weights[[i]]) == "list"){
          if (i > 1){
            if (typeof(grad[[i+1]]) == "list"){
              interm1 <- grad[[i+1]][[1]] %*%  t(weights[[i+1]][[1]])
              interm2 <- grad[[i+1]][[2]] %*%  t(weights[[i+1]][[2]])
              grad[[i]][[1]]<- interm1 * derivative_activation(layers[[i]][, 1:ncol(interm1) , drop = F])
              d_wts[[i]][[1]] <- t(layers[[i-1]][, 1:(ncol(layers[[i-1]])/2) , drop = F]) %*% grad[[i]][[1]]
              grad[[i]][[2]] <- interm2 * derivative_activation(layers[[i]][, (ncol(interm1)+1):ncol(layers[[i]]), drop = F])
              d_wts[[i]][[2]] <- t(layers[[i-1]][, ((ncol(layers[[i-1]])/2)+1):ncol(layers[[i-1]]), drop = F]) %*% grad[[i]][[2]]
            } else {
              interm <- grad[[i+1]] %*%  t(weights[[i+1]])
              grad[[i]][[1]]<- interm[, 1:ceiling(ncol(interm)/2) , drop = F] * derivative_activation(layers[[i]][, 1:ceiling(ncol(interm)/2) , drop = F])
              d_wts[[i]][[1]] <- t(layers[[i-1]][, 1:(ncol(layers[[i-1]])/2) , drop = F]) %*% grad[[i]][[1]]
              grad[[i]][[2]] <- interm[, (ceiling(ncol(interm)/2)+1):ncol(interm), drop = F] * derivative_activation(layers[[i]][, (ceiling(ncol(interm)/2)+1):ncol(interm), drop = F])
              d_wts[[i]][[2]] <- t(layers[[i-1]][, ((ncol(layers[[i-1]])/2)+1):ncol(layers[[i-1]]), drop = F]) %*% grad[[i]][[2]]
            }
          } else {
            if (typeof(grad[[i+1]]) == "list"){
              interm1 <- grad[[i+1]][[1]] %*%  t(weights[[i+1]][[1]])
              interm2 <- grad[[i+1]][[2]] %*%  t(weights[[i+1]][[2]])
              grad[[i]][[1]]<- interm1 * derivative_activation(layers[[i]][, 1:ncol(interm1) , drop = F])
              d_wts[[i]][[1]] <- t(input[[1]][batch, , drop = F]) %*% grad[[i]][[1]]
              grad[[i]][[2]] <- interm2 * derivative_activation(layers[[i]][, (ncol(interm1)+1):ncol(layers[[i]]), drop = F])
              d_wts[[i]][[2]] <- t(input[[2]][batch, , drop = F]) %*% grad[[i]][[2]]
            } else {
              interm <- grad[[i+1]] %*%  t(weights[[i+1]])
              grad[[i]][[1]]<- interm[, 1:ceiling(ncol(interm)/2) , drop = F] * derivative_activation(layers[[i]][, 1:ceiling(ncol(interm)/2) , drop = F])
              d_wts[[i]][[1]] <- t(input[[1]][batch, , drop = F]) %*% grad[[i]][[1]]
              grad[[i]][[2]] <- interm[, (ceiling(ncol(interm)/2)+1):ncol(interm), drop = F] * derivative_activation(layers[[i]][, (ceiling(ncol(interm)/2)+1):ncol(interm), drop = F])
              d_wts[[i]][[2]] <- t(input[[2]][batch, , drop = F]) %*% grad[[i]][[2]]
            }
          }
        } else {
          if (i > 1){
            if (typeof(grad[[i+1]]) == "list"){
              interm1 <- grad[[i+1]][[1]] %*% t(weights[[i+1]][[1]]) 
              interm2 <- grad[[i+1]][[2]] %*% t(weights[[i+1]][[2]]) 
              interm <- cbind(interm1, interm2)
              grad[[i]] <- interm * derivative_activation(layers[[i]])
              d_wts[[i]] <- t(layers[[i-1]]) %*% grad[[i]]
            } else {
              grad[[i]] <- grad[[i+1]] %*% t(weights[[i+1]]) 
              grad[[i]] <- grad[[i]] * derivative_activation(layers[[i]])
              d_wts[[i]] <- t(layers[[i-1]]) %*% grad[[i]]
            }
          } else {
            if (typeof(grad[[i+1]]) == "list"){
              interm1 <- grad[[i+1]][[1]] %*% t(weights[[i+1]][[1]]) 
              interm2 <- grad[[i+1]][[2]] %*% t(weights[[i+1]][[2]]) 
              interm <- cbind(interm1, interm2)
              grad[[i]] <- interm * derivative_activation(layers[[i]])
              d_wts[[i]] <- t(input[batch, , drop = F]) %*% grad[[i]]
            } else {
              grad[[i]] <- grad[[i+1]] %*%  t(weights[[i+1]])
              grad[[i]] <- grad[[i]] * derivative_activation(layers[[i]])
              d_wts[[i]] <- t(input[batch, , drop = F]) %*% grad[[i]]
            }
          }
        }
      }
      
      # update the weights using the derivative (slope) of the loss function
      for (i in c(1:length(d_wts))){
        if (typeof(d_wts[[i]]) == "list"){
          momentums[[i]][[1]] <- momentum * momentums[[i]][[1]] + ((d_wts[[i]][[1]] - (lambda * weights[[i]][[1]]))/length(batch))
          momentums[[i]][[2]] <- momentum * momentums[[i]][[2]] + ((d_wts[[i]][[2]] - (lambda * weights[[i]][[2]]))/length(batch))
          
          weights[[i]][[1]] <- weights[[i]][[1]] + (learning_rates[i] *  momentums[[i]][[1]])
          weights[[i]][[2]] <- weights[[i]][[2]] + (learning_rates[i] *  momentums[[i]][[2]])
          
        } else {
          momentums[[i]] <- momentum * momentums[[i]] + ((d_wts[[i]] - (lambda * weights[[i]])))
          weights[[i]] <- weights[[i]] + (learning_rates[i] *  momentums[[i]])
        }
      }
    } else {
      for (i in c((length(weights)-1):1)){
        if (i > 1){
          grad[[i]] <- grad[[i+1]] %*% t(weights[[i+1]]) 
          grad[[i]] <- grad[[i]] * derivative_activation(layers[[i]])
          d_wts[[i]] <- t(layers[[i-1]]) %*% grad[[i]]
        } else {
          grad[[i]] <- grad[[i+1]] %*%  t(weights[[i+1]])
          grad[[i]] <- grad[[i]] * derivative_activation(layers[[i]])
          d_wts[[i]] <- t(input[batch, , drop = F]) %*% grad[[i]]
        }
      }
      
      # update the weights using the derivative (slope) of the loss function
      for (i in c(1:length(d_wts))){
        momentums[[i]] <- momentum * momentums[[i]] + ((d_wts[[i]] - (lambda * weights[[i]]))/length(batch))
        weights[[i]] <- weights[[i]] + (learning_rates[i] *  momentums[[i]])
      }
    }
  }
  return(list(weights, momentums))
}


init_weights <- function(nin = 2, nout = 2, distr = runif, mean_min = -1, sd_max = 1, 
                         constrained = F, bias = F, prev_wts = NA){
  if(constrained & is.na(prev_wts)){
    wts_out <- c()
    wts_out[[1]] <- matrix(distr(ceiling(nin/2) * ceiling(nout/2), mean_min, sd_max), nrow = ceiling(nin/2), ncol = ceiling(nout/2))
    wts_out[[2]] <- matrix(distr(floor(nin/2) * floor(nout/2), mean_min, sd_max), nrow = floor(nin/2), ncol = floor(nout/2))
    if (bias){
      wts_out[[1]] <- rbind(0, cbind(0, wts_out[[1]]))
      wts_out[[2]] <- rbind(0, cbind(0, wts_out[[2]]))
    }
  } else if (!constrained & is.na(prev_wts)) {
    wts_out <- matrix(distr(nin * nout, mean_min, sd_max), nrow = nin, ncol = nout)
    if (bias){
      wts_out <- rbind(0, cbind(0, wts_out))
    } 
  }
  if (!is.na(prev_wts) & !constrained){
    wts_out <- prev_wts[[1]]
  } else if (!is.na(prev_wts) & constrained){
    if (bias){
      wts_out <- c()
      wts_out[[1]] <- prev_wts[[1]][1:(ceiling(nrow(prev_wts[[1]])/2)), 1:(ceiling(ncol(prev_wts[[1]])/2)), drop = F]
      wts_out[[2]] <- prev_wts[[1]][c(1, ((ceiling(nrow(prev_wts[[1]])/2))+1):nrow(prev_wts[[1]])), 
                               c(1, ((ceiling(ncol(prev_wts[[1]])/2))+1):ncol(prev_wts[[1]])), drop = F]
    } else {
      wts_out <- c()
      wts_out[[1]] <- prev_wts[[1]][1:(ceiling(nrow(prev_wts[[1]])/2)), 1:(ceiling(ncol(prev_wts[[1]])/2)), drop = F]
      wts_out[[2]] <- prev_wts[[1]][((ceiling(nrow(prev_wts[[1]])/2))+1):nrow(prev_wts[[1]]), 
                               ((ceiling(ncol(prev_wts[[1]])/2))+1):ncol(prev_wts[[1]]), drop = F]
    }
  }
  return(wts_out)
}

adj_wms <- function(wt_list, hidden){
  wt_list_out <- list()
  for (i in c(1:length(wt_list))){
    if (typeof(wt_list[[i]]) == "list"){
      wt_list_out[[i]] <- adj_wm(wt_list[[i]], neurons[i])
    } else { 
      wt_list_out[[i]] <- wt_list[[i]]
    }
  }
  return(wt_list_out)
}


#############################################


pdp <- function(x, y, n_hidden = c(4, 4), batch_size = 4, n_iter = 100, momentum = 0.5, 
                lambda = 0.001, activation_function, derivative_activation, loss_function, 
                plot_loss = F, constrained = F, threshold = 0.55, learning_rates = c(0.1, 0.1), 
                prev_wts, min_correct, plot_wtchange = F, plot_every = 10, use_accuracy = T,
                distribution = runif, mean_min = -0.5, sd_max = 0.5, accuracy = 0.8, binarize = T){
  nrow_x <- nrow(x)
  ncol_x <- ncol(x)
  
  if (missing(prev_wts)){
    prev_wts <- rep(NA, length(n_hidden)+1)
  } 
  
  if(missing(min_correct)){
    min_correct <- n_iter
  }
  
  if (binarize){
    labels <- unique(y)
    # Get the indexes of each unique label in y
    idx <- vector('list', length = length(labels))
    # Save indexes
    for (i in 1:length(labels)) {
      idx[[i]]<- which(y == labels[i])
    }
    # Make binarized vectors of the labels
    y <- LabelBinarizer(y)
  } 
  
  
  # make a previous weights list that works if the previous weights are less than the newe ones
  if (!all(is.na(prev_wts)) & length(prev_wts) != (length(n_hidden)+1)){
    list_int <- list()
    for (wt in 1:(length(n_hidden)+1)){
      list_int[[wt]] <- NA  
    }
    for (wt in 1:length(prev_wts)){
      list_int[[wt]] <- prev_wts[[wt]]
    }
    prev_wts <- list_int
  }
  
  
  if (constrained[1]){
    x_c <- c()
    x_c[[1]] <- x[, 1:(ncol(x)/2)]
    x_c[[2]] <- x[, ((ncol(x)/2)+1):ncol(x)]
    x <- x_c
  } 
  
  if(length(constrained) != length(n_hidden)){
    constrained <- rep(F, length(n_hidden)+1)
  } else {
    constrained <- c(constrained, F)
  }
  
  if(length(learning_rates) != (length(n_hidden)+1)){
    learning_rates <- rep(learning_rates[1], length(n_hidden)+1)
  }

  if (is.na(n_hidden)){
    neurons <- c(ncol_x, ncol(y))
  } else {
    neurons <- c(ncol_x, n_hidden, ncol(y))
  }
  
  
  
  weights <- list()
  # Initialize the weights, n.features * n.hidden with values from gaussian distribution
  for (i in c(1:(length(neurons)-1))){
    weights[[i]] <- init_weights(nin = neurons[i], nout = neurons[i+1], constrained = constrained[i], 
                                 distr = distribution, mean_min = mean_min, sd_max = sd_max, 
                                 prev_wts = prev_wts[i], bias = F)
  }
  
  momentums <- list()
  for (i in c(1:(length(neurons)-1))){
    momentums[[i]] <- init_weights(nin = neurons[i], nout = neurons[i+1], constrained = constrained[i], 
                                 distr = distribution, mean_min = 0, sd_max = 0, bias = F)
  }

  weigths_momentums <- list(weights, momentums)
  
  
  if(plot_wtchange){
    plot_net(adj_wms(weights, neurons), no_bias_in_input = T, min_lw = 0)
  }
  
  out <- list(iterations = c(), 
                 loss = c(), 
              accuracy = c())
  
  # this list stores the state of our neural net as it is trained
  #the activation function
  if (missing(activation_function) | missing(derivative_activation)){
    activation_function <- function(x) { #sigmoid activation
      return(1.0 / (1.0 + exp(-x)))
    }
    #' the derivative of the activation function
    derivative_activation <- function(x) { # derivative of sigmoid 
      return(x * (1.0 - x))
    }
  }
  
  if (missing(loss_function)){ 
    loss_function <- function(y, output, batch) { #sum of squared 
      return(sum((y[batch, , drop = F] - output) ^ 2))
    }
  }
  
  acc_low_bs <- rep(NA, min_correct)
  
  # number of times to perform feedforward and backpropagation
  # data frame to store the results of the loss function.
  # this data frame is used to produce the plot in the 
  # next code chunk
  # if (min_correct<batch_size){
  #   min_correct <- batch_size
  # } 
  iter <- 1
  counter <- 1
  plot_counter <- 1
  while (iter < n_iter & counter <= min_correct) {
    batch <- sample(1:nrow_x, batch_size, replace = F)
    
    layers <- feedforward(input = x, wts_mmtms = weigths_momentums, batch = batch, 
                          activation_function = activation_function, constrained = any(constrained))
    
    weigths_momentums <- backprop(input = x, layers = layers, wts_mmtms = weigths_momentums, 
                                  y = y, batch = batch, momentum = momentum, momentums = momentums, 
                                  lambda = lambda, activation_function = activation_function, 
                                  learning_rates = learning_rates, constrained = any(constrained), 
                                  derivative_activation = derivative_activation)
    
    # store the result of the loss function.  We will plot this later
    out$loss[iter] <- loss_function(y = y[batch, , drop = F], output = layers[[length(layers)]])
    out$predicted[[iter]] <- layers[[length(layers)]]
    out$true[[iter]] <- y[batch, , drop = F]
    out$performance[[iter]] <- table(((layers[[length(layers)]] * (y[batch, , drop = F] *2-1) - threshold + (y[batch, , drop = F]-1)^2 >= 0) & min_correct!=0)[, 1] == matrix(rep(1, batch_size)))
    out$order[[iter]] <- batch
    if (use_accuracy){
      acc <- ifelse(is.na(out$performance[[iter]]["TRUE"]), 0, out$performance[[iter]]["TRUE"]/batch_size)
      if(batch_size < 4){
        acc_low_bs <- acc_low_bs[2:min_correct]
        acc_low_bs <- c(acc_low_bs, ifelse(is.na(out$performance[[iter]]["TRUE"]), 0, out$performance[[iter]]["TRUE"])[[1]])
        acc <- ifelse(is.na(table(acc_low_bs)["1"][[1]]), 0, table(acc_low_bs)["1"][[1]]/min_correct)
        }
      out$accuracy[iter] <- acc
      if (!is.nan(acc) & acc >= accuracy){
        counter <- counter+1
      } else {
        counter <- 1
      }
    } else {
        if (all(labels(out$performance[[iter]])[[1]] == T)){
        counter <- counter+1
      } else {
        counter <- 1
      }
    }
    plot_counter <- plot_counter+1
    iter <- iter+1
    if((plot_counter == plot_every) & plot_wtchange){
      plot_net(adj_wms(weigths_momentums[[1]], n_hidden), no_bias_in_input = T, min_lw = 0)
      Sys.sleep(0.2)
      plot_counter <- 1
    }
  }
  
  out$iterations <- iter
  out$weights <- adj_wms(weigths_momentums[[1]], n_hidden)
  out$layers <- layers

  if (plot_loss){
    plot(min(1), max(1))
    dev.off()
    plot(out$loss, frame = F, cex = 0.5, 
         xlab = "Iteration", ylab = "Loss", ylim = c(0, max(out$loss)+0.5), xlim = c(0, n_iter))
  }
  return(out)
}

relu <- function(x){
  if (is.matrix(x)){
    return(matrix(pmax(0, x), nrow = nrow(x), byrow = F))
  } else {
    return(matrix(pmax(0, x), ncol = length(x), byrow = T))
  }
}

d_relu <- function(x){
  if (is.matrix(x)){
    return(matrix(ifelse(x<0, 0, 1), nrow = nrow(x), byrow = F))
  } else {
    return(matrix(ifelse(x<0, 0, 1), ncol = length(x), byrow = T))
  }
}

l_relu <- function(x){
  if (is.matrix(x)){
    return(matrix(pmax(0.05*x, x), nrow = nrow(x), byrow = F))
  } else {
    return(matrix(pmax(0.05*x, x), ncol = length(x), byrow = T))
  }
}

l_d_relu <- function(x){
  if (is.matrix(x)){
    return(matrix(ifelse(x<0, 0.05*x, 1), nrow = nrow(x), byrow = F))
  } else {
    return(matrix(ifelse(x<0, 0.05*x, 1), ncol = length(x), byrow = T))
  }
}






