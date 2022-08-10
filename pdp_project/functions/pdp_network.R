# little neural network

feedforward_old <- function(nn, batch, constrained = F) {
  if (constrained){
    nn$layer1.1 <- activation_function(nn$input[[1]][batch, , drop = F] %*% nn$weights1[[1]])
    nn$layer1.2 <- activation_function(nn$input[[2]][batch, , drop = F] %*% nn$weights1[[2]])
    nn$output <- activation_function(cbind(nn$layer1.1, nn$layer1.2) %*% nn$weights2)
  } else {
    nn$layer1 <- activation_function(nn$input[batch, , drop = F] %*% nn$weights1)
    nn$output <- activation_function(nn$layer1 %*% nn$weights2)
  }
  return(nn)
}

backprop_old <- function(nn, batch, constrained = F, learning_rate) {
  
  if (constrained){
    # application of the chain rule to find derivative of the loss function with
    # respect to weights2 and weights1
    d_weights2 <- ( # `2 * (nn$y - nn$output)` is the derivative of the sum on squared loss function
      t(cbind(nn$layer1.1, nn$layer1.2)) %*% (2 * (nn$y[batch, , drop = F] - nn$output) * derivative_activation(nn$output))
    )
    
    d_weights1 <- ( 2 * (nn$y[batch, , drop = F] - nn$output) * derivative_activation(nn$output)) %*%
      t(nn$weights2)
    
    d_weights1.1 <- d_weights1[, 1:(ncol(d_weights1)/2), drop = F] * derivative_activation(nn$layer1.1)
    d_weights1.1 <- t(nn$input[[1]][batch, , drop = F]) %*% d_weights1.1

    d_weights1.2 <- d_weights1[, ((ncol(d_weights1)/2)+1) : ncol(d_weights1), drop = F] *
      derivative_activation(nn$layer1.2)
    d_weights1.2 <- t(nn$input[[2]][batch, , drop = F]) %*% d_weights1.2
    
    # update the weights using the derivative (slope) of the loss function
    nn$weights1[[1]] <- nn$weights1[[1]] + (learning_rate * d_weights1.1)
    nn$weights1[[2]] <- nn$weights1[[2]] + (learning_rate * d_weights1.2)
    nn$weights2 <- nn$weights2 + (learning_rate * d_weights2)
  } else {
    # application of the chain rule to find derivative of the loss function with
    # respect to weights2 and weights1
    d_weights2 <- (
      t(nn$layer1) %*%
        # `2 * (nn$y - nn$output)` is the derivative of the sigmoid loss function
        (2 * (nn$y[batch, , drop = F] - nn$output) *
           derivative_activation(nn$output))
    )
    
    d_weights1 <- ( 2 * (nn$y[batch, , drop = F] - nn$output) * derivative_activation(nn$output)) %*%
      t(nn$weights2)
    d_weights1 <- d_weights1 * derivative_activation(nn$layer1)
    d_weights1 <- t(nn$input[batch, , drop = F]) %*% d_weights1
    
    # update the weights using the derivative (slope) of the loss function
    nn$weights1 <- nn$weights1 + (learning_rate * d_weights1)
    nn$weights2 <- nn$weights2 + (learning_rate * d_weights2)
  }
  return(nn)
}

activation_function <- function(x) { #sigmoid activation
  return(1.0 / (1.0 + exp(-x)))
}
derivative_activation <- function(x) { # derivative of sigmoid 
  return(x * (1.0 - x))
}

init_weights_old <- function(nin = 2, nout = 2, dist = runif, mean_min = -1, sd_max = 1, 
                         constrained = F, bias = F, prev_wts){
  if(constrained & missing(prev_wts)){
    wts_out <- c()
    wts_out[[1]] <- matrix(dist(ceiling(nin/2) * ceiling(nout/2), mean_min, sd_max), nrow = ceiling(nin/2), ncol = ceiling(nout/2))
    wts_out[[2]] <- matrix(dist(floor(nin/2) * floor(nout/2), mean_min, sd_max), nrow = floor(nin/2), ncol = floor(nout/2))
    if (bias){
      wts_out[[1]] <- rbind(0, cbind(0, wts_out[[1]]))
      wts_out[[2]] <- rbind(0, cbind(0, wts_out[[2]]))
    }
  } else if (!constrained & missing(prev_wts)) {
    wts_out <- matrix(dist(nin * nout, mean_min, sd_max), nrow = nin, ncol = nout)
    if (bias){
      wts_out <- rbind(0, cbind(0, wts_out))
    } 
  }
  if (!missing(prev_wts) & !constrained){
      wts_out <- prev_wts
  } else if (!missing(prev_wts) & constrained){
    if (bias){
      wts_out <- c()
      wts_out[[1]] <- prev_wts[1:(ceiling(nrow(prev_wts)/2)), 1:(ceiling(ncol(prev_wts)/2)), drop = F]
      wts_out[[2]] <- prev_wts[c(1, ((ceiling(nrow(prev_wts)/2))+1):nrow(prev_wts)), 
                               c(1, ((ceiling(ncol(prev_wts)/2))+1):ncol(prev_wts)), drop = F]
    } else {
      wts_out <- c()
      wts_out[[1]] <- prev_wts[1:(ceiling(nrow(prev_wts)/2)), 1:(ceiling(ncol(prev_wts)/2)), drop = F]
      wts_out[[2]] <- prev_wts[((ceiling(nrow(prev_wts)/2))+1):nrow(prev_wts), 
                               ((ceiling(ncol(prev_wts)/2))+1):ncol(prev_wts), drop = F]
    }
  }
  return(wts_out)
}




#############################################


pdp_old <- function(x, y, n_hidden = 4, batch_size = 4, n_iter = 100, 
                activation_function, derivative_activation, loss_function, 
                plot_loss = F, constrained = F, threshold = 0.55, learning_rate = 0.1, 
                prev_wts_1, prev_wts_2, min_correct, plot_wtchange = F, plot_every = 10){
  nrow_x <- nrow(x)
  ncol_x <- ncol(x)
  
  if(missing(min_correct)){
    min_correct <- n_iter
  }
  
  if (constrained){
    x_c <- c()
    x_c[[1]] <- x[, 1:(ncol(x)/2)]
    x_c[[2]] <- x[, ((ncol(x)/2)+1):ncol(x)]
    x <- x_c
  } 
  
  labels <- unique(y)
  # Get the indexes of each unique label in y
  idx <- vector('list', length = length(labels))
  # Save indexes
  for (i in 1:length(labels)) {
    idx[[i]]<- which(y == labels[i])
  }
  # Make binarized vectors of the labels
  y <- LabelBinarizer(y)
  
  # Initialize the weights, n.features * n.hidden with values from gaussian distribution
  weights_in_hidden <- init_weights(nin = ncol_x, nout = n_hidden, constrained = constrained, 
                                    dist = runif, mean_min = -0.5, sd_max = 0.5, 
                                    prev_wts = prev_wts_1)
  
  weights_hidden_out <- init_weights(nin = n_hidden, nout = ncol(y), constrained = F, 
                                     dist = runif, mean_min = -0.5, sd_max = 0.5, 
                                     prev_wts = prev_wts_2)
  
  loss <- c()
  
  if(plot_wtchange){
    if(constrained){
      plot_net(list(adj_wm(weights_in_hidden, n.hid = n_hidden), weights_hidden_out), no_bias_in_input = T, min_lw = 0)
    } else {
      plot_net(list(weights_in_hidden, weights_hidden_out), no_bias_in_input = T, min_lw = 0)
    }
  }
  
  # this list stores the state of our neural net as it is trained
  my_nn <- list(
    # predictor variables
    input = x,
    # weights for layer 1
    weights1 = weights_in_hidden,
    # weights for layer 2
    weights2 = weights_hidden_out,
    # actual observed
    y = y,
    # stores the predicted outcome
    output = matrix(
      rep(0, times = length(y)),
      ncol = 2
    ),
    iterations = c()
  )
  
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
    loss_function <- function(nn, batch) { #sum of squared 
      return(sum((nn$y[batch, , drop = F] - nn$output) ^ 2))
    }
  }
  
  # number of times to perform feedforward and backpropagation
  # data frame to store the results of the loss function.
  # this data frame is used to produce the plot in the 
  # next code chunk
  counter <- 1
  iter <- 1
  plot_counter <- 1
  while (iter < n_iter & counter < min_correct/batch_size) {
    batch <- sample(1:nrow_x, batch_size, replace = F)
    
    my_nn <- feedforward_old(my_nn, batch, constrained)
    my_nn <- backprop_old(my_nn, batch, constrained, learning_rate)
    
    # store the result of the loss function.  We will plot this later
    loss[iter] <- loss_function(my_nn, batch)
    if (all(my_nn$output * (y[batch, , drop = F] *2-1) - threshold + (y[batch, , drop = F]-1)^2 >= 0) & min_correct!=0){
      counter <- counter+1
    } else (
      counter <- 1
    )
    plot_counter <- plot_counter+1
    iter <- iter+1
    if(plot_counter == plot_every & plot_wtchange){
      if(constrained){
        plot_net(list(adj_wm(my_nn$weights1, n.hid = n_hidden), my_nn$weights2), no_bias_in_input = T, min_lw = 0)
      } else {
        plot_net(list(my_nn$weights1, my_nn$weights2), no_bias_in_input = T, min_lw = 0)
      }
      Sys.sleep(0.1)
      plot_counter <- 1
    }
  }
  my_nn$iterations <- iter
  if (constrained){
    my_nn$weights1 <- adj_wm(list(my_nn$weights1[[1]], my_nn$weights1[[2]]), 
                             n.hid = n_hidden, bias = F)
  }
  
  if (plot_loss){
    dev.off()
    plot(loss, frame = F, type = "l", 
         xlab = "Iteration", ylab = "Loss", ylim = c(0, max(loss)+0.5), xlim = c(0, n_iter))
    # print the predicted outcome next to the actual outcome
    
    print(data.frame(
      "Predicted" = round(my_nn$output, 3),
      "Actual" = y[batch, , drop = F]
    ))
  }
  return(my_nn)
}

