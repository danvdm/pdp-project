

simulate_pdp <- function(X, Y1, Y2, Y3, nsim, constrained = F, n_hidden = 4, n_iter = 1000, 
                         batch_size = 1, min_correct = 10, threshold = 0.55, plot_loss = F, 
                         learning_rates1 = 0.1, learning_rates2 = 0.1, learning_rates3 = 0.1, 
                         momentum = 0.5, lambda = 0, accuracy = 0.8, add, use_acc = T) {
  if (missing(add)){
    out <- list(pre = list(), 
                rev = list(), 
                nrev = list())
    s <- 0
  } else {
    out <- add
    s <- length(out$pre$iterations)
  }
  
  
  for (sim in c(1:nsim)){
    print(paste("Running simulation nr:", sim))
    pre <- pdp(X, Y1, n_hidden = n_hidden, plot_loss = plot_loss, 
               constrained = constrained, n_iter = n_iter, batch_size = batch_size, learning_rates = learning_rates1, 
               min_correct = min_correct, threshold = threshold, momentum = momentum, 
               lambda = lambda, accuracy = accuracy, use_accuracy = use_acc)
    
    rev <- pdp(X, Y2, n_hidden = n_hidden, plot_loss = plot_loss, 
               constrained = constrained, n_iter = n_iter, batch_size = batch_size, learning_rates = learning_rates2, 
               prev_wts = pre$weights, min_correct = min_correct, threshold = threshold,
               momentum = momentum, lambda = lambda, accuracy = accuracy, use_accuracy = use_acc)
    
    nrev <- pdp(X, Y3, n_hidden = n_hidden, plot_loss = plot_loss, 
                constrained = constrained, n_iter = n_iter, batch_size = batch_size, learning_rates = learning_rates3, 
                prev_wts = pre$weights, min_correct = min_correct, threshold = threshold, 
                momentum = momentum, lambda = lambda, accuracy = accuracy, use_accuracy = use_acc)
    
    out$pre$iterations[sim+s] <- pre$iterations
    out$pre$loss[[sim+s]] <- pre$loss
    out$pre$performance[[sim+s]] <- pre$performance
    out$rev$iterations[sim+s] <- rev$iterations
    out$rev$loss[[sim+s]] <- rev$loss
    out$rev$performance[[sim+s]] <- rev$performance
    out$nrev$iterations[sim+s] <- nrev$iterations
    out$nrev$loss[[sim+s]] <- nrev$loss
    out$nrev$performance[[sim+s]] <- nrev$performance
  }
  return(out)
}

simulate_pdp_2 <- function(X, Y1, Y2, Y3, Y4, nsim, constrained = F, n_hidden = 4, n_iter = 1000, 
                         batch_size = 1, min_correct = 10, threshold = 0.55, plot_loss = F, 
                         learning_rates1 = 0.1, learning_rates2 = 0.1, learning_rates3 = 0.1, 
                         momentum = 0.5, lambda = 0, accuracy = 0.8, add, transfer_all = T, use_acc = T, 
                         binarize = c(T, T, T, T)) {
  if (missing(add)){
    out <- list(pre = list(), 
                rev = list(), 
                nrev = list())
    s <- 0
  } else {
    out <- add
    s <- length(out$pre$iterations)
  }
  
  if (length(n_hidden) != 4){
    n_hidden <- rep(n_hidden[1], 4)
  }
  
  
  for (sim in c(1:nsim)){
    print(paste("Running simulation nr:", sim))
    pre <- pdp(X, Y1, n_hidden = n_hidden[1], plot_loss = plot_loss, 
               constrained = constrained, n_iter = n_iter, batch_size = batch_size, learning_rates = learning_rates1, 
               min_correct = min_correct, threshold = threshold, momentum = momentum, 
               lambda = lambda, accuracy = accuracy, use_accuracy = use_acc, binarize = binarize[1])
    if (transfer_all) {
      pre2 <- pdp(X, Y3, n_hidden = n_hidden[2], plot_loss = plot_loss, 
                  constrained = constrained, n_iter = n_iter, batch_size = batch_size, learning_rates = learning_rates2, 
                  min_correct = min_correct, threshold = threshold, momentum = momentum, 
                  lambda = lambda, accuracy = accuracy, prev_wts = pre$weights, use_accuracy = use_acc, binarize = binarize[2])
    } else {
      pre2 <- pdp(X, Y3, n_hidden = n_hidden[2], plot_loss = plot_loss, 
                  constrained = constrained, n_iter = n_iter, batch_size = batch_size, learning_rates = learning_rates2, 
                  min_correct = min_correct, threshold = threshold, momentum = momentum, 
                  lambda = lambda, accuracy = accuracy, prev_wts = list(pre$weights[[1]]), use_accuracy = use_acc, binarize = binarize[2])
    }
    
    rev <- pdp(X, Y4, n_hidden = n_hidden[3], plot_loss = plot_loss, 
               constrained = constrained, n_iter = n_iter, batch_size = batch_size, learning_rates = learning_rates2, 
               prev_wts = pre2$weights, min_correct = min_correct, threshold = threshold,
               momentum = momentum, lambda = lambda, accuracy = accuracy, use_accuracy = use_acc, binarize = binarize[3])
    
    nrev <- pdp(X, Y2, n_hidden = n_hidden[4], plot_loss = plot_loss, 
                constrained = constrained, n_iter = n_iter, batch_size = batch_size, learning_rates = learning_rates3, 
                prev_wts = pre2$weights, min_correct = min_correct, threshold = threshold, 
                momentum = momentum, lambda = lambda, accuracy = accuracy, use_accuracy = use_acc, binarize = binarize[4])
    
    out$pre$iterations[sim+s] <- pre$iterations
    out$pre$loss[[sim+s]] <- pre$loss
    out$pre$performance[[sim+s]] <- pre$performance
    out$pre2$iterations[sim+s] <- pre2$iterations
    out$pre2$loss[[sim+s]] <- pre2$loss
    out$pre2$performance[[sim+s]] <- pre2$performance
    out$rev$iterations[sim+s] <- rev$iterations
    out$rev$loss[[sim+s]] <- rev$loss
    out$rev$performance[[sim+s]] <- rev$performance
    out$nrev$iterations[sim+s] <- nrev$iterations
    out$nrev$loss[[sim+s]] <- nrev$loss
    out$nrev$performance[[sim+s]] <- nrev$performance
  }
  return(out)
}


get_active_neurons <- function(weights, data, true1, true2, layer, labs = c("Big", "Small", "circle", "triangle"), 
                               activation_function = sigmoid){
  cat1 <- c()
  cat2 <- c()
  cat3 <- c()
  cat4 <- c()
  for (i in 1:length(true1)){
    f <- feedforward(data, list(weights), batch = i, activation_function = activation_function, constrained = T)
    ifelse(true1[i] == 0, 
           cat1 <- c(cat1, which(f[[layer]] == max(f[[layer]]))), 
           cat2 <- c(cat2, which(f[[layer]] == max(f[[layer]]))))
    
  }
  for (i in 1:length(true2)){
    f <- feedforward(data, list(weights), batch = i, activation_function = activation_function, constrained = T)
    ifelse(true2[i] == 0, 
           cat3 <- c(cat3, which(f[[layer]] == max(f[[layer]]))), 
           cat4 <- c(cat4, which(f[[layer]] == max(f[[layer]]))))
    
  }
  cols <- c(rgb(0.8, 0.5, 0.5, 0.8), 
            rgb(0.8, 0.8, 0.5, 0.8), 
            rgb(0.5, 0.5, 0.8, 0.8), 
            rgb(0.5, 0.7, 0.7, 0.7))
  ylim <- max(c(table(cat1), table(cat2), table(cat3), table(cat4)))+20
  par(mfrow = c(1, 1), mar = c(3, 3, 3, 3))
  brks <- seq(0.8, (length(f[[layer]])+0.2), 0.1)
  hist(cat1-0.1, col = cols[1], ylim = c(0,ylim), axes = F, 
       xlim = c(0.6, (length(f[[layer]])+0.3)), breaks = brks, 
       main = "Frequency of neuron being the most active")
  hist(cat2, add = T, col = cols[2], axes = F, breaks = brks)
  hist(cat3+0.1, add = T, col = cols[3], axes = F, breaks = brks)
  hist(cat4+0.2, add = T, col = cols[4], axes = F, breaks = brks)
  axis(2,seq(0,ylim, 100))
  axis(1,1:(length(f[[layer]])+0.3))
  legend("topright", legend = labs, fill = cols, border = F, bg = NA,
         box.lwd = 0)
}

get_active_neurons_means <- function(weights, data, true1, true2, layer, labs = c("circle left", "circle right", "big left", "big right"), 
                               activation_function = sigmoid, main = c(""), legend = T, pos_leg = "topleft", plot_lines = T){
  l <- length(feedforward(data, list(weights), batch = 1, activation_function = activation_function, constrained = T)[[layer]])
  cat1 <- rep(0, l)
  cat2 <- rep(0, l)
  cat3 <- rep(0, l)
  cat4 <- rep(0, l)
  
  for (i in 1:length(true1)){
    f <- feedforward(data, list(weights), batch = i, activation_function = activation_function, constrained = T)
    ifelse(true1[i] == 0, 
           cat1 <- cat1 + f[[layer]], 
           cat2 <- cat2 + f[[layer]])
    
  }
  for (i in 1:length(true2)){
    f <- feedforward(data, list(weights), batch = i, activation_function = activation_function, constrained = T)
    ifelse(true2[i] == 0, 
           cat3 <- cat3 + f[[layer]],
           cat4 <- cat4 + f[[layer]])
    
  }
  
  average1 <- cat1 / length(true1)
  average2 <- cat2 / length(true1)
  average3 <- cat3 / length(true1)
  average4 <- cat4 / length(true1)
  av_max <- max(average1, average2, average3, average4)

  cat1.1 <- c()
  cat2.1 <- c()
  cat3.1 <- c()
  cat4.1 <- c()
  
  for (i in 1:length(f[[layer]])){
    cat1.1 <- c(cat1.1, rep(i,round(cat1[i]*100)))
  }
  for (i in 1:length(f[[layer]])){
    cat2.1 <- c(cat2.1, rep(i,round(cat2[i]*100)))
  }
  for (i in 1:length(f[[layer]])){
    cat3.1 <- c(cat3.1, rep(i,round(cat3[i]*100)))
  }
  for (i in 1:length(f[[layer]])){
    cat4.1 <- c(cat4.1, rep(i,round(cat4[i]*100)))
  }
  
  cat1 <- cat1.1
  cat2 <- cat2.1
  cat3 <- cat3.1
  cat4 <- cat4.1
  lw <- 2
  cols <- c(rgb(0.8, 0.5, 0.5, 0.8), 
            rgb(0.8, 0.8, 0.5, 0.8), 
            rgb(0.5, 0.5, 0.8, 0.8), 
            rgb(0.5, 0.7, 0.7, 0.7))
  y_max <- max(c(table(cat1), table(cat2), table(cat3), table(cat4)))
  ylim <- y_max  + (y_max/3.4)
  av_lim <- av_max+(av_max/3.4)
  brks <- seq(0.8, (length(f[[layer]])+0.2), 0.1)
  hist(cat1-0.1, col = cols[1], ylim = c(0,ylim), axes = F, 
       xlim = c(0.6, (length(f[[layer]])+0.3)), breaks = brks, 
       main = main, ylab = "Activation", xlab = "Neuron")
  hist(cat2, add = T, col = cols[2], axes = F, breaks = brks)
  hist(cat3+0.1, add = T, col = cols[3], axes = F, breaks = brks)
  hist(cat4+0.2, add = T, col = cols[4], axes = F, breaks = brks)
  if (plot_lines){
    segments(which(table(cat1) == max(table(cat1)))-0.15, rev(range(table(cat1)))[1], 
             which(table(cat1) == min(table(cat1)))-0.15, rev(range(table(cat1)))[2], col = cols[1], lw = lw, lty = 2)
    segments(which(table(cat2) == max(table(cat2)))-0.05, rev(range(table(cat2)))[1], 
             which(table(cat2) == min(table(cat2)))-0.05, rev(range(table(cat2)))[2], col = cols[2], lw = lw, lty = 2)
    segments(which(table(cat3) == max(table(cat3)))+0.05, rev(range(table(cat3)))[1], 
             which(table(cat3) == min(table(cat3)))+0.05, rev(range(table(cat3)))[2], col = cols[3], lw = lw, lty = 2)
    segments(which(table(cat4) == max(table(cat4)))+0.15, rev(range(table(cat4)))[1], 
             which(table(cat4) == min(table(cat4)))+0.15, rev(range(table(cat4)))[2], col = cols[4], lw = lw, lty = 2)
    # abline(h = rev(range(table(cat1))), lty = c(1, 2), col = cols[1], lw = lw)
    # abline(h = rev(range(table(cat2))), lty = c(1, 2), col = cols[2], lw = lw)
    # abline(h = rev(range(table(cat3))), lty = c(1, 2), col = cols[3], lw = lw)
    # abline(h = rev(range(table(cat4))), lty = c(1, 2), col = cols[4], lw = lw)
  }
  axis(1,1:(length(f[[layer]])+0.3))
  axis(2,at = seq(0, ylim, ylim/(length(seq(0, av_lim, av_lim/(av_lim*20)))-1)), labels = seq(0, av_lim, av_lim/(av_lim*20)))
  if (legend){
    legend(pos_leg, legend = labs, fill = cols, border = F, bg = NA,
           box.lwd = 0)
  }
}


