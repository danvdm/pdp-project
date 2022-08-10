

get_attributes <- function(weights_list){
  list_out <- list("n_layers" = c(),
                   "n_neurons" = c(),
                   "weights" = c(), 
                   "biases" = list())
  list_out$n_layers <- length(weights_list)+1
  list_out$n_neurons[1] <- nrow(weights_list[[1]])-1
  list_out$n_neurons[2] <- ncol(weights_list[[1]])-1
  if (length(weights_list)>1){
    for (mat in c(2:(length(weights_list)))){
      list_out$n_neurons <- c(list_out$n_neurons, ncol(weights_list[[mat]])-1)
    }
  }
  for (mat in c(1:length(weights_list))){
    list_out$weights[[mat]] <- weights_list[[mat]][-1, -1, drop = F]
    list_out$biases[[mat]] <- list(weights_list[[mat]][, 1][-1],
                                   weights_list[[mat]][1, ][-1])
  }
  return(list_out)
}

divider <- function(min, max, n){
  seq(min, max, max / (n + 1))[2:(n+1)]
}

plot_net <- function(model, scale_factor = 1, min_opacity = 50, 
                       do_no_scale = F, labels_in = NA, labels_in_2 = NA, 
                       labels_hidden = NA, labels_out = NA, labels_out_2 = NA, 
                       text_size = 1, min_lw = 0.1, main = NA, plot_bias = F, 
                     no_bias_in_input = F){
  x.range<-c(0,100)
  y.range<-c(0,100)
  
  max_lw = 5 * scale_factor
  point_size = 4 * scale_factor
  
  if (no_bias_in_input){
    for (m in 1:length(model)){
      model[[m]] <- cbind(0, rbind(0, model[[m]]))
      if(plot_bias){
        plot_bias <- F
        print("You indicated that no bias was present - plot_bias was set to FALSE")
      }
    }
  }
  
  attributes <- get_attributes(model)
  
  y_vals_layers <- rev(divider(0, 100, attributes$n_layers))
  y_vals_biases <- rep(c(y_vals_layers + y_vals_layers[length(y_vals_layers)]/3)[-1], 
                       each = 2)
  
  xvals <- list()
  for (i in 1:length(attributes$n_neurons)){
    xvals[[i]] <- divider(0, 100, attributes$n_neurons[i])
  }
  
  xvals_biases_r <- c()
  xvals_biases_l <- c()

  for (i in 1:length(model)){
    xvals_biases_r <- c(xvals_biases_r, xvals[[i]][1]/2 + xvals[[i]][length(xvals[[i]])])
    xvals_biases_l <- c(xvals_biases_l, xvals[[i]][1]/2)
    #xvals_biases <- c(xvals_biases, c(xvals_biases_l[i], xvals_biases_r[i]))
  }
  
  xvals_biases <- c()
  for (i in 1:length(xvals_biases_l)){
    xvals_biases <- c(xvals_biases, c(xvals_biases_l[i], xvals_biases_r[i]))
  }
  
  
  range <- range(abs(c(unlist(attributes$weights), unlist(attributes$biases))))
  
  plot(x.range,y.range,type='n',axes=F,ylab='',xlab='', main = main)

  
  for (i in 1:(length(model))){
    for (line_end in c(1: attributes$n_neurons[i+1])){
      for (line_start in c(1:attributes$n_neurons[i])){
        lw = scales::rescale(abs(attributes$weights[[i]][line_start, line_end, drop = F]), 
                             from = range, 
                             to=c(min_lw, max_lw))
        if (attributes$weights[[i]][line_start, line_end] == 0){lw = 0}
        if (attributes$weights[[i]][line_start, line_end, drop = F] <= 0){
          col = rgb(100, 100, 255, max = 255, 
                    alpha = scales::rescale(abs(attributes$weights[[i]][line_start, line_end, drop = F]), 
                                            from = range, 
                                            to=c(min_opacity, 255)))
        }
        else
          col = rgb(255, 100, 100, max = 255, 
                    alpha = scales::rescale(attributes$weights[[i]][line_start, line_end, drop = F], 
                                            from = range, 
                                            to=c(min_opacity, 255)))
        segments(x0 = xvals[[i]][line_start], y0 =  y_vals_layers[i], 
                 x1 = xvals[[i+1]][line_end], y1 = y_vals_layers[i+1], 
                 col = col, lw = lw)
      }
    }
    
    if (plot_bias == T){
      for (line_start in c(1: attributes$n_neurons[i])){
        lw = scales::rescale(abs(attributes$biases[[i]][[1]][line_start]), 
                             from = range, 
                             to=c(1, max_lw))
        if (attributes$weights[[i]][line_start, line_end] == 0){lw = 0}
        if (attributes$biases[[i]][[1]][line_start]<= 0){
          col = rgb(100, 100, 255, max = 255, 
                    alpha = scales::rescale(abs(abs(attributes$biases[[i]][[1]][line_start])), 
                                            from = range, 
                                            to=c(min_opacity, 255)))
        }else
          col = rgb(255, 100, 100, max = 255, 
                    alpha = scales::rescale(abs(attributes$biases[[i]][[1]][line_start]), 
                                            from = range, 
                                            to=c(min_opacity, 255)))
        segments(x0 = xvals[[i]][line_start], y0 =  y_vals_layers[i], 
                 x1 = xvals_biases_l[i], y1 = unique(y_vals_biases)[i], 
                 col = col, lw = lw)
      }
      for (line_start in c(1: attributes$n_neurons[i+1])){ 
        lw = scales::rescale(abs(attributes$biases[[i]][[2]][line_start]), 
                             from = range, 
                             to=c(1, max_lw))
        if (attributes$biases[[i]][[2]][line_start]<= 0){
          col = rgb(100, 100, 255, max = 255, 
                    alpha = scales::rescale(abs(abs(attributes$biases[[i]][[2]][line_start])), 
                                            from = range, 
                                            to=c(min_opacity, 255)))
        }else
          col = rgb(255, 100, 100, max = 255, 
                    alpha = scales::rescale(abs(attributes$biases[[i]][[2]][line_start]), 
                                            from = range, 
                                            to=c(min_opacity, 255)))
        segments(x0 = xvals[[i+1]][line_start], y0 =  y_vals_layers[i+1], 
                 x1 = xvals_biases_r[i], y1 = unique(y_vals_biases)[i], 
                 col = col, lw = lw)
      }
      points(xvals_biases, y_vals_biases, 
             pch = 20, cex = point_size, col = "gray")
    }
    points(xvals[[i]], rep(y_vals_layers[i], length(xvals[[i]])), 
           pch = 20, cex = point_size, col = "gray")
  }
  points(xvals[[i+1]], rep(y_vals_layers[i+1], length(xvals[[i+1]])), 
         pch = 20, cex = point_size, col = "gray")
}


find_mode <- function(x) {
  u <- unique(x)
  tab <- tabulate(match(x, u))
  return(u[tab == max(tab)][1])
}

find_mode_freq <- function(x) {
  u <- unique(x)
  return(max(tabulate(match(x, u))))
}

plot_freqs <- function(input, breaks = 100, xlim = c(0, 3000), main = NA, xlab = NA, 
                       ylab = NA, ylim = c(0, 20), stat = mean, pos_leg = "top"){
  dat <- list(input$pre$iterations, input$rev$iterations, input$nrev$iterations)
  min_max <- c(floor(range(unlist(dat))/breaks)[1], ceiling(range(unlist(dat))/breaks)[2])*breaks
  cols <- c(rgb(0.3, 1, 0.3, 0.5), 
            rgb(1, 0.3, 0.3, 0.5), 
            rgb(0.3, 0.3, 1, 0.5))
  if(missing(xlim)){
    xlim <- c(0, max(unlist(dat))+breaks)
  }
  if(missing(ylim)){
    ylim <- c(0, find_mode_freq(round(unlist(dat)/(breaks)))+5)
  }
  par(cex.main = 1.5, mgp = c(3.5, 1, 0), cex.lab = 1.2 , font.lab = 2, cex.axis = 1.3, bty = "n", las = 1)
  hist(unlist(dat[1]), xlim = xlim, breaks = seq(min_max[1],min_max[2], breaks),
       ylim = ylim, col = cols[1], main = main, xlab = xlab, ylab = ylab)
  hist(unlist(dat[2]), add = T, breaks = seq(min_max[1],min_max[2], breaks), 
       col = cols[2])
  hist(unlist(dat[3]),add = T, breaks = seq(min_max[1],min_max[2], breaks),
       col = cols[3])
  text(paste(round(stat(unlist(dat[1])), 2)), x = stat(unlist(dat[1])), 
       y = find_mode_freq(round(unlist(dat[1])/(breaks*1.5))))
  abline(v = stat(unlist(dat[1])), col = cols[1])
  text(paste(round(stat(unlist(dat[2])), 2)), x = stat(unlist(dat[2])), 
       y = find_mode_freq(round(unlist(dat[2])/(breaks*1.5))))
  abline(v = stat(unlist(dat[2])), col = cols[2])
  text(paste(round(stat(unlist(dat[3])), 2)), x = stat(unlist(dat[3])), 
       y = find_mode_freq(round(unlist(dat[3])/(breaks))))
  abline(v = stat(unlist(dat[3])), col = cols[3])
  legend(pos_leg, legend = c("Pre", "Rev", "Non-Rev"), box.lwd = 0, fill = cols, 
         border = T, bg="transparent")
}




