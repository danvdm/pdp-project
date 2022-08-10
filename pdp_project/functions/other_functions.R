
# adds some random noise to the input
sandwicher <- function(input, buffer = 4, pos_shift = F){
  out <- matrix(nrow = nrow(input), ncol = ncol(input)+buffer)
  for (row in 1:nrow(input)){
    if (pos_shift){
      first <- sample(0:buffer, 1)
    } else {
      first <- ceiling(buffer/2)
    }
    out[row, ] <- c(runif(first), input[row, ], runif(buffer - first))
  }
  return(out)
}

noiser <- function(input, amount = 0.1){
  out <- input - (input*2-1)*abs(matrix(rnorm(nrow(input)*ncol(input), 0, sd = amount), ncol = ncol(input)))
  return(out)
}

rbinder <- function(input, times = 2){
  out <- input
  if (times > 1){
    for (rep in 2:times){
      out <- rbind(out, input)
    }
  }
  return(out)
}





