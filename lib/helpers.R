
##
## auxiliar functions
##


agg_class <-function( X, alpha, allPars )
{
  
  # Function: agg_class
  # - inputs:
  #   - X: X is a matrix of which the columns are the training vectors
  #   - alpha: vector of voting weights
  #   - allPars: matrix containing parameters of all weak learners
  # - outputs:
  #   - vector which evaluates the boosting classifier
  
  # total number of decision stumps used
  B <- length( alpha )
  
  if(B == 1)
  {
    j     <- allPars[1]
    theta <- allPars[2]
    m     <- allPars[3]
  } else {
    j     <- allPars[,1]
    theta <- allPars[,2]
    m     <- allPars[,3]
  }
  
  # initialize vector Cb(xi)
  c_b <- 0
  
  # calculate classification associated with each stump b
  for( b in 1:B )
  {
    #c_b  <- c_b + alpha[b] * ifelse( X[ , j[b] ] > theta[b], m[b], - m[b] )
    c_b  <- c_b + alpha[b] * classify( X, c( j[b], theta[b], m[b] ) )
  }
  
  # returns c_hat
  return( sign( c_b ) )
  
}

# training function
train <- function( X, w, y )
{
  
  # Function: train
  # - inputs:
  #   - X: matrix with training vector in columns
  #   - w: vector containing weights
  #   - y: vector containing class labels
  # - output:
  #   - pars: vector of parameters (j, theta, m) specifying the decision stump
  
  # function to calculate decision stump
  decision.stump <- function( y, xj, w )
  { 
    # create df to get indexes after x is sorted
    df.data <- data.frame( "index" = seq( 1, length(xj) ) , 
                           "xj" = xj, "y" = y, "w" = w )
    
    w.cum <- sapply( unique(df.data$xj), 
                     function(theta) sum( df.data$w * df.data$y * (df.data$xj <= theta) ) )
    
    # optimum theta
    opt.theta <- df.data$xj[ which.max( abs( w.cum ) ) ]
    
    # orientation of the hyperplane -OBS:RIGHT??
    m <- sign( w.cum[ which.max( abs( w.cum ) ) ] )
    
    # classify using optimal theta found
    y.pred <- ifelse( xj <= opt.theta, m, - m )
    
    # calculate cost associated with opt.theta
    cost   <-sum( w * ifelse( y.pred != y, 1, 0 ) ) / sum( w )
    
    return( c( opt.theta, cost, m ) )
  }
  
  # calculate optimal theta and associated cost for each feature
  df.res <- apply( X, MARGIN = 2, function(x) decision.stump( y, x, w ) )
  
  opt.j     <- which.min( df.res[2,] )
  opt.theta <- df.res[ 1 , opt.j ]
  m         <- df.res[ 3 , opt.j ]
  
  return(  c( opt.j, opt.theta, m ) )
  
}

# classifying function
classify <- function( X, pars )
{
  # Function: classify
  # - inputs:
  #   - X: data on which the weak learner will be evaluated
  #   - pars: parameters j, theta, m
  # - output:
  #   - label: vector with estimated classification for the data
  
  # classify using optimal parameters for weak learner found
  y.pred <- ifelse( X[ , pars[1] ] <= pars[2], pars[3], -pars[3] )
  return( y.pred )
}

# adaBoost model function
adaBoost <- function( X, y, B, w, nFolds = 5 )
{
  
  # Function: AdaBoost
  # - inputs:
  #   - X: data on which the weak learner will be evaluated
  #   - y: class labels
  #   - B:
  #   - w:
  #   - nFolds:
  # - output:
  #   - list containing training and test errors
  
  alpha          <- matrix(NA, nrow = nFolds, ncol = B)
  error.train    <- matrix(NA, nrow = nFolds, ncol = B)
  error.test     <- matrix(NA, nrow = nFolds, ncol = B)
  
  # sort indexes to be used in cross-validation
  index  <- rep( seq( 1, nFolds), nrow(X)/nFolds )
  set.seed(1)
  index  <- sample( index, replace = FALSE )
  
  # initialize train.w and list where each element is matrix of parameters for a fold
  train.w      <- matrix( NA, nrow = nFolds, ncol = round((nFolds-1)*nrow(X)/nFolds))
  list.cv.pars <- list()
  for(iFold in 1:nFolds)
  {
    train.w[iFold,] <- w[ which( index != iFold ) ]
    list.cv.pars[[iFold]] <- matrix(NA, nrow = B, ncol = 3 )
  }
  
  for( b in 1:B )
  {
    
    for( iFold in 1:nFolds )
    {
      # separate data into train and test sets
      train.X  <- X[ which( index != iFold ) , ]
      train.y  <- y[ which( index != iFold ) ]
      
      test.X  <- X[ which( index == iFold ) , ]
      test.y  <- y[ which( index == iFold )  ]
      
      # train weak learner on training set
      pars                      <- train( train.X, train.w[iFold,], train.y)
      list.cv.pars[[iFold]][b,] <- pars 
      
      # predict on training set using weak learner
      y.pred.train <- classify( train.X, pars )
      
      # compute error
      error  <- sum( train.w[iFold,] * 
                       ifelse(train.y != y.pred.train, 1, 0) ) / sum( train.w[iFold,] )
      
      # compute voting weights (scalar)
      alpha[ iFold, b ] <- log( (1-error)/error )
      
      # recompute weights
      train.w[iFold,] <- train.w[iFold,] * exp( alpha[iFold, b] * 
                                                  ifelse(train.y != y.pred.train, 1, 0) )
      
      # compute aggregate classifier on training set
      agg.clas.train <- agg_class( train.X, 
                                   alpha[iFold,][!is.na(alpha[iFold,])], 
                                   as.matrix( list.cv.pars[[iFold]]
                                              [!is.na(list.cv.pars[[iFold]][,1]),] ) )
      
      # compute aggregate classifier on test set
      agg.clas.test  <- agg_class( test.X, 
                                   alpha[iFold,][!is.na(alpha[iFold,])], 
                                   as.matrix( list.cv.pars[[iFold]]
                                              [!is.na(list.cv.pars[[iFold]][,1]),] ) )
      
      
      # compute misclassification error of aggregate classifier on train and test sets
      # train error rate
      error.train[iFold, b] <- sum( agg.clas.train != train.y )/length(train.y)
      
      # test error rate
      error.test[iFold, b]  <- sum( agg.clas.test != test.y )/length(test.y)
      
    }
    
  }
  
  return( list( "error.train" = error.train,
                "error.test" = error.test ) )
  
}
