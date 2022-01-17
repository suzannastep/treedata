library(dplyr)
library(flashier)
library(tidyverse)
library(ebnm)
library(fields)
library(dyno)
library(dequer)

#TODO document this code

#just a helper for debugging
get_sums_by_label <- function(tree,loading){
  summary <- as.data.frame(table(tree$csv$Labels))
  summary$NumActivated <- 0
  idx <- 1
  for(label in rownames(summary)){
    start_idx <- idx
    idx <- idx + summary[label,"Freq"]
    summary[label,"NumActivated"] <- sum(loading[start_idx:(idx-1)])
  }
  return(summary)
}

#helper function that adds one factor
add_factor <- function(dat,loading,fl,prior,Fprior){
  K <- fl$n.factors
  ls.soln  <- t(crossprod(loading,  dat - fitted(fl))/sum(loading))
  EF <- list(loading, ls.soln)
  next_fl <- fl %>%
    flash.init.factors(
      EF,
      prior.family = c(prior,Fprior)
    ) %>%
    flash.fix.loadings(kset = K + 1, mode = 1L, is.fixed = (EF[[1]] == 0)) %>%
    flash.backfit(kset = K + 1)
  return(next_fl)
}

#helper function that gets one divergence factor
get_divergence_factor <- function(dat,loading,fl,divprior,Fprior){
  K <- fl$n.factors
  ls.soln  <- t(crossprod(loading,  dat - fitted(fl))/sum(loading))
  EF <- list(loading, ls.soln)
  next_fl <- fl %>%
    flash.init.factors(
      EF,
      prior.family = c(divprior,Fprior)
    ) %>%
    flash.fix.loadings(kset = K + 1, mode = 1L, is.fixed = (EF[[1]] == 0)) %>%
    flash.backfit(kset = K + 1)
  return(next_fl$loadings.pm[[1]][,K+1])
}

#fits a drift factorization to the data
drift_fit <- function(tree,dat,filename,
                    divprior = prior.point.laplace(),
                    driftprior = as.prior(ebnm.fn = ebnm_point_exponential,sign=1),
                    Fprior = prior.normal(),
                    Kmax = Inf,
                    min_pve = 0,
                    verbose.lvl = 0,
                    eps=1e-2) {
  #the first loading will be the all-ones vector
  ones <- matrix(1, nrow = nrow(dat), ncol = 1)
  #first factor will be least sq soln: argmin_f ||Y - ones t(f)||_F^2
  ls.soln <- t(crossprod(ones, dat)/nrow(dat))

  #create flash object with initial drift loading and initial divergence loading
  fl <- flash.init(dat) %>%
    flash.set.verbose(verbose.lvl) %>%
    #initialize L to be the ones vector, and F to be the least squares solution
    flash.init.factors(list(ones, ls.soln),
                       prior.family = c(driftprior,Fprior)) %>%
    #only fixing the first factor, and the we want to fix row loadings, so mode=1
    flash.fix.loadings(kset = 1, mode = 1) %>%
    #backfit to match the priors
    flash.backfit() %>%
    #add initial divergence loading
    flash.add.greedy(
      Kmax = 1,
      prior.family = c(divprior, Fprior)
    )

  #add divergence factor to a queue (Breadth-first)
  divergence_queue <- queue()
  pushback(divergence_queue,fl$loadings.pm[[1]][,2])
  #remove divergence factor
  fl <- fl %>%
    flash.remove.factors(kset = 2)
  K <- 1

  while(length(divergence_queue) > 0 && K < Kmax) {
    #pop the first divergence off the queue
    current_divergence <- pop(divergence_queue)
    #split into loadings for positive and negative parts (1-0 indicator vectors)
    splus <- matrix(1L * (current_divergence > eps), ncol = 1)
    if (sum(splus) > 0) {
      if (verbose.lvl > 0) {
        cat(get_sums_by_label(tree,splus))
      }
      #add drift loading
      next_fl <- add_factor(dat,splus,fl,driftprior,Fprior)
      if (next_fl$pve[K + 1] > min_pve){
        fl <- next_fl
        K <- fl$n.factors
        #enqueue new divergence
        new_div <- get_divergence_factor(dat,splus,fl,divprior,Fprior)
        pushback(divergence_queue,new_div)
      }
    }
    sminus <- matrix(1L * (current_divergence < -eps), ncol = 1)
    if (sum(sminus) > 0 && K < Kmax) {
      if (verbose.lvl > 0) {
        cat(get_sums_by_label(tree,splus))
      }
      #add drift loading
      next_fl <- add_factor(dat,sminus,fl,driftprior,Fprior)
      if (next_fl$pve[K + 1] > min_pve){
        fl <- next_fl
        K <- fl$n.factors
        #enqueue new divergence
        new_div <- get_divergence_factor(dat,sminus,fl,divprior,Fprior)
        pushback(divergence_queue,new_div)
      }
    }

    if (verbose.lvl > 0) {
      cat("K:", K, "\n")
    }

    # image.plot(fl$loadings.pm[[1]],xlab = fl$n.factors)
  }

  L <- fl$loadings.pm[[1]]
  F <- fl$loadings.pm[[2]]
  scale <- fl$loadings.scale

  write.table(L,file=paste(filename,"L.csv",sep=''),sep=',')
  write.table(F,file=paste(filename,"F.csv",sep=''),sep=',')
  write.table(scale,file=paste(filename,"scale.csv",sep=''),sep=',')
  write.table(fl$pve,file=paste(filename,"pve.csv",sep=''),sep=',')

  return(fl)
}

#fits a divergence factorization to the data
div_fit <- function(tree,dat,filename,
                      divprior = prior.point.laplace(),
                      driftprior = as.prior(ebnm.fn = ebnm_point_exponential,sign=1),
                      Fprior = prior.normal(),
                      Kmax = Inf,
                      min_pve = 0,
                      verbose.lvl = 0,
                      eps=1e-2) {
  #the first loading will be the all-ones vector
  ones <- matrix(1, nrow = nrow(dat), ncol = 1)
  #first factor will be least sq soln: argmin_f ||Y - ones t(f)||_F^2
  ls.soln <- t(crossprod(ones, dat)/nrow(dat))

  #create flash object with initial drift loading and initial divergence loading
  fl <- flash.init(dat) %>%
    flash.set.verbose(verbose.lvl) %>%
    #initialize L to be the ones vector, and F to be the least squares solution
    flash.init.factors(list(ones, ls.soln),
                       prior.family = c(driftprior,Fprior)) %>%
    #only fixing the first factor, and the we want to fix row loadings, so mode=1
    flash.fix.loadings(kset = 1, mode = 1) %>%
    #backfit to match the priors
    flash.backfit() %>%
    #add initial divergence loading
    flash.add.greedy(
      Kmax = 1,
      prior.family = c(divprior, Fprior)
    )

  #add divergence factor to a queue (Breadth-first)
  divergence_queue <- queue()
  pushback(divergence_queue,fl$loadings.pm[[1]][,2])

  K <- fl$n.factors

  while(length(divergence_queue) > 0 && K < Kmax) {
    #pop the first divergence off the queue
    current_divergence <- pop(divergence_queue)

    #add drift loading
    snonzero <- matrix(1L * (abs(current_divergence) > eps), ncol = 1)
    if (sum(snonzero) > 0 && (K != 2)) {
      if (verbose.lvl > 0) {
        cat(get_sums_by_label(tree,snonzero))
      }
      #add drift loading
      next_fl <- add_factor(dat,snonzero,fl,driftprior,Fprior)
      if (next_fl$pve[K + 1] > min_pve){
        fl <- next_fl
        K <- fl$n.factors
      }
    }
    #try to add a divergence within the positive loadings of the current divergence
    splus <- matrix(1L * (current_divergence > eps), ncol = 1)
    if (sum(splus) > 0 && K < Kmax) {
      #add divergence loading
      next_fl <- add_factor(dat,splus,fl,divprior,Fprior)
      if (next_fl$pve[K + 1] > min_pve){
        fl <- next_fl
        K <- fl$n.factors
        new_div <- fl$loadings.pm[[1]][,K]
        #enqueue new divergence
        pushback(divergence_queue,new_div)
      }
    }
    #try to add a divergence within the negative loadings of the current divergence
    sminus <- matrix(1L * (current_divergence < -eps), ncol = 1)
    if (sum(sminus) > 0 && K < Kmax) {
      #add divergence loading
      next_fl <- add_factor(dat,sminus,fl,divprior,Fprior)
      if (next_fl$pve[K + 1] > min_pve){
        fl <- next_fl
        K <- fl$n.factors
        new_div <- fl$loadings.pm[[1]][,K]
        #enqueue new divergence
        pushback(divergence_queue,new_div)
      }
    }

    if (verbose.lvl > 0) {
      cat("K:", K, "\n")
    }
    # image.plot(fl$loadings.pm[[1]],xlab = fl$n.factors)
  }

  L <- fl$loadings.pm[[1]]
  F <- fl$loadings.pm[[2]]
  scale <- fl$loadings.scale

  write.table(L,file=paste(filename,"L.csv",sep=''),sep=',')
  write.table(F,file=paste(filename,"F.csv",sep=''),sep=',')
  write.table(scale,file=paste(filename,"scale.csv",sep=''),sep=',')
  write.table(fl$pve,file=paste(filename,"pve.csv",sep=''),sep=',')

  return(fl)
}

#fits a divergence factorization to the covariance matrix
div_cov_fit <- function(covmat, filename, prior = prior.point.laplace(), Kmax = 1000) {
  fl <- div_fit(covmat, filename, prior, Kmax)
  s2 <- max(0, mean(diag(covmat) - diag(fitted(fl))))
  s2_diff <- Inf
  while(s2 > 0 && abs(s2_diff - 1) > 1e-4) {
    covmat_minuss2 <- covmat - diag(rep(s2, ncol(covmat)))
    fl <- div_fit(covmat_minuss2,filename, prior, Kmax)
    old_s2 <- s2
    s2 <- max(0, mean(diag(covmat) - diag(fitted(fl))))
    s2_diff <- s2 / old_s2
  }

  fl$ebcovmf_s2 <- s2

  return(fl)
}

form_tree_from_file <- function(filename){
  tree <- vector(mode="list")
  tree$csv <- read.csv(filename,row.names=1)
  tree$raw <- tree$csv %>%
    select(Raw0:Raw499)
  tree$matrix <- as.matrix(tree$raw)
  tree$dimred <- tree$csv %>%
    select(tsne0:tsne1)
  #As input, dynwrap requires raw counts and normalized (log2) expression data.
  tree$counts <- round(2**(tree$raw)-1)
  tree$counts[tree$counts<0] <- 0
  tree$dataset <- wrap_expression(
    expression = tree$matrix,
    counts = as.matrix(tree$counts)
  )
  tree$dataset <- add_prior_information(
    tree$dataset,
    start_id = "Row0",
    start_n = 1,
    end_n = 4,
  )
  tree$trajectory <- vector(mode="list")
  return(tree)
}

run_methods <- function(tree,outfile,Kmax,eps){
  #drift factorization method
  tree$trajectory$drift <- drift_fit(tree,
                                     tree$matrix,
                                     paste(outfile,"drift",sep=""),
                                     Kmax = Kmax,
                                     eps = eps)
  tree$trajectory$div <- div_fit(tree,
                                     tree$matrix,
                                     paste(outfile,"div",sep=""),
                                     Kmax = Kmax,
                                     eps = eps)
  return(tree)
}
