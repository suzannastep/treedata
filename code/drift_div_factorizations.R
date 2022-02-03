library(dplyr)
library(flashier)
library(tidyverse)
library(ebnm)
library(fields)
library(dyno)
library(dequer)

#' Helper function for debugging. Shows how many samples with each label type are activated on 
#' a particular loading
#' 
#' @param tree tree object; output of form_tree_from_file that has been used in computing a trajectory
#' @param loading the loading in question 
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

#' Helper function that adds one factor to the flash object.
#' 
#' @param dat the data matrix
#' @param loading binary initial loading to specify the sparsity pattern in the loading
#' @param fl flash object to add the factor to
#' @param prior prior for loadings
#' @param Fprior prior for the factors. Defaults to a normal prior.
#' @returns a new flash object with a new factor such that the associated loading matches the 
#' sparsity pattern from loading
add_factor <- function(dat,loading,fl,prior,Fprior){
  K <- fl$n.factors
  #initializes factor to the least squares solution
  ls.soln  <- t(crossprod(loading,  dat - fitted(fl))/sum(loading))
  EF <- list(loading, ls.soln)
  #create new flash object
  next_fl <- fl %>%
    flash.init.factors(
      EF,
      prior.family = c(prior,Fprior)
    ) %>%
    flash.fix.loadings(kset = K + 1, mode = 1L, is.fixed = (loading == 0)) %>%
    #only backfit the most recently added factor
    flash.backfit(kset = K + 1)
  return(next_fl)
}

#' Helper function that returns the loading from one additional divergence factor
#' 
#' @param dat the data matrix
#' @param loading binary initial loading to specify known sparsity pattern in the loading (e.g. parental sparsity)
#' @param fl flash object containing the current fit
#' @param divprior divergence prior for loadings
#' @param Fprior prior for the factors. Defaults to a normal prior.
#' @returns the new posterior loading for the additional divergence factor
get_divergence_factor <- function(dat,loading,fl,divprior,Fprior){
  K <- fl$n.factors
  #initializes factor to the least squares solution
  ls.soln  <- t(crossprod(loading,  dat - fitted(fl))/sum(loading))
  EF <- list(loading, ls.soln)
  next_fl <- fl %>%
    flash.init.factors(
      EF,
      prior.family = c(divprior,Fprior)
    ) %>%
    flash.fix.loadings(kset = K + 1, mode = 1L, is.fixed = (loading == 0)) %>%
    flash.backfit(kset = K + 1)
  return(next_fl$loadings.pm[[1]][,K+1])
}

#' Fits a drift factorization to the data
#' 
#' @param tree tree object; output of form_tree_from_file. 
#' @param divprior prior for intermediate divergence loadings. Defaults to a point Laplace prior.
#' @param driftprior prior for drift loadings. Defaults to a point exponential prior.
#' @param Fprior prior for the factors. Defaults to a normal prior.
#' @param Kmax maximum number of factors to add.
#' @param min_pve If the pve for a factor is less than this tolerance, the factor is rejected
#' @param verbose.lvl The level of verbosity of the function
#' @param eps Tolerance for nonzero values in the loadings. 
drift_fit <- function(tree,
                    divprior = prior.point.laplace(),
                    driftprior = as.prior(ebnm.fn = ebnm_point_exponential,sign=1),
                    Fprior = prior.normal(),
                    Kmax = Inf,
                    min_pve = 0,
                    verbose.lvl = 0,
                    eps=1e-2) {
  dat <- tree$matrix
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
        cat(get_sums_by_label(tree,sminus))
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
  }

  return(fl)
}

#' Fits a divergence factorization to the data
#' 
#' @param tree tree object; output of form_tree_from_file from fileIO_plotting.R. 
#' @param divprior prior for intermediate divergence loadings. Defaults to a point Laplace prior.
#' @param driftprior prior for drift loadings. Defaults to a point exponential prior.
#' @param Fprior prior for the factors. Defaults to a normal prior.
#' @param Kmax maximum number of factors to add.
#' @param min_pve If the pve for a factor is less than this tolerance, the factor is rejected
#' @param verbose.lvl The level of verbosity of the function
#' @param eps Tolerance for nonzero values in the loadings. 
div_fit <- function(tree,
                      divprior = prior.point.laplace(),
                      driftprior = as.prior(ebnm.fn = ebnm_point_exponential,sign=1),
                      Fprior = prior.normal(),
                      Kmax = Inf,
                      min_pve = 0,
                      verbose.lvl = 0,
                      eps=1e-2) {
  dat <- tree$matrix
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
    # TODO I think this is why the divergence factorization plots look weird?
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
  }

  return(fl)
}

#' fits a divergence factorization to the covariance matrix
#' this function is less well tested
#' 
#' @param covmat the covariance matrix
#' @param prior prop for the loadings. Defaults to a point Laplace prior.
#' @param Kmax the maximum number of factors to add. Defaults to 1000 
div_cov_fit <- function(covmat, prior = prior.point.laplace(), Kmax = 1000) {
  fl <- div_fit(covmat, prior, Kmax)
  s2 <- max(0, mean(diag(covmat) - diag(fitted(fl))))
  s2_diff <- Inf
  while(s2 > 0 && abs(s2_diff - 1) > 1e-4) {
    covmat_minuss2 <- covmat - diag(rep(s2, ncol(covmat)))
    fl <- div_fit(covmat_minuss2, prior, Kmax)
    old_s2 <- s2
    s2 <- max(0, mean(diag(covmat) - diag(fitted(fl))))
    s2_diff <- s2 / old_s2
  }

  fl$ebcovmf_s2 <- s2

  return(fl)
}

#' Run drift and divergence factorizations
#' 
#' @param tree a vector that is interpreted as a tree object; the output of form_tree_from_file from fileIO_plotting.R
#' @param Kmax parameter for the maximum number of factors to add
#' @param eps tolerance for when values are nonzero
#' 
#' This function does not return anything, but instead saves the results into tree$trajectory
run_methods <- function(tree,Kmax,eps){
  #drift factorization method
  tree$trajectory$drift <- drift_fit(tree,
                                     tree$matrix,
                                     Kmax = Kmax,
                                     eps = eps)
  #divergence factorization method
  tree$trajectory$div <- div_fit(tree,
                                     tree$matrix,
                                     Kmax = Kmax,
                                     eps = eps)
  return(tree)
}
