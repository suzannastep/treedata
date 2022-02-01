library(dplyr)
library(flashier)
library(tidyverse)
library(ebnm)
library(fields)
library(dyno)
library(dequer)
source("code/drift_div_factorizations.R")

#' Fits a drift factorization to the data and returns the associated flashier object
#' 
#' @param tree tree object; output of form_tree_from_file. 
#' @param divprior prior for intermediate divergence loadings. Defaults to a point Laplace prior.
#' @param driftprior prior for drift loadings. Defaults to a point exponential prior.
#' @param Fprior prior for the factors. Defaults to a normal prior.
#' @param Kmax maximum number of factors to add.
#' @param lfsr_tol Tolerance for local false sign rate
#' @param min_pve If the pve for a factor is less than this tolerance, the factor is rejected
#' @param verbose.lvl The level of verbosity of the function
lfsr_algorithm <- function(tree,
                      divprior = prior.point.laplace(),
                      driftprior = as.prior(ebnm.fn = ebnm_point_exponential,sign=1),
                      Fprior = prior.normal(),
                      Kmax = Inf,
                      lfsr_tol = 0.95,
                      min_pve = 0,
                      verbose.lvl = 0) {
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
        flash.backfit()
    
    #add first divergence factor to a queue (Breadth-first)
    divergence_queue <- queue()
    new_div <- get_divergence_factor(dat,loading=ones,fl,divprior,Fprior)
    pushback(divergence_queue,new_div)
    
    while(length(divergence_queue) > 0 && fl$n.factors < Kmax) {
        #pop the first divergence off the queue
        current_divergence <- pop(divergence_queue)
        #set loadings to zero for samples where we cannot conclude the sign of the loading
        # TODO
        #split into positive and negative parts
        splus <- matrix(1L * (current_divergence > 0), ncol = 1)
        if (sum(splus) > 0 && fl$n.factors < Kmax) {
            if (verbose.lvl > 0) {cat(get_sums_by_label(tree,splus))}
            #add drift loading
            next_fl <- add_factor(dat,splus,fl,driftprior,Fprior)
            if (next_fl$pve[next_fl$n.factors] > min_pve){
                fl <- next_fl
                #enqueue new divergence
                new_div <- get_divergence_factor(dat,splus,fl,divprior,Fprior)
                pushback(divergence_queue,new_div)
            }
        }
        sminus <- matrix(1L * (current_divergence < 0), ncol = 1)
        if (sum(sminus) > 0 && fl$n.factors < Kmax) {
            if (verbose.lvl > 0) {cat(get_sums_by_label(tree,sminus))}
            #add drift loading
            next_fl <- add_factor(dat,sminus,fl,driftprior,Fprior)
            if (next_fl$pve[next_fl$n.factors] > min_pve){
                fl <- next_fl
                #enqueue new divergence
                new_div <- get_divergence_factor(dat,sminus,fl,divprior,Fprior)
                pushback(divergence_queue,new_div)
            }
        }
        if (verbose.lvl > 0) {cat("Factors:", fl$n.factors, "\n")}
    }
    
    return(fl)
}