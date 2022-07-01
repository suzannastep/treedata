library(dplyr)
library(flashier)
library(tidyverse)
library(ebnm)
library(fields)
library(dyno)
library(dequer)
source("code/drift_div_factorizations.R")

#' Helper function that returns the loading from one additional divergence factor.
#' Using the local false sign rate, it sets to zero those loadings for which we cannot be
#' confident of the sign.
#'
#' @param dat the data matrix
#' @param lfsr_tol Tolerance for local false sign rate
#' @param loading binary initial loading to specify known sparsity pattern in the loading (e.g. parental sparsity)
#' @param fl flash object containing the current fit
#' @param divprior divergence prior for loadings
#' @param Fprior prior for the factors. Defaults to a normal prior.
#' @returns the new posterior loading for the additional divergence factor
get_divergence_factor_lfsr <- function(dat,lfsr_tol,loading,fl,divprior,Fprior){
    #print("entering get divergence lfsr")
    K <- fl$n.factors
    #initializes factor to the least squares solution
    ls.soln  <- t(crossprod(loading,  dat - fitted(fl))/sum(loading))
    EF <- list(loading, ls.soln)
    #print("entering flash")
    next_fl <- fl %>%
        flash.init.factors(
            EF,
            ebnm.fn = c(divprior,Fprior)
        ) %>%
        flash.fix.factors(kset = K + 1, mode = 1L, is.fixed = (loading == 0)) %>%
        flash.backfit(kset = K + 1,extrapolate=FALSE)
    #print("entering flash")
    loading <- next_fl$L.pm[,K+1]
    lfsr <- next_fl$L.lfsr[,K+1]
    loading[lfsr > lfsr_tol] <- 0
    #print("leaving get divergence lfsr")
    return(loading)
}

#' Fits a drift factorization to the data and returns the associated flashier object
#'
#' @param dat data matrix. rows are samples, columns are features
#' @param covar If true, fits to the covariance instead of the data matrix
#' @param divprior prior for intermediate divergence loadings. Defaults to a point Laplace prior.
#' @param driftprior prior for drift loadings. Defaults to a point exponential prior.
#' @param Fprior prior for the factors. Defaults to a normal prior.
#' @param Kmax maximum number of factors to add.
#' @param lfsr_tol Tolerance for local false sign rate
#' @param min_pve If the pve for a factor is less than this tolerance, the factor is rejected
#' @param verbose.lvl The level of verbosity of the function
#' @param labels Ground truth data labels for testing purposes
lfsr_algorithm <- function(dat,
                      covar=FALSE,
                      divprior = ebnm_point_laplace,
                      driftprior = ebnm_point_exponential,
                      Fprior = ebnm_normal,
                      Kmax = Inf,
                      lfsr_tol = 1e-3,
                      min_pve = 0,
                      verbose.lvl = 0,
                      labels=NULL,
                      allfixed=FALSE,
                      S=1e-12) {
    if (covar){
        dat <- cov(t(dat))
    }
    else{
        dat <- dat
    }
    #the first loading will be the all-ones vector
    ones <- matrix(1, nrow = nrow(dat), ncol = 1)
    #first factor will be least sq soln: argmin_f ||Y - ones t(f)||_F^2
    ls.soln <- t(crossprod(ones, dat)/nrow(dat))

    #create flash object with initial drift loading and initial divergence loading
    if (verbose.lvl > 0) {cat("S =",S,"\n")}
    fl <- flash.init(dat,S=S) %>%
        flash.set.verbose(verbose.lvl) %>%
        #initialize L to be the ones vector, and F to be the least squares solution
        flash.init.factors(list(ones, ls.soln),
                           ebnm.fn = c(driftprior,Fprior)) %>%
        #only fixing the first factor, and the we want to fix row loadings, so mode=1
        flash.fix.factors(kset = 1, mode = 1) %>%
        #backfit to match the priors
        flash.backfit(extrapolate=FALSE)

    #add first divergence factor to a queue (Breadth-first)
    divergence_queue <- queue()
    new_div <- get_divergence_factor_lfsr(dat,lfsr_tol,loading=ones,fl,divprior,Fprior)
    pushback(divergence_queue,new_div)

    while(length(divergence_queue) > 0 && fl$n.factors < Kmax) {
        if (verbose.lvl > 0) {cat("Length of Queue",length(divergence_queue),"\n")}
        #pop the first divergence off the queue
        current_divergence <- pop(divergence_queue)
        #split into positive and negative parts
        splus <- matrix(1L * (current_divergence > 0), ncol = 1)
        if (sum(splus) > 0 && fl$n.factors < Kmax) {
            if ((verbose.lvl > 0)&(!is.null(labels))) {print(get_sums_by_label(labels,splus))}
            #add drift loading
            next_fl <- add_factor(dat,splus,fl,driftprior,Fprior,allfixed=allfixed)
            if (next_fl$pve[next_fl$n.factors] > min_pve){
                fl <- next_fl
                #enqueue new divergence
                new_div <- get_divergence_factor_lfsr(dat,lfsr_tol,loading=splus,fl,divprior,Fprior)
                pushback(divergence_queue,new_div)
            }
        }
        sminus <- matrix(1L * (current_divergence < 0), ncol = 1)
        if (sum(sminus) > 0 && fl$n.factors < Kmax) {
            if ((verbose.lvl > 0)&(!is.null(labels))) {print(get_sums_by_label(labels,sminus))}
            #add drift loading
            next_fl <- add_factor(dat,sminus,fl,driftprior,Fprior,allfixed=allfixed)
            if (next_fl$pve[next_fl$n.factors] > min_pve){
                fl <- next_fl
                #enqueue new divergence
                new_div <- get_divergence_factor_lfsr(dat,lfsr_tol,loading=sminus,fl,divprior,Fprior)
                pushback(divergence_queue,new_div)
            }
        }
        if (verbose.lvl > 0) {cat("Factors:", fl$n.factors, "\n")}
    }

    return(fl)
}
