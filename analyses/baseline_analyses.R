#!/usr/bin/env Rscript
suppressPackageStartupMessages({
    library(boot)
    library(readr)
    library(tidyr)
    library(dplyr)
    library(gridExtra)
    library(lme4)
    library(MASS)
    library(brms)
    library(optparse)
    library(cmdstanr)
})

# Parse arguments from command line
options <- list(
    make_option(c("-r", "--responseVar"), action = "store", default = "FPRT", type = "character", help = "Response variable to model."),
    make_option(c("-i", "--iterations"), action = "store", default = 6000, type = "integer", help = "How many iterations to run.")
)

args <- parse_args(OptionParser(option_list = options))

ITERATIONS <- args$i
WUP_ITERATIONS <- 2000

options(mc.cores = parallel::detectCores())
run_loglin_model <- function(formula, data) {
    model <- brm(
        formula,
        data = data,
        family = lognormal(),
        prior = c(prior(normal(6, 1.5), class = Intercept),
                prior(normal(0, 1), class = sigma),
                prior(normal(0, 1), class = b, coef = word_length),
                prior(normal(0, 1), class = b, coef = log_lex_freq),
                prior(normal(0, 1), class = b, coef = surprisal),
                prior(normal(0, 1), class = b, coef = last_in_line)
        ),
        warmup = WUP_ITERATIONS,
        iter = ITERATIONS,
        chains = 4,
        seed = 42,
        backend = "cmdstanr"
    )
    return(model)
}

run_binary_logistic_model <- function(formula, data) {
    model <- brm(
        formula,
        family = bernoulli(link = "logit"),
        prior = c(prior(normal(0, 4), class = Intercept),
                prior(normal(0, 1), class = b, coef = word_length),
                prior(normal(0, 1), class = b, coef = log_lex_freq),
                prior(normal(0, 1), class = b, coef = surprisal),
                prior(normal(0, 1), class = b, coef = last_in_line)
        ),
        data = data,
        warmup = WUP_ITERATIONS,
        iter = ITERATIONS,
        chains = 4,
        seed = 42,
        backend = "cmdstanr"
    )
    return(model)
}

run_poisson_model <- function(formula, data) {
    model <- brm(
        formula,
        data = data,
        family = poisson(),
        warmup = WUP_ITERATIONS,
        iter = ITERATIONS,
        chains = 4,
        seed = 42,
        backend = "cmdstanr"
    )
    return(model)
}

preprocess <- function(df) {
    df$subject_id <- as.factor(df$subject_id)
    #  convert to log lex freq
    df$log_lex_freq <- scale(df$zipf_freq)
    df$word_length <- scale(df$word_length_with_punct)
    df$surprisal <- scale(df$surprisal_gpt2)
    return(df)
}

LOG_LINEAR_VARIABLES <- c("FFD", "SFD", "FD", "FPRT", "FRT", "TFT", "RRT", "word_rt")
BINARY_VARIABLES <- c("Fix", "FPF", "RR", "FPReg")
COUNT_VARIABLES <- c("TFC")

data_raw <- read.csv("data/reading_measures_corrected.csv", header = TRUE, sep = "\t")
data <- preprocess(data_raw)

formula <- paste0(
    args$r, " ~ (1 | subject_id) + word_length + log_lex_freq + surprisal + last_in_line"
)
#  check if response variable is a log-linear variable
if (args$r %in% LOG_LINEAR_VARIABLES) {
    #  remove all rows with 0 values in the response variable
    data_in <- data[data[, args$r] != 0, ]
    #  Do not log transform -> this is done in the model call
    # data_in[, args$r] <- log(data_in[, args$r])
    model <- run_loglin_model(formula, data_in)
} else if (args$r %in% BINARY_VARIABLES) {
    model <- run_binary_logistic_model(formula, data)
} else if (args$r %in% COUNT_VARIABLES) {
    model <- run_poisson_model(formula, data)
}

saveRDS(model, file = paste0("model_fits/decoding_", args$r, ".rds"))