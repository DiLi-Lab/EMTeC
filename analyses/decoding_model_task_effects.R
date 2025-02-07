#setwd("../")

# load the necessary packages
source("analyses/packages.R")

# clear all variables
rm(list = ls())

set.seed(123)
theme_set(theme_light())
options(digits = 8)
options(dplyr.summarise.inform = TRUE)

z_score <- function(x) {
    return((x - mean(x)) / sd(x))
}

z_score_test <- function(x, sd) {
    return((x - mean(x)) / sd)
}

remove_outlier <- function(df, reading_measure) {
    reading_times <- as.numeric(df[[reading_measure]])
    z_score <- z_score(reading_times)
    abs_z_score <- abs(z_score)
    df$outlier <- abs_z_score > 3
    # print number of outliers / total number of reading times
    print(paste(sum(df$outlier), "/", length(df$outlier)))
    # remove outliers
    df <- df[df$outlier == FALSE, ]
    return(df)
}

preprocess <- function(df, predictors_to_normalize, is_linear) {
    # first, copy df in order to not overwrite original
    df_copy <- df
    df_copy$subj_id <- as.factor(df_copy$subject_id)
    #  convert to log lex freq
    df_copy$log_lex_freq <- as.numeric(df_copy$zipf_freq)

    # normalize baseline predictors
    df_copy$log_lex_freq <- scale(df_copy$log_lex_freq)
    df_copy$word_length <- scale(df_copy$word_length_with_punct)

    # normalize surprisal/entropy predictors
    for (predictor in predictors_to_normalize) {
        df_copy[[predictor]] <- as.numeric(df_copy[[predictor]])
        df_copy[[predictor]] <- scale(df_copy[[predictor]])
    }
    return(df_copy)
}

is_significant <- function(p_value, alpha = 0.05) {
    ifelse(p_value < alpha, "sig.", "not sig.")
}

named.contr.sum<-function(x, ...) {
    if (is.factor(x)) {
        x <- levels(x)
    } else if (is.numeric(x) & length(x)==1L) {
        stop("cannot create names with integer value. Pass factor levels")
    }
    x<-contr.sum(x, ...)
    colnames(x) <- paste0(rownames(x)[1:(nrow(x)-1)],'_vs_grandmean')
    return(x)
}

rms_df <- read.csv("../decoding/data/reading_measures_corrected.csv", header = TRUE, sep = "\t")

# normalize the predictors 
rms_df$surprisal_gpt2 <- scale(rms_df$surprisal_gpt2)
rms_df$zipf_freq <- scale(rms_df$zipf_freq)

# convert to factors
rms_df$model <- as.factor(rms_df$model)
rms_df$subject_id <- as.factor(rms_df$subject_id)

# rms_df$decoding_strategy <- factor(rms_df$decoding_strategy, levels = c("greedy_search", "beam_search", "sampling", "topk", "topp"))
rms_df$decoding_strategy <- factor(rms_df$decoding_strategy, levels = c("beam_search", "sampling", "topk", "topp", "greedy_search"))
contrasts(rms_df$decoding_strategy) <- named.contr.sum(levels(rms_df$decoding_strategy))
# rms_df$task <- factor(rms_df$task, levels = c("article_synopsis", "non-fiction", "fiction", "poetry", "summarization", "words_given"))
rms_df$task <- factor(rms_df$task, levels = c("non-fiction", "fiction", "poetry", "summarization", "words_given", "article_synopsis"))
contrasts(rms_df$task) <- named.contr.sum(levels(rms_df$task))
# rms_df$model <- factor(rms_df$model, levels = c("mistral", "phi2", "wizardlm"))
rms_df$model <- factor(rms_df$model, levels = c("phi2", "wizardlm", "mistral"))
contrasts(rms_df$model) <- named.contr.sum(levels(rms_df$model))

RMS <- c("TFT", "FPRT", "RRT")
RMS <- c("RRT")
PREDICTORS <- c("task", "decoding_strategy", "model")

# check if folder results exists, if not, create one
if (!file.exists("analyses/results")) {
    dir.create("analyses/results")
}

priors_rt <- c(
  prior(normal(6, 1), class = Intercept),
  prior(normal(0, 1), class = b),
  prior(exponential(2), class = sd),
  prior(exponential(2), class = sigma),
  prior(lkj(2), class = cor)
)


for (rm in RMS) {
    print(paste("Reading measure:", rm))
    # remove zeros, and log-transform the continuous RMs
    rms_df_log <- rms_df[rms_df[[rm]] != 0, ]
    # rms_df_log[[rm]] <- log(rms_df_log[[rm]])

    # formula for including all predictors 
    formula_lsf <- paste(rm, "~ 1 + decoding_strategy + model + task + (1 + decoding_strategy + model + task | subject_id) ", sep = "")

    # fitting the models 
    print(paste("fitting model with formula:", formula_lsf))
    reg_model_lsf <- brm(formula_lsf, 
                         data = rms_df_log,
                         chains = 4,
                         family = lognormal(),
                         warmup = 1000,
                         iter = 2000,
                         file = paste0("analyses/results/", "emtec_brms_no_lex_feat_sumcode_", rm, ".rds"),
                         cores = 4,
                         backend = "cmdstan",
                         silent = 0
    )
}