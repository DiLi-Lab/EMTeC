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

rms_df <- read.csv("data/reading_measures_corrected.csv", header = TRUE, sep = "\t")

# normalize the predictors 
rms_df$surprisal_gpt2 <- scale(rms_df$surprisal_gpt2)
rms_df$zipf_freq <- scale(rms_df$zipf_freq)

# convert to factors
rms_df$model <- as.factor(rms_df$model)
rms_df$subject_id <- as.factor(rms_df$subject_id)
rms_df$decoding_strategy <- factor(rms_df$decoding_strategy, levels = c("greedy_search", "beam_search", "sampling", "topk", "topp"))
rms_df$task <- factor(rms_df$task, levels = c("article_synopsis", "non-fiction", "fiction", "poetry", "summarization", "words_given"))
rms_df$model <- factor(rms_df$model, levels = c("mistral", "phi2", "wizardlm"))

RMS <- c("TFT", "FPRT", "RRT")
PREDICTORS <- c("task", "decoding_strategy", "model")

# check if folder results exists, if not, create one
if (!file.exists("analyses/results")) {
    dir.create("analyses/results")
}

all_results <- data.frame() 

for (rm in RMS) {
    for (predictor in PREDICTORS) {
        print(paste("Reading measure:", rm, "Predictor:", predictor))
        
        # remove zeros, and log-transform the continuous RMs
        rms_df_log <- rms_df[rms_df[[rm]] != 0, ]
        rms_df_log[[rm]] <- log(rms_df_log[[rm]])

        formula <- paste(rm, "~ 1 + (1|subject_id) + word_length_with_punct + zipf_freq + surprisal_gpt2 + ", predictor, " + surprisal_gpt2:", predictor, sep = "")
        reg_model <- lmer(formula, data = rms_df_log)
        effects <- data.frame(summary(reg_model)$coefficients[-1, c("Estimate", "Pr(>|t|)", "Std. Error")])
        colnames(effects)[2] <- "pval"
        effects$effect <- rownames(effects)
        # move effect to first column
        effects <- effects[, c(ncol(effects), 1:(ncol(effects)-1))]
        row.names(effects) <- NULL
        effects$reading_measure <- rm
        effects$predictor <- predictor
        all_results <- rbind(all_results, effects)
    }
}

# check significance
all_results$significance <- is_significant(all_results$pval)
# remove "surprisal_gpt" from effect column
all_results <- all_results %>%
  mutate(effect = ifelse(grepl("^surprisal_gpt2:.*", effect), 
                          gsub("^surprisal_gpt2", "", effect), 
                          effect))
# refactor effect
all_results$effect <- factor(all_results$effect)

# write results to csv 
write.csv(all_results, "analyses/results/all_results.csv", row.names = FALSE)


# plotting 
for (pred in PREDICTORS) {
    sub_results <- all_results[all_results$predictor == pred, ]
    sub_results <- sub_results[grepl(":", sub_results$effect), ]
    
    sub_results$effect <- gsub(paste(":", pred, sep = ""), "", sub_results$effect)
    if (pred == "task") {
        sub_results$effect <- gsub("words_given", "Key Words", sub_results$effect)
        sub_results$effect <- gsub("non-fiction", "Non-Fiction", sub_results$effect)
        sub_results$effect <- gsub("poetry", "Poetry", sub_results$effect)
        sub_results$effect <- gsub("summarization", "Summarization", sub_results$effect)
        sub_results$effect <- gsub("fiction", "Fiction", sub_results$effect)
        sub_results$effect <- factor(sub_results$effect, levels = c("Non-Fiction", "Fiction", "Poetry", "Summarization", "Key Words"))
    }
    else if (pred == "decoding_strategy") {
        sub_results$effect <- gsub("beam_search", "Beam Search", sub_results$effect)
        sub_results$effect <- gsub("sampling", "Sampling", sub_results$effect)
        sub_results$effect <- gsub("topk", "Top-k", sub_results$effect)
        sub_results$effect <- gsub("topp", "Top-p", sub_results$effect)
        sub_results$effect <- factor(sub_results$effect, levels = c("Beam Search", "Sampling", "Top-k", "Top-p"))
    }
    else {
        sub_results$effect <- gsub("mistral", "Mistral", sub_results$effect)
        sub_results$effect <- gsub("phi2", "Phi2", sub_results$effect)
        sub_results$effect <- gsub("wizardlm", "WizardLM", sub_results$effect)
        sub_results$effect <- factor(sub_results$effect, levels = c("Phi2", "WizardLM"))
    }
    if (pred == "decoding_strategy") {
        pred_name <- "decoding strategy"
    }
    else {
        pred_name <- pred
    }
    ggplot(data = sub_results, aes(x = effect, y = Estimate, colour = reading_measure, shape = significance)) +
        geom_point(aes(colour = reading_measure), position = position_dodge(width = .5), size = 2) +
        geom_errorbar(aes(ymin = Estimate - Std..Error, ymax = Estimate + Std..Error), width = 0.1, position = position_dodge(width = .5)) +
        theme(axis.text.x = element_text(angle = 20, hjust = 1)) +
        theme(text = element_text(family = "sans")) +
        geom_hline(yintercept = 0, linetype = "dashed") +
        labs(shape = "Significance", colour = "Resp. Var.") +
        scale_shape_manual(values = c(1, 19)) +
        #xlab("Effect of decoding strategy (sum-contrast coded)") +
        xlab(paste("Effect of", pred_name, "(sum-contrast coded)")) +
        ylab("Coefficient estimate") +
        theme(axis.text.x = element_text(size = 10), axis.text.y = element_text(size = 10), 
              axis.title.x = element_text(size = 12), axis.title.y = element_text(size = 12), 
              legend.title = element_text(size = 12), legend.text = element_text(size = 12),
              strip.text.x = element_text(size = 10), strip.text.y = element_text(size = 10)) 
        #facet_wrap(~reading_measure, scales = "free_y")
    
    ggsave(paste("analyses/results/", pred, ".png", sep = ""), width = 10, height = 5)
}
