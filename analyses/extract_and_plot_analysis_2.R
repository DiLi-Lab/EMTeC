#!/usr/bin/env Rscript
source("analyses/packages.R")

# clear all vars
rm(list = ls())

myspread <- function(df, key, value) {
  # quote key
  keyq <- rlang::enquo(key)
  # break value vector into quotes
  valueq <- rlang::enquo(value)
  s <- rlang::quos(!!valueq)
  df %>%
    gather(variable, value, !!!s) %>%
    unite(temp, !!keyq, variable) %>%
    spread(temp, value)
}


test_model_fit <- readRDS("analyses/model_fits/emtec_brms_no_lex_feat_sumcode_TFT.rds")
mod_sum <- summary(test_model_fit)

# get rownames of test$fixed
eff_names <- rownames(mod_sum$fixed)
# remove intercept
fixef_names <- eff_names[eff_names != "Intercept"]

# fixef_predictors <- c("word_length", "log_lex_freq", "surprisal", "last_in_line")
reading_measures <- c("FPRT", "TFT", "RRT")

df_all <- data.frame()
for (reading_measure in reading_measures) {
  model_fit <- readRDS(paste0("analyses/model_fits/emtec_brms_no_lex_feat_sumcode_", reading_measure, ".rds"))
  for (predictor in eff_names) {
    summary <- summary(model_fit)
    summary <- summary$fixed
    estimates <- summary[predictor, c("Estimate", "l-95% CI", "u-95% CI")]
    df_all <- rbind(df_all, data.frame(predictor = predictor, reading_measure = reading_measure, estimates))
  }
}
colnames(df_all) <- c("predictor", "reading_measure", "m", "lower", "upper")
row.names(df_all) <- NULL

df_all_wide <- df_all %>%
  gather(variable, value, -(predictor:reading_measure)) %>%
  unite(temp, predictor, variable) %>%
  spread(temp, value)

to_backtf <- c("FPRT", "TFT", "RRT")
df_backtf <- data.frame()
for (rm in reading_measures) {
  #  get subset
  df_rm <- df_all_wide[df_all_wide$reading_measure == rm, ]
  if (rm %in% to_backtf) {
    for (predictor in fixef_names) {
      this_row <- data.frame(
        predictor = predictor,
        reading_measure = rm,
        m = as.vector(exp(df_rm["Intercept_m"]) * exp(df_rm[paste0(predictor, "_m")]) - exp(df_rm["Intercept_m"]))[[1]],
        lower = as.vector(exp(df_rm["Intercept_m"]) * exp(df_rm[paste0(predictor, "_lower")]) - exp(df_rm["Intercept_m"]))[[1]],
        upper = as.vector(exp(df_rm["Intercept_m"]) * exp(df_rm[paste0(predictor, "_upper")]) - exp(df_rm["Intercept_m"]))[[1]]
      )
      df_backtf <- rbind(df_backtf, this_row)
    }
  }
}

#  get df with all reading measures without c("FPRT", "TFT") from df_all
df_all_no_backtf <- df_all[!df_all$reading_measure %in% to_backtf, ]
#  remove Intercept predictor
df_all_no_backtf <- df_all_no_backtf[df_all_no_backtf$predictor != "Intercept", ]
#  bind back together
df_final <- rbind(df_all_no_backtf, df_backtf)

# new variable: effect_name, set all to na
df_final$effect_name <- NA
# set effect names for all predictors: if task in predictor, set to task, if decoding_strategy in predictor, set to decoding_strategy, if model in predictor, set to model
df_final$effect_name <- ifelse(str_detect(df_final$predictor, "task"), "task", df_final$effect_name)
df_final$effect_name <- ifelse(str_detect(df_final$predictor, "decoding_strategy"), "decoding_strategy", df_final$effect_name)
df_final$effect_name <- ifelse(str_detect(df_final$predictor, "model"), "model", df_final$effect_name)
# if still Na set to lexical features
df_final$effect_name <- ifelse(is.na(df_final$effect_name), "lexical_features", df_final$effect_name)

df_final$predictor <- as.factor(df_final$predictor)

levels(df_final$predictor) <- c("Beam search", "Sampling", "Top-k", "Top-p", "Phi-2", "Wizard-LM", "Fiction", "Non-fiction", "Poetry", "Summarization", "Words given")

effect.labs <- c("Decoding Strategy", "Model", "Text Type")
names(effect.labs) <- c("decoding_strategy", "model", "task")


ggplot(data = df_final, aes(x = predictor, y = m, colour = reading_measure)) +
  geom_point(
    position = position_dodge(width = .5), size = 1.3
  ) +
  geom_errorbar(aes(ymin = lower, ymax = upper),
    width = .05, position = position_dodge(width = .5), linewidth = 0.6
  ) +
  #  scale_y_continuous(labels = function(x) format(x, scientific = TRUE)) +
  facet_grid(.~effect_name, scales = "free_x", space = "free", labeller = labeller(effect_name =effect.labs)) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  ylab("Effect size") +
  xlab("Predictor") +
  theme_light() +
  #  increase font size
    theme(text = element_text(family = "sans"), axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 15), axis.title = element_text(size = 15),
  strip.text = element_text(size = 15), legend.text = element_text(size = 15), legend.title = element_text(size = 15)) +
  # increase font size of strip of facets and light theme
  theme(axis.text.x = element_text(angle = 45, hjust = 1), legend.position = "bottom") +
  # rename label of colour legend
  guides(colour = guide_legend(title = "Reading measure"))

ggsave("effect_sizes_sum_coded_no_lex_feat.pdf", width = 12, height = 7, dpi = 150)
