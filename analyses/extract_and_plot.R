#!/usr/bin/env Rscript
library(boot)
library(readr)
library(tidyr)
library(dplyr)
library(stringr)
library(grid)
library(gridExtra)
library(ggplot2)
library(distributional)
library(ggdist)
library(cowplot)
library(patchwork)
library(wesanderson)
library(RColorBrewer)
library(colorspace)
library(lme4)
library(MASS)
library(brms)
library(rstan)
library(testit) # for assert
library(viridis)

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

predictors <- c("Intercept", "word_length", "log_lex_freq", "surprisal", "last_in_line")
fixef_predictors <- c("word_length", "log_lex_freq", "surprisal", "last_in_line")
reading_measures <- c("Fix", "FPReg", "FPRT", "TFT")

df_all <- data.frame()
for (reading_measure in reading_measures) {
  model_fit <- readRDS(paste0("./model_fits/decoding_", reading_measure, ".rds"))
  for (predictor in predictors) {
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

to_backtf <- c("FPRT", "TFT")
df_backtf <- data.frame()
for (rm in reading_measures) {
  #  get subset
  df_rm <- df_all_wide[df_all_wide$reading_measure == rm, ]
  if (rm %in% to_backtf) {
    for (predictor in fixef_predictors) {
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

# reorder factors for plotting (like above)
df_final$predictor <- factor(df_final$predictor, levels = c("word_length", "log_lex_freq", "surprisal", "last_in_line"))
#  rename predictors
df_final$predictor <- factor(df_final$predictor, labels = c("Word length", "Lexical frequency", "Surprisal", "Last in line"))
# rename reading measures
df_final$reading_measure <- factor(df_final$reading_measure, labels = c("Fixated", "First-pass regression", "First-pass reading time", "Total fixation time"))

#  remove last_in_line predictor
df_final <- df_final[df_final$predictor != "Last in line", ]
#  drop unused levels
df_final$predictor <- droplevels(df_final$predictor)

ggplot(data = df_final, aes(x = predictor, y = m, colour = predictor)) +
  geom_point(
    position = position_dodge(width = .5), size = 1
  ) +
  geom_errorbar(aes(ymin = lower, ymax = upper),
    width = .1, position = position_dodge(width = .5), linewidth = 0.4
  ) +
  #  scale_y_continuous(labels = function(x) format(x, scientific = TRUE)) +
  facet_wrap(~reading_measure, scales = "free_y", ncol = 4) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  ylab("Effect size") +
  xlab("Predictor") +
  # theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  #  scale_colour_viridis(discrete = TRUE, option = "H") +
  scale_colour_manual(values = wes_palette("Zissou1")[c(1, 3, 5)]) +
  #  increase font size
  theme(text = element_text(family = "sans"), axis.text.x = element_text(size = 11), axis.text.y = element_text(size = 11), axis.title = element_text(size = 13)) +
  theme(legend.position = "bottom") + #  remove legend for color
  guides(color = "none")

ggsave("effect_sizes.pdf", width = 16, height = 5, dpi = 150)
