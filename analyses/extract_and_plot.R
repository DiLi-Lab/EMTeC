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

predictors <- c("Intercept", "word_length", "log_lex_freq", "surprisal", "last_in_line")
fixef_predictors <- c("word_length", "log_lex_freq", "surprisal", "last_in_line")
reading_measures <- c("Fix", "FPReg", "FPRT", "TFT")

df_all <- data.frame()
for (reading_measure in reading_measures) {
  model_fit <- readRDS(paste0("./analyses/model_fits/decoding_", reading_measure, ".rds"))
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

# new variable variable type: binary (Fix or Fpreg)  vs continuous (FPRT or TFT)
df_final$variable_type <- ifelse(df_final$reading_measure %in% c("Fixated", "First-pass regression"), "Binary", "Continuous")

ggplot(data = df_final, aes(x = predictor, y = m, fill = reading_measure, colour = reading_measure)) +
  geom_point(
    position = position_dodge(width = .5), size = 1.3
  ) +
  geom_errorbar(aes(ymin = lower, ymax = upper),
    width = .05, position = position_dodge(width = .5), linewidth = 0.6
  ) +
  facet_wrap(~variable_type, scales = "free", nrow=1) +
  geom_hline(yintercept = 0, linetype = "dashed") +
  ylab("Effect size") +
  xlab("Predictor") +
  # theme(axis.text.x = element_text(angle = 20, hjust = 1)) +
  #  scale_colour_viridis(discrete = TRUE, option = "H") +
  scale_colour_manual(values = wes_palette("Zissou1")[c(1, 2, 4, 5)]) +
  #  increase font size (also of color / fill legned)
  theme_light() +
  theme(text = element_text(family = "sans"), axis.text.x = element_text(size = 15), axis.text.y = element_text(size = 15), axis.title = element_text(size = 15),
  strip.text = element_text(size = 15), legend.text = element_text(size = 15), legend.title = element_text(size = 15)) +
  # increase font size of strip of facets
  # color / fill legend at bottom
  theme(legend.position = "bottom") +
  # title for color and fill
  labs(fill = "Reading measure", color = "Reading measure")

ggsave("effect_sizes.pdf", width = 12, height = 7, dpi = 150)
