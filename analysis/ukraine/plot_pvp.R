library(ggplot2)
library(ggh4x)
library(data.table)

res = fread("results_pvp.csv")

## Overall F1
to_plot = res[label == "Overall" & measure == "F1"]

overall = ggplot(to_plot) +
  aes(x = verbalizer, y = value, fill = as.character(sample)) +
  geom_point(position=position_dodge(width = 0.6), alpha = 0.25) +
  geom_point(stat = "summary", fun = "mean", col = "blue", pch = 4, size = 2.5,
             position=position_dodge(width = 0.6)) +
  geom_point(aes(y = majority), position=position_dodge(width = 0.6),
             col = "blue", pch = 17, size = 2) +
  geom_point(aes(fill = NA), stat = "summary", fun = "mean", col = "red",
             pch = 4, size = 2) +
  geom_point(aes(y = majority, fill = NA), stat = "summary", fun = "mean",
             col = "red", pch = 17, size = 2) +
  xlab("") + ylab("Macro-F1") + labs(fill = "Sample") +
  scale_x_discrete(labels = c(alpha = "Alpha",
                              shuffle = "Shuffle",
                              normal = "Normal")) +
  facet_nested(sampling+nshot~pvp, scales = "free",
               labeller = labeller(
                 sampling = c(equal = "Equal Sampling",
                              random = "Random Sampling",
                              stratified = "Stratified Sampling"),
                 pvp = c(prompt = "Prompt",
                         no = "No"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0))
