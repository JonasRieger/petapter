library(ggplot2)
library(ggh4x)
library(data.table)

res = fread("results_sinnlos.csv")
res[, arch := factor(arch, levels = c("ia3", "lora", "pfeiffer"))]
res[, verbalizer := "intuitiv"]
res[grepl("sinnlos", model), verbalizer := "shuffle"]
res[grepl("a-e", model), verbalizer := "a-e"]
res[, model := gsub("_a-e", "", gsub("_sinnlos", "", model))]

# times
res[, .(time_train = mean(time_train), time_test = mean(time_test)),
    by = c("model", "verbalizer", "dataset", "arch", "pvp", "nshot")]

to_plot = res[label == "Overall" & measure == "Accuracy"]

overall = ggplot(to_plot) +
  aes(x = arch, y = value, fill = as.character(sample)) +
  geom_point(position=position_dodge(width = 0.6), alpha = 0.25) +
  geom_point(stat = "summary", fun = "mean", col = "blue", pch = 4, size = 2.5,
             position=position_dodge(width = 0.6)) +
  geom_point(aes(y = majority), position=position_dodge(width = 0.6),
             col = "blue", pch = 17, size = 2) +
  geom_point(aes(fill = NA), stat = "summary", fun = "mean", col = "red",
             pch = 4, size = 2) +
  geom_point(aes(y = majority, fill = NA), stat = "summary", fun = "mean",
             col = "red", pch = 17, size = 2) +
  xlab("") + ylab("Accuracy") + labs(fill = "Sample") +
  scale_x_discrete(labels = c(ia3 = expression((IA)^3),
                              lora = expression(LoRA),
                              pfeiffer = expression(Pfeiffer),
                              pet = expression(PET))) +
  facet_nested(model+nshot~pvp+verbalizer, scales = "free",
               labeller = labeller(
                 pvp = c(prompt = "Prompt Pattern",
                         qa = "Q&A Pattern"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0))

pdf("results_sinnlos.pdf", width = 11.5, height = 8)
print(overall)
dev.off()
