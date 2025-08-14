library(ggplot2)
library(ggh4x)
library(data.table)

res = rbind(fread("results.csv"), fread("results_pet.csv"))
res[, arch := factor(arch, levels = c("ia3", "lora", "pfeiffer", "pet"))]
res[arch == "pet", pvp := "pet"]

# times
res[, .(time_train = mean(time_train), time_test = mean(time_test)),
    by = c("arch", "pvp", "nshot")]

## Overall F1
to_plot = res[label == "Overall" & measure == "F1"]

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
  xlab("") + ylab("Macro-F1") + labs(fill = "Sample") +
  scale_x_discrete(labels = c(ia3 = expression((IA)^3),
                              lora = expression(LoRA),
                              pfeiffer = expression(Pfeiffer),
                              pet = expression(PET))) +
  facet_nested(sampling+nshot~pvp, scales = "free",
               labeller = labeller(
                 sampling = c(equal = "Equal Sampling",
                              random = "Random Sampling",
                              stratified = "Stratified Sampling"),
                 pvp = c(prompt = "PETapter",
                         standard_head = "Standard Head",
                         pet = "PET"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0))

## Precision
to_plot = res[label != "Overall" & measure == "Precision" &
                (arch == "pfeiffer" | arch == "pet") & nshot == 250]

prec = ggplot(to_plot) +
  aes(x = label, y = value, fill = as.character(sample)) +
  geom_point(position=position_dodge(width = 0.6), alpha = 0.25) +
  geom_point(stat = "summary", fun = "mean", col = "blue", pch = 4, size = 2.5,
             position=position_dodge(width = 0.6)) +
  geom_point(aes(y = majority), position=position_dodge(width = 0.6),
             col = "blue", pch = 17, size = 2) +
  geom_point(aes(fill = NA), stat = "summary", fun = "mean", col = "red",
             pch = 4, size = 2) +
  geom_point(aes(y = majority, fill = NA), stat = "summary", fun = "mean",
             col = "red", pch = 17, size = 2) +
  xlab("") + ylab("Precision") + labs(fill = "Sample") +
  scale_x_discrete(labels = c(ia3 = expression((IA)^3),
                              lora = expression(LoRA),
                              pfeiffer = expression(Pfeiffer),
                              pet = expression(PET))) +
  facet_nested(sampling+nshot~pvp, scales = "free",
               labeller = labeller(
                 sampling = c(equal = "Equal Sampling",
                              random = "Random Sampling",
                              stratified = "Stratified Sampling"),
                 pvp = c(prompt = "PETapter",
                         standard_head = "Standard Head",
                         pet = "PET"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0)) +
  ggtitle("Precision values for Pfeiffer adapter with 250 shots")

## Recall
to_plot = res[label != "Overall" & measure == "Recall" &
                (arch == "pfeiffer" | arch == "pet") & nshot == 250]

rec = ggplot(to_plot) +
  aes(x = label, y = value, fill = as.character(sample)) +
  geom_point(position=position_dodge(width = 0.6), alpha = 0.25) +
  geom_point(stat = "summary", fun = "mean", col = "blue", pch = 4, size = 2.5,
             position=position_dodge(width = 0.6)) +
  geom_point(aes(y = majority), position=position_dodge(width = 0.6),
             col = "blue", pch = 17, size = 2) +
  geom_point(aes(fill = NA), stat = "summary", fun = "mean", col = "red",
             pch = 4, size = 2) +
  geom_point(aes(y = majority, fill = NA), stat = "summary", fun = "mean",
             col = "red", pch = 17, size = 2) +
  xlab("") + ylab("Recall") + labs(fill = "Sample") +
  scale_x_discrete(labels = c(ia3 = expression((IA)^3),
                              lora = expression(LoRA),
                              pfeiffer = expression(Pfeiffer),
                              pet = expression(PET))) +
  facet_nested(sampling+nshot~pvp, scales = "free",
               labeller = labeller(
                 sampling = c(equal = "Equal Sampling",
                              random = "Random Sampling",
                              stratified = "Stratified Sampling"),
                 pvp = c(prompt = "PETapter",
                         standard_head = "Standard Head",
                         pet = "PET"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0)) +
  ggtitle("Recall values for Pfeiffer adapter with 250 shots")

# F1
to_plot = res[label != "Overall" & measure == "F1" &
                (arch == "pfeiffer" | arch == "pet") & nshot == 250]

f1_label = ggplot(to_plot) +
  aes(x = label, y = value, fill = as.character(sample)) +
  geom_point(position=position_dodge(width = 0.6), alpha = 0.25) +
  geom_point(stat = "summary", fun = "mean", col = "blue", pch = 4, size = 2.5,
             position=position_dodge(width = 0.6)) +
  geom_point(aes(y = majority), position=position_dodge(width = 0.6),
             col = "blue", pch = 17, size = 2) +
  geom_point(aes(fill = NA), stat = "summary", fun = "mean", col = "red",
             pch = 4, size = 2) +
  geom_point(aes(y = majority, fill = NA), stat = "summary", fun = "mean",
             col = "red", pch = 17, size = 2) +
  xlab("") + ylab("Recall") + labs(fill = "Sample") +
  scale_x_discrete(labels = c(ia3 = expression((IA)^3),
                              lora = expression(LoRA),
                              pfeiffer = expression(Pfeiffer),
                              pet = expression(PET))) +
  facet_nested(sampling+nshot~pvp, scales = "free",
               labeller = labeller(
                 sampling = c(equal = "Equal Sampling",
                              random = "Random Sampling",
                              stratified = "Stratified Sampling"),
                 pvp = c(prompt = "PETapter",
                         standard_head = "Standard Head",
                         pet = "PET"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0)) +
  ggtitle("F1 values for Pfeiffer adapter with 250 shots")

## Precision
to_plot = res[label != "Overall" & measure == "Precision" &
                (arch == "pfeiffer" | arch == "pet") & nshot == 100]

prec_100 = ggplot(to_plot) +
  aes(x = label, y = value, fill = as.character(sample)) +
  geom_point(position=position_dodge(width = 0.6), alpha = 0.25) +
  geom_point(stat = "summary", fun = "mean", col = "blue", pch = 4, size = 2.5,
             position=position_dodge(width = 0.6)) +
  geom_point(aes(y = majority), position=position_dodge(width = 0.6),
             col = "blue", pch = 17, size = 2) +
  geom_point(aes(fill = NA), stat = "summary", fun = "mean", col = "red",
             pch = 4, size = 2) +
  geom_point(aes(y = majority, fill = NA), stat = "summary", fun = "mean",
             col = "red", pch = 17, size = 2) +
  xlab("") + ylab("Precision") + labs(fill = "Sample") +
  scale_x_discrete(labels = c(ia3 = expression((IA)^3),
                              lora = expression(LoRA),
                              pfeiffer = expression(Pfeiffer),
                              pet = expression(PET))) +
  facet_nested(sampling+nshot~pvp, scales = "free",
               labeller = labeller(
                 sampling = c(equal = "Equal Sampling",
                              random = "Random Sampling",
                              stratified = "Stratified Sampling"),
                 pvp = c(prompt = "PETapter",
                         standard_head = "Standard Head",
                         pet = "PET"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0)) +
  ggtitle("Precision values for Pfeiffer adapter with 100 shots")

## Recall
to_plot = res[label != "Overall" & measure == "Recall" &
                (arch == "pfeiffer" | arch == "pet") & nshot == 100]

rec_100 = ggplot(to_plot) +
  aes(x = label, y = value, fill = as.character(sample)) +
  geom_point(position=position_dodge(width = 0.6), alpha = 0.25) +
  geom_point(stat = "summary", fun = "mean", col = "blue", pch = 4, size = 2.5,
             position=position_dodge(width = 0.6)) +
  geom_point(aes(y = majority), position=position_dodge(width = 0.6),
             col = "blue", pch = 17, size = 2) +
  geom_point(aes(fill = NA), stat = "summary", fun = "mean", col = "red",
             pch = 4, size = 2) +
  geom_point(aes(y = majority, fill = NA), stat = "summary", fun = "mean",
             col = "red", pch = 17, size = 2) +
  xlab("") + ylab("Recall") + labs(fill = "Sample") +
  scale_x_discrete(labels = c(ia3 = expression((IA)^3),
                              lora = expression(LoRA),
                              pfeiffer = expression(Pfeiffer),
                              pet = expression(PET))) +
  facet_nested(sampling+nshot~pvp, scales = "free",
               labeller = labeller(
                 sampling = c(equal = "Equal Sampling",
                              random = "Random Sampling",
                              stratified = "Stratified Sampling"),
                 pvp = c(prompt = "PETapter",
                         standard_head = "Standard Head",
                         pet = "PET"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0)) +
  ggtitle("Recall values for Pfeiffer adapter with 100 shots")

# F1
to_plot = res[label != "Overall" & measure == "F1" &
                (arch == "pfeiffer" | arch == "pet") & nshot == 100]

f1_label_100 = ggplot(to_plot) +
  aes(x = label, y = value, fill = as.character(sample)) +
  geom_point(position=position_dodge(width = 0.6), alpha = 0.25) +
  geom_point(stat = "summary", fun = "mean", col = "blue", pch = 4, size = 2.5,
             position=position_dodge(width = 0.6)) +
  geom_point(aes(y = majority), position=position_dodge(width = 0.6),
             col = "blue", pch = 17, size = 2) +
  geom_point(aes(fill = NA), stat = "summary", fun = "mean", col = "red",
             pch = 4, size = 2) +
  geom_point(aes(y = majority, fill = NA), stat = "summary", fun = "mean",
             col = "red", pch = 17, size = 2) +
  xlab("") + ylab("Recall") + labs(fill = "Sample") +
  scale_x_discrete(labels = c(ia3 = expression((IA)^3),
                              lora = expression(LoRA),
                              pfeiffer = expression(Pfeiffer),
                              pet = expression(PET))) +
  facet_nested(sampling+nshot~pvp, scales = "free",
               labeller = labeller(
                 sampling = c(equal = "Equal Sampling",
                              random = "Random Sampling",
                              stratified = "Stratified Sampling"),
                 pvp = c(prompt = "PETapter",
                         standard_head = "Standard Head",
                         pet = "PET"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0)) +
  ggtitle("F1 values for Pfeiffer adapter with 100 shots")

## Precision
to_plot = res[label != "Overall" & measure == "Precision" &
                (arch == "pfeiffer" | arch == "pet") & nshot == 10]

prec_10 = ggplot(to_plot) +
  aes(x = label, y = value, fill = as.character(sample)) +
  geom_point(position=position_dodge(width = 0.6), alpha = 0.25) +
  geom_point(stat = "summary", fun = "mean", col = "blue", pch = 4, size = 2.5,
             position=position_dodge(width = 0.6)) +
  geom_point(aes(y = majority), position=position_dodge(width = 0.6),
             col = "blue", pch = 17, size = 2) +
  geom_point(aes(fill = NA), stat = "summary", fun = "mean", col = "red",
             pch = 4, size = 2) +
  geom_point(aes(y = majority, fill = NA), stat = "summary", fun = "mean",
             col = "red", pch = 17, size = 2) +
  xlab("") + ylab("Precision") + labs(fill = "Sample") +
  scale_x_discrete(labels = c(ia3 = expression((IA)^3),
                              lora = expression(LoRA),
                              pfeiffer = expression(Pfeiffer),
                              pet = expression(PET))) +
  facet_nested(sampling+nshot~pvp, scales = "free",
               labeller = labeller(
                 sampling = c(equal = "Equal Sampling",
                              random = "Random Sampling",
                              stratified = "Stratified Sampling"),
                 pvp = c(prompt = "PETapter",
                         standard_head = "Standard Head",
                         pet = "PET"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0)) +
  ggtitle("Precision values for Pfeiffer adapter with 10 shots")

## Recall
to_plot = res[label != "Overall" & measure == "Recall" &
                (arch == "pfeiffer" | arch == "pet") & nshot == 10]

rec_10 = ggplot(to_plot) +
  aes(x = label, y = value, fill = as.character(sample)) +
  geom_point(position=position_dodge(width = 0.6), alpha = 0.25) +
  geom_point(stat = "summary", fun = "mean", col = "blue", pch = 4, size = 2.5,
             position=position_dodge(width = 0.6)) +
  geom_point(aes(y = majority), position=position_dodge(width = 0.6),
             col = "blue", pch = 17, size = 2) +
  geom_point(aes(fill = NA), stat = "summary", fun = "mean", col = "red",
             pch = 4, size = 2) +
  geom_point(aes(y = majority, fill = NA), stat = "summary", fun = "mean",
             col = "red", pch = 17, size = 2) +
  xlab("") + ylab("Recall") + labs(fill = "Sample") +
  scale_x_discrete(labels = c(ia3 = expression((IA)^3),
                              lora = expression(LoRA),
                              pfeiffer = expression(Pfeiffer),
                              pet = expression(PET))) +
  facet_nested(sampling+nshot~pvp, scales = "free",
               labeller = labeller(
                 sampling = c(equal = "Equal Sampling",
                              random = "Random Sampling",
                              stratified = "Stratified Sampling"),
                 pvp = c(prompt = "PETapter",
                         standard_head = "Standard Head",
                         pet = "PET"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0)) +
  ggtitle("Recall values for Pfeiffer adapter with 10 shots")

# F1
to_plot = res[label != "Overall" & measure == "F1" &
                (arch == "pfeiffer" | arch == "pet") & nshot == 10]

f1_label_10 = ggplot(to_plot) +
  aes(x = label, y = value, fill = as.character(sample)) +
  geom_point(position=position_dodge(width = 0.6), alpha = 0.25) +
  geom_point(stat = "summary", fun = "mean", col = "blue", pch = 4, size = 2.5,
             position=position_dodge(width = 0.6)) +
  geom_point(aes(y = majority), position=position_dodge(width = 0.6),
             col = "blue", pch = 17, size = 2) +
  geom_point(aes(fill = NA), stat = "summary", fun = "mean", col = "red",
             pch = 4, size = 2) +
  geom_point(aes(y = majority, fill = NA), stat = "summary", fun = "mean",
             col = "red", pch = 17, size = 2) +
  xlab("") + ylab("Recall") + labs(fill = "Sample") +
  scale_x_discrete(labels = c(ia3 = expression((IA)^3),
                              lora = expression(LoRA),
                              pfeiffer = expression(Pfeiffer),
                              pet = expression(PET))) +
  facet_nested(sampling+nshot~pvp, scales = "free",
               labeller = labeller(
                 sampling = c(equal = "Equal Sampling",
                              random = "Random Sampling",
                              stratified = "Stratified Sampling"),
                 pvp = c(prompt = "PETapter",
                         standard_head = "Standard Head",
                         pet = "PET"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0)) +
  ggtitle("F1 values for Pfeiffer adapter with 10 shots")

pdf("results_ukraine.pdf", width = 11.5, height = 8)
print(overall)
print(prec)
print(rec)
print(f1_label)
print(prec_100)
print(rec_100)
print(f1_label_100)
print(prec_10)
print(rec_10)
print(f1_label_10)
dev.off()
