library(ggplot2)
library(ggh4x)
library(data.table)

res = rbind(fread("results.csv"), fread("results_pet.csv"))
res[, arch := factor(arch, levels = c("ia3", "lora", "pfeiffer", "pet"))]
pet_ref = fread("pet_ref.csv")

# times
res[, .(time_train = mean(time_train), time_test = mean(time_test)),
    by = c("model", "dataset", "arch", "pvp", "nshot")]

## Large
to_plot = res[label == "Overall" & measure == "Accuracy" & model == "roberta-large"]

overall_large = ggplot(to_plot) +
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
  facet_nested(dataset+nshot~pvp, scales = "free",
               labeller = labeller(
                 dataset = c(ag = "AG's News",
                             yelp = "Yelp Full",
                             yahoo = "Yahoo Questions"),
                 pvp = c(prompt = "Prompt Pattern",
                         qa = "Q&A Pattern",
                         standard_head = "Standard Head"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0)) +
  ggtitle("RoBERTa Large")

## Base
to_plot = res[label == "Overall" & measure == "Accuracy" & model == "roberta"]

overall_base = ggplot(to_plot) +
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
  facet_nested(dataset+nshot~pvp, scales = "free",
               labeller = labeller(
                 dataset = c(ag = "AG's News",
                             yelp = "Yelp Full",
                             yahoo = "Yahoo Questions"),
                 pvp = c(prompt = "Prompt Pattern",
                         qa = "Q&A Pattern",
                         standard_head = "Standard Head"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "none",
        axis.text.x = element_text(vjust=0)) +
  ggtitle("RoBERTa Base")

if(FALSE){
  ggplot(to_plot) +
    aes(x = arch, y = value, fill = as.character(sample)) +
    geom_hline(data = pet_ref, aes(yintercept = value), col = "orange") +
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
                                pfeiffer = expression(Pfeiffer))) +
    facet_nested(dataset+nshot~pvp, scales = "free_y",
                 labeller = labeller(
                   dataset = c(ag = "AG's News",
                               yelp = "Yelp Full",
                               yahoo = "Yahoo Questions"),
                   pvp = c(prompt = "Prompt Pattern",
                           qa = "Q&A Pattern",
                           standard_head = "Standard Head"),
                   nshot = function(x) paste0("n=", x))) +
    theme(legend.position = "none",
          axis.text.x = element_text(vjust=0))
}

temp = pet_ref[,c("dataset", "pvp", "nshot", "value")]
temp[, mean_accuracy := value]
temp[, model := "PET"]

comp = merge(res[label == "Overall" & measure == "Accuracy",
                 .(mean_accuracy = mean(value), mean_majority_accuracy = mean(majority)),
                 by = c("model", "dataset", "arch", "pvp", "nshot")],
             temp, all = TRUE)

to_plot = res[label == "Overall" & measure == "Accuracy"]
ggplot(to_plot) +
  aes(x = arch, y = value, col = model) +
  geom_hline(data = pet_ref, aes(yintercept = value), col = "orange") +
  #geom_point(position=position_dodge(width = 0.6), alpha = 0.25) +
  #geom_point(stat = "summary", fun = "mean", col = "blue", pch = 4, size = 2.5,
  #           position=position_dodge(width = 0.6)) +
  #geom_point(aes(y = majority), position=position_dodge(width = 0.6),
  #           col = "blue", pch = 17, size = 2) +
  geom_point(stat = "summary", fun = "mean",
             pch = 4, size = 2, position=position_dodge(width = 0.4)) +
  geom_point(aes(y = majority), stat = "summary", fun = "mean",
             pch = 17, size = 2, position=position_dodge(width = 0.4)) +
  xlab("") + ylab("Accuracy") + labs(col = "Model") +
  scale_x_discrete(labels = c(ia3 = expression((IA)^3),
                              lora = expression(LoRA),
                              pfeiffer = expression(Pfeiffer))) +
  facet_nested(dataset+nshot~pvp, scales = "free_y",
               labeller = labeller(
                 dataset = c(ag = "AG's News",
                             yelp = "Yelp Full",
                             yahoo = "Yahoo Questions"),
                 pvp = c(prompt = "Prompt Pattern",
                         qa = "Q&A Pattern"),
                 nshot = function(x) paste0("n=", x))) +
  theme(legend.position = "top",
        axis.text.x = element_text(vjust=0))


pdf("results.pdf", width = 11.5, height = 8)
print(overall_large)
print(overall_base)
dev.off()
