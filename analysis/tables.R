library(data.table)
library(xtable)

res = rbind(fread("results.csv"), fread("results_pet.csv"))
res[, arch := factor(arch, levels = c("ia3", "lora", "pfeiffer", "pet"))]

tab = res[label == "Overall" & measure == "Accuracy",
          .(mean = mean(value), sd = sd(value),
            majority = mean(majority), sd_majority = sd(majority)),
          by = c("model", "arch", "pvp", "nshot", "dataset")]
setkeyv(tab, c("model", "nshot", "dataset", "pvp", "arch"))
tab[, cell := paste0(sprintf("%.3f", tab$mean),
                     " +-.", sprintf("%03.0f", 1000*tab$sd))]
##

print(xtable(matrix(tab$cell, ncol=11, byrow = TRUE)), include.rownames = F)
