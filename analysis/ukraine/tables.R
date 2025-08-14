library(data.table)
library(xtable)

res = rbind(fread("results.csv"), fread("results_pet.csv"))
res[, arch := factor(arch, levels = c("ia3", "lora", "pfeiffer", "pet"))]

f1 = res[label == "Overall" & measure == "F1",
         .(mean = mean(value), sd = sd(value),
           majority = mean(majority), sd_majority = sd(majority)),
         by = c("arch", "pvp", "nshot", "sampling")]
setkeyv(f1, c("nshot", "sampling", "pvp", "arch"))
f1[, cell := paste0(sprintf("%.2f", f1$mean),
                    " +-.", sprintf("%03.0f", 1000*f1$sd))]

f1_label = res[label != "Overall" & measure == "F1" &
                 (arch == "pfeiffer" | arch == "pet") & nshot == 250,
               .(mean = mean(value), sd = sd(value),
                 majority = mean(majority), sd_majority = sd(majority)),
               by = c("arch", "pvp", "sampling", "label")]
setkeyv(f1_label, c("label", "sampling", "pvp"))
f1_label[, cell := paste0(sprintf("%.2f", f1_label$mean),
                          " +-.", sprintf("%03.0f", 1000*f1_label$sd))]

prec = res[label != "Overall" & measure == "Precision" &
             (arch == "pfeiffer" | arch == "pet") & nshot == 250,
           .(mean = mean(value), sd = sd(value),
             majority = mean(majority), sd_majority = sd(majority)),
           by = c("arch", "pvp", "sampling", "label")]
setkeyv(prec, c("label", "sampling", "pvp"))
prec[, cell := paste0(sprintf("%.2f", prec$mean),
                      " +-.", sprintf("%03.0f", 1000*prec$sd))]

rec = res[label != "Overall" & measure == "Recall" &
            (arch == "pfeiffer" | arch == "pet") & nshot == 250,
          .(mean = mean(value), sd = sd(value),
            majority = mean(majority), sd_majority = sd(majority)),
          by = c("arch", "pvp", "sampling", "label")]
setkeyv(rec, c("label", "sampling", "pvp"))
rec[, cell := paste0(sprintf("%.2f", rec$mean),
                     " +-.", sprintf("%03.0f", 1000*rec$sd))]

##

print(xtable(matrix(f1$cell, ncol=7, byrow = TRUE)), include.rownames = F)
print(xtable(matrix(prec$cell, ncol=9, byrow = TRUE)[, c(1:3, 7:9)]), include.rownames = F)
print(xtable(matrix(rec$cell, ncol=9, byrow = TRUE)[, c(1:3, 7:9)]), include.rownames = F)
print(xtable(matrix(f1_label$cell, ncol=9, byrow = TRUE)[, c(1:3, 7:9)]), include.rownames = F)

print(xtable(matrix(prec$cell, ncol=9, byrow = TRUE)[, 4:9]), include.rownames = F)
print(xtable(matrix(rec$cell, ncol=9, byrow = TRUE)[, 4:9]), include.rownames = F)
print(xtable(matrix(f1_label$cell, ncol=9, byrow = TRUE)[, 4:9]), include.rownames = F)



