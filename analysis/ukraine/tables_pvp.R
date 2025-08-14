library(data.table)
library(xtable)

res = fread("results_pvp.csv")

f1 = res[label == "Overall" & measure == "F1",
         .(mean = mean(value), sd = sd(value),
           majority = mean(majority), sd_majority = sd(majority)),
         by = c("pattern", "verbalizer", "nshot", "sampling")]
setkeyv(f1, c("nshot", "sampling", "pattern", "verbalizer"))
f1[, cell := paste0(sprintf("%.2f", f1$mean),
                    " +-.", sprintf("%03.0f", 1000*f1$sd))]

f1_label = res[label != "Overall" & measure == "F1" & nshot == 100,
               .(mean = mean(value), sd = sd(value),
                 majority = mean(majority), sd_majority = sd(majority)),
               by = c("pattern", "verbalizer", "sampling", "label")]
setkeyv(f1_label, c("label", "sampling", "pattern", "verbalizer"))
f1_label[, cell := paste0(sprintf("%.2f", f1_label$mean),
                          " +-.", sprintf("%03.0f", 1000*f1_label$sd))]

prec = res[label != "Overall" & measure == "Precision" & nshot == 100,
           .(mean = mean(value), sd = sd(value),
             majority = mean(majority), sd_majority = sd(majority)),
           by = c("pattern", "verbalizer", "sampling", "label")]
setkeyv(prec, c("label", "sampling", "pattern", "verbalizer"))
prec[, cell := paste0(sprintf("%.2f", prec$mean),
                      " +-.", sprintf("%03.0f", 1000*prec$sd))]

rec = res[label != "Overall" & measure == "Recall" & nshot == 100,
          .(mean = mean(value), sd = sd(value),
            majority = mean(majority), sd_majority = sd(majority)),
          by = c("pattern", "verbalizer", "sampling", "label")]
setkeyv(rec, c("label", "sampling", "pattern", "verbalizer"))
rec[, cell := paste0(sprintf("%.2f", rec$mean),
                     " +-.", sprintf("%03.0f", 1000*rec$sd))]

##

print(xtable(matrix(f1$cell, ncol=6, byrow = TRUE)), include.rownames = F)
print(xtable(matrix(prec$cell, ncol=9, byrow = TRUE)[, c(1:3, 7:9)]), include.rownames = F)
print(xtable(matrix(rec$cell, ncol=9, byrow = TRUE)[, c(1:3, 7:9)]), include.rownames = F)
print(xtable(matrix(f1_label$cell, ncol=9, byrow = TRUE)[, c(1:3, 7:9)]), include.rownames = F)

print(xtable(matrix(prec$cell, ncol=9, byrow = TRUE)[, 4:9]), include.rownames = F)
print(xtable(matrix(rec$cell, ncol=9, byrow = TRUE)[, 4:9]), include.rownames = F)
print(xtable(matrix(f1_label$cell, ncol=9, byrow = TRUE)[, 4:9]), include.rownames = F)



