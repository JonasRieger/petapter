library(data.table)
library(xtable)

res = fread("results.csv")
res[, pattern := "prompt"]
res[, verbalizer := "normal"]
res1 = res[label == "Overall" & measure == "F1" & arch == "lora" &
      pvp == "prompt" & model == "roberta-large",
    .(mean = mean(value), sd = sd(value),
      majority = mean(majority), sd_majority = sd(majority)),
    by = c("dataset", "nshot", "pattern", "verbalizer")]
res_ukraine = fread(file.path("ukraine", "results_pvp.csv"))
res2 = res_ukraine[label == "Overall" & measure == "F1" & arch == "lora" & sampling == "stratified" &
              ((pvp == "prompt" & verbalizer == "normal") | (pvp == "no" & verbalizer == "alpha")),
            .(mean = mean(value), sd = sd(value),
              majority = mean(majority), sd_majority = sd(majority)),
            by = c("dataset", "nshot", "pattern", "verbalizer")]

f1 = rbind(res1, res2)

f1[, cell1 := paste0(sprintf("%.2f", f1$mean),
                     " +-.", sprintf("%03.0f", 1000*f1$sd))]
f1[, cell2 := paste0(sprintf("%.2f", f1$majority),
                     " +-.", sprintf("%03.0f", 1000*f1$sd_majority))]
f1 = f1[c(1:6, 10:12, 7:9)]
f1 = f1[, -c("mean", "sd", "majority", "sd_majority")]

print(xtable(as.matrix(f1)), include.rownames = F)

