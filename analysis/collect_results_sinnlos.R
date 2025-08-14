library(data.table)
library(jsonlite)

source("get_scores.R")
setwd("..")

datasets = c("yelp")
models = c("roberta", "roberta-large",
           "roberta_sinnlos", "roberta-large_sinnlos",
           "roberta_a-e", "roberta-large_a-e")
archs = c("ia3", "lora", "pfeiffer")
pvps = c("qa", "prompt")
nshots = c(10L, 100L)
samples = 1:5L
runs = 1:5L

dat = rbindlist(lapply(datasets, function(dataset){
  actual_label = as.factor(fread(file.path("data", "test", dataset, "test.csv"),
                                 header = FALSE)$V2)
  rbindlist(lapply(models, function(model){
    rbindlist(lapply(archs, function(arch){
      rbindlist(lapply(pvps, function(pvp){
        rbindlist(lapply(nshots, function(nshot){
          rbindlist(lapply(samples, function(sample_i){
            path_to_pred = file.path(dataset, model, arch, pvp, nshot, sample_i)
            if (dir.exists(path_to_pred)){
              res = get_scores(actual_label, path_to_pred, majority = TRUE)
              res[, model := model]
              res[, dataset := dataset]
              res[, arch := arch]
              res[, pvp := pvp]
              res[, nshot := nshot]
              res[, sample := sample_i]
              ind_runs = rbindlist(lapply(runs, function(run_i){
                path_to_pred = file.path(path_to_pred, run_i)
                if (file.exists(file.path(path_to_pred, "predictions.csv"))){
                  res = get_scores(actual_label, path_to_pred)
                  res[, model := model]
                  res[, dataset := dataset]
                  res[, arch := arch]
                  res[, pvp := pvp]
                  res[, nshot := nshot]
                  res[, sample := sample_i]
                  res[, run := run_i]
                  res
                }
              }))
              merge(ind_runs, res)
            }
          }))
        }))
      }))
    }))
  }))
}))

fwrite(dat, file.path("analysis", "results_sinnlos.csv"))
