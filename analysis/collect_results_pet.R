library(data.table)
library(jsonlite)

source("get_scores_pet.R")
setwd("..")

datasets = c("ag", "yelp", "yahoo")
models = c("roberta", "roberta-large")
pvps = c("qa"="p6", "prompt"="p1")
nshots = c(10L, 100L)
samples = 1:5L
runs = 1:5L

dat = rbindlist(lapply(datasets, function(dataset){
  actual_label = as.factor(fread(file.path("data", "test", dataset, "test.csv"),
                                 header = FALSE)$V2)
  rbindlist(lapply(models, function(model){
    rbindlist(lapply(names(pvps), function(pvp){
      rbindlist(lapply(nshots, function(nshot){
        rbindlist(lapply(samples, function(sample_i){
          path_to_pred = file.path(dataset, model, "pet", nshot, sample_i)
          path_pvp = ifelse(dataset == "yelp", gsub("6", "4", pvps[pvp]), pvps[pvp])
          if (dir.exists(path_to_pred)){
            res = get_scores_pet(actual_label, path_to_pred, majority = TRUE, path_pvp)
            res[, model := model]
            res[, dataset := dataset]
            res[, arch := "pet"]
            res[, pvp := pvp]
            res[, nshot := nshot]
            res[, sample := sample_i]
            ind_runs = rbindlist(lapply(runs, function(run_i){
              path_to_pred = file.path(path_to_pred, paste0(path_pvp, "-i", run_i-1L))
              if (file.exists(file.path(path_to_pred, "predictions.jsonl"))){
                res = get_scores_pet(actual_label, path_to_pred)
                res[, model := model]
                res[, dataset := dataset]
                res[, arch := "pet"]
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

fwrite(dat, file.path("analysis", "results_pet.csv"))
