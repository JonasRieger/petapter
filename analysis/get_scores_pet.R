require(data.table)
require(stringr)

get_scores_pet = function(actual_label, path_to_pred, majority = FALSE, path_pvp){
  if (majority){
    preds = sapply(grep(path_pvp, list.dirs(path_to_pred), value = TRUE),
                   function(x) str_extract(readLines(file.path(x, "predictions.jsonl")),
                                           "\"label\": \"([0-9])\"", group = 1))
    pred_label = factor(apply(preds, 1, function(x) names(which.max(table(x)))),
                        levels = levels(actual_label))
  }else{
    pred_label = factor(str_extract(readLines(file.path(path_to_pred, "predictions.jsonl")),
                                    "\"label\": \"([0-9])\"", group = 1),
                        levels = levels(actual_label))
  }
  
  conf = table(actual_label, pred_label)
  res = sapply(colnames(conf), function(j){
    rec = conf[j,j] / sum(conf[j,])
    rec[is.na(rec)] = 0
    pre = conf[j,j] / sum(conf[,j])
    pre[is.na(pre)] = 0
    f1 = 2*pre*rec/(pre+rec)
    f1[is.na(f1)] = 0
    c(f1 = f1, precision = pre, recall = rec)
  })
  out = rbind(
    data.table(measure = c("Accuracy", rep("F1", length(colnames(res))+1)),
               label = c("Overall", "Overall", colnames(res)),
               value = c(mean(actual_label == pred_label), mean(res["f1",]), res["f1",])),
    data.table(measure = rep(c("Precision", "Recall"), each = length(colnames(res))),
               label = c(colnames(res), colnames(res)),
               value = c(res["precision",], res["recall", ])))
  if (majority){
    out[, majority := value]
    out[, value := NULL]
  }else{
    out[, time_train := NA_real_]
    out[, time_test := NA_real_]
  }
  out
}
