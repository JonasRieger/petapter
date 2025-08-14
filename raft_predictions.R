outdir = "C:\\Users\\riege\\OneDrive\\Dokumente\\Git\\raft\\data"

arch = "lora"
runs = 1:5

tasks = gsub(".yml", "", list.files("raft", ".yml"), fixed = TRUE)
banking_77_labels = readLines(file.path("raft", "data", "banking_77", "labels.txt"))
task = "banking_77"

for(task in tasks){
  if (task == "banking_77"){
    lik_list = lapply(runs, function(run){
      tmp = read.csv(file.path("raft", task, arch, run, "likelihoods.csv"))
      tmp = as.data.frame(matrix(tmp$Yes - tmp$No, ncol = 77, byrow = TRUE))
      colnames(tmp) = banking_77_labels
      tmp
    })
  }else{
    lik_list = lapply(runs, function(run)
      read.csv(file.path("raft", task, arch, run, "likelihoods.csv"))
    )
  }
  lik = Reduce("+", lik_list)
  ind = apply(lik, 1, which.max)
  res = data.frame(ID = seq_along(ind)+49,
                   Label = colnames(lik)[ind])
  res$Label = gsub(".", " ", res$Label, fixed = TRUE)
  if (task == "neurips_impact_statement_risks")
    res$Label = gsub("doesn t", "doesn't", res$Label, fixed = TRUE)
  write.csv(res, file.path(outdir, task, "predictions.csv"),
            quote = FALSE, row.names = FALSE)
}

if(FALSE){ # dumb prediction for banking_77
  res = read.csv(file.path(outdir, task, "predictions.csv"))
  res$Label = "pin_blocked"
  write.csv(res, file.path(outdir, task, "predictions.csv"),
            quote = FALSE, row.names = FALSE)
}