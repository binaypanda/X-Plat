#!/usr/bin/env Rscript

# harmony_run.R
# Run HARMONY on TDM-prepared fold matrices for each organism.

suppressPackageStartupMessages({
  library(CONORData)
  library(HARMONY)   
  library(cluster)
})

args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 1) {
  stop("Usage: Rscript harmony_run.R <base_dir>", call. = FALSE)
}

base_path <- args[1]
dir.create("results/harmony", recursive = TRUE, showWarnings = FALSE)
output_base <- "results/harmony"
organisms <- c("rat", "arabidopsis", "human")
n_splits <- 10

for (org in organisms) {
  cat("Processing organism:", org, "\n")

  input_dir_arr <- file.path(base_path, paste0(org, "_tdm_input_arr"))
  input_dir_seq <- file.path(base_path, paste0(org, "_tdm_input_seq"))
  output_dir    <- file.path(output_base, paste0(org, "_harmony_output"))

  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  for (i in seq_len(n_splits)) {
    file_arr <- file.path(input_dir_arr, paste0("fold", i, "_arr.txt"))
    file_seq <- file.path(input_dir_seq, paste0("fold", i, "_seq.txt"))

    if (file.exists(file_arr) && file.exists(file_seq)) {
      cat("  Running fold", i, "\n")

      dat_arr <- read.table(file_arr, sep = "\t", header = FALSE, row.names = 1)
      dat_seq <- read.table(file_seq, sep = "\t", header = FALSE, row.names = 1)

      # Apply HARMONY: platform1 = reference distribution
      res <- harmony(
        platform1.data = as.matrix(dat_arr),
        platform2.data = as.matrix(dat_seq),
        skip.match     = FALSE,
        iterations     = 1
      )

      out_file <- file.path(output_dir, paste0("fold", i, ".txt"))
      write.table(res, file = out_file, sep = "\t", quote = FALSE)
    } else {
      cat("  Skipping fold", i, "for", org, "due to missing input files.\n")
    }
  }
}

cat("HARMONY benchmark complete.\n")