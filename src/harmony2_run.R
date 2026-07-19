library(matrixStats)

Shambhala2 <- function( InputFileName, PFileName, QFileName, delete_buffer_files = TRUE, k = 5 ) {

    IFN = InputFileName
    MAS = read.table(IFN, header = FALSE, sep = "\t")
    colnames(MAS)[1] = "SYMBOL"
    MAS0 = as.matrix(MAS[,-1])
    NS = ncol(MAS0)
    SYMBOL = as.vector(MAS[,1])
    CN = colnames(MAS)

    PFN = PFileName 
    P = read.table(PFN, header = FALSE, sep = "\t")
    colnames(P)[1] = "SYMBOL"
    pool = merge(MAS,P,by = "SYMBOL")

    NS1 = ncol(pool)

    NG1 = nrow(pool)

    for ( j in (NS+2):NS1 ) {
        pool[,j] = as.numeric(pool[,j])
    }     
 
    if (file.exists("P_prim.txt")) file.remove("P_prim.txt")
    if (file.exists("args.txt")) file.remove("args.txt")
  
    P1FN = "P_prim.txt"
    write.table(pool, P1FN, row.names = FALSE, col.names = TRUE, sep = "\t")

    NH = ncol(MAS) - 1
    NP = ncol(P) - 1

    args = c(NH,NP,k)

    AFN = "args.txt"
    write.table(args, AFN, row.names = FALSE, col.names = FALSE)

    cmd <- "matlab -batch \"run('Shambhala2.m')\""
    system(cmd, wait = TRUE, intern = TRUE)
    QFN = QFileName
    Q = read.table(QFN, header = FALSE, sep = "\t")

    timeout <- 300
    t0 <- Sys.time()

    repeat {

      if (file.exists("Cu_bis.txt")) {

        # try reading safely
        ok <- try(readLines("Cu_bis.txt", n = 1), silent = TRUE)

        if (!inherits(ok, "try-error")) {
          Sys.sleep(2)  # extra safety buffer
          break
        }
      }

      if (as.numeric(difftime(Sys.time(), t0, units = "secs")) > timeout) {
        stop("Timeout: Cu_bis.txt not ready")
      }

      Sys.sleep(1)
    }

    Cu2FN = "Cu_bis.txt"
    MAS = read.table(Cu2FN, header = FALSE, sep = " ")

    MAS = as.matrix(MAS)

    MAS1 = merge(MAS,Q, by = 1)

    NS1 = ncol(MAS1)

    for ( j in 2:NS1 ) {
        MAS1[,j] = as.numeric(MAS1[,j])
    } 

    MAS3 = MAS1[,(NS+2):ncol(MAS1)]

    cat("Minimum MAS3:", min(as.matrix(MAS3), na.rm = TRUE), "\n")
    cat("Maximum MAS3:", max(as.matrix(MAS3), na.rm = TRUE), "\n")
    cat("Negative values:", sum(as.matrix(MAS3) < 0, na.rm = TRUE), "\n")
    cat("Zero values:", sum(as.matrix(MAS3) == 0, na.rm = TRUE), "\n")

    RM = rowMeans(log(as.matrix(MAS3)))
    RS = rowSds(log(as.matrix(MAS3))) 

    MAS2 = MAS1[,2:(NS+1)]

    MAS22 = log(MAS2+1)

    NR = nrow(MAS22)
 
    for ( nr in 1:NR ) {
        MAS22[nr,] = RM[nr] + RS[nr]*MAS22[nr,]
    }

    MAS23 = exp(MAS22)

    SYMBOL = as.vector(MAS1[,1])

    MAS33 = cbind(SYMBOL,MAS23)

    for ( j in 2:(NS+1) ) {
        MAS33[,j] = as.vector(as.numeric(MAS33[,j]))
    } 

    if (delete_buffer_files) {

        if (file.exists(P1FN)) file.remove(P1FN)
        if (file.exists(AFN)) file.remove(AFN)

        # ONLY delete Cu_bis.txt if MATLAB is confirmed finished
        if (file.exists(Cu2FN)) {

            # extra safety check: file is not locked
            can_delete <- try(file.remove(Cu2FN), silent = TRUE)

            if (inherits(can_delete, "try-error")) {
                warning("Cu_bis.txt is locked — skipping deletion")
            }
        }
    }    
    colnames(MAS33) = CN
    
    return(MAS33)
    
}    

organisms <- c("arabidopsis", "rat", "human")

folds <- 1:10
k_val <- 5

ref_base <- "../../results/harmony2_train_inputs"
tar_base <- "../../results/tdm_inputs"

out_base <- "../../results/harmony2/shambhala2_outputs"
dir.create(out_base, recursive = TRUE, showWarnings = FALSE)

run_one <- function(org, fold, direction) {

  # ---------------------------
  # FILES (as you specified)
  # ---------------------------
  seq_file1 <- file.path(ref_base,
                         paste0(org, "_harmony2_train_input_seq"),
                         paste0("fold", fold, "_seq.txt"))

  arr_file1 <- file.path(ref_base,
                         paste0(org, "_harmony2_train_input_arr"),
                         paste0("fold", fold, "_arr.txt"))

  arr_file2 <- file.path(tar_base,
                         paste0(org, "_tdm_input_arr"),
                         paste0("fold", fold, "_arr.txt"))

  seq_file2 <- file.path(tar_base,
                         paste0(org, "_tdm_input_seq"),
                         paste0("fold", fold, "_seq.txt"))

  # ---------------------------
  # SELECT TEST/TRAIN CORRECTLY
  # ---------------------------
  if (direction == "seq_to_arr") {

    test_file  <- seq_file2   # target seq
    train_file <- arr_file1   # reference arr
    out_label  <- "arr"

  } else if (direction == "arr_to_seq") {

    test_file  <- arr_file2   # target arr
    train_file <- seq_file1   # reference seq
    out_label  <- "seq"

  } else {
    stop("invalid direction")
  }

  # ---------------------------
  # LOAD
  # ---------------------------
  test <- read.delim(test_file, header = FALSE)
  train <- read.delim(train_file, header = FALSE)

  test_mat <- as.matrix(test[,-1])
  train_mat <- as.matrix(train[,-1])

  # ---------------------------
  # INVERT log1p (your pipeline)
  # ---------------------------
  test_lin <- exp(test_mat) - 1
  train_lin <- exp(train_mat) - 1

  test_lin[test_lin < 0] <- 0
  train_lin[train_lin < 0] <- 0

  eps <- 1e-8
  test_lin <- test_lin + eps
  train_lin <- train_lin + eps

  # ---------------------------
  # REBUILD
  # ---------------------------
  test_new <- data.frame(test[,1], test_lin, check.names = FALSE)
  train_new <- data.frame(train[,1], train_lin, check.names = FALSE)

cat("test_new dims:", dim(test_new), "\n")
cat("train_new dims:", dim(train_new), "\n")

  # ---------------------------
  # WRITE TEMP FILES
  # ---------------------------
  tmp_dir <- file.path(out_base, "tmp_inputs")
dir.create(tmp_dir, showWarnings = FALSE, recursive = TRUE)

test_file_tmp <- file.path(tmp_dir,
                           paste0("test_", org, "_", fold, ".txt"))

train_file_tmp <- file.path(tmp_dir,
                            paste0("train_", org, "_", fold, ".txt"))

  write.table(test_new, test_file_tmp,
              sep = "\t", row.names = FALSE,
              col.names = FALSE, quote = FALSE)
tmp <- read.delim(test_file_tmp, header=FALSE)
cat("written file dims:", dim(tmp), "\n")
head(test_file_tmp)

  write.table(train_new, train_file_tmp,
              sep = "\t", row.names = FALSE,
              col.names = FALSE, quote = FALSE)

  # ---------------------------
  # SHAMBHALA2
  # ---------------------------
  result <- Shambhala2(
    test_file_tmp,
    train_file_tmp,
    train_file_tmp,
    delete_buffer_files = FALSE,
    k = k_val
  )
  # ---------------------------
  # LOG OUTPUT
  # ---------------------------
  result_log <- result
  result_log[,-1] <- log1p(as.matrix(result[,-1]))

  # ---------------------------
  # OUTPUT DIR
  # ---------------------------
  outdir <- file.path(out_base, org, direction)

  if (!dir.exists(outdir)) {
    dir.create(outdir, recursive = TRUE)
  }

  write.table(
    result_log,
    file.path(outdir, paste0("fold", fold, "_", out_label, ".txt")),
    sep = "\t",
    row.names = FALSE,
    col.names = FALSE,
    quote = FALSE
  )

  cat("done:", org, direction, "fold", fold, "\n")

}
for (org in organisms) {
  for (f in folds) {

    run_one(org, f, "seq_to_arr")
    run_one(org, f, "arr_to_seq")
  }
}
