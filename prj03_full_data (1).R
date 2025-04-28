##########################################################
####      Project 03: Reading the Full Data &         ####
####  Training the expression dataset with XGBoost    ####
##########################################################
library(caret)
library(xgboost)
library(tibble)
library(dplyr)
library(doParallel)
library(e1071)

cc <- detectCores()
cl <- makePSOCKcluster(cc - 1)
registerDoParallel(cl)
init_time <- proc.time()[3]
batch_size <- 1000
seed = 1234567

#install.packages("devtools")
#devtools::install_github("yaowuliu/ACAT")
#library("ACAT") # Cauchy Combination Test
# This code requires the matrix of hyperparameters: params_matrix, obtained from simulation stage and can be found 
#    in the directory '~/Desktop/Stat Research Group/parameter_values_all_49_tissues_xgboost.csv'

# =============================== #
#   Reading Reference Panel Data  #
# =============================== #

# Define the directories
                
base_expr_dir <- "~/Desktop/Stat Research Group/Project 03-Assoc.Dataset/expr/All_gene"
base_snp_dir  <- "~/Desktop/Stat Research Group/Project 03-Assoc.Dataset/SNPm/All_gene"
gokind_dir    <- "~/Desktop/Stat Research Group/Project 03-Assoc.Dataset/GOKIND/All_gene"

# List the gene folders (assuming folder names are the gene names)
gene_folders <- list.files(base_expr_dir)

# Initialize an empty list to store the data for all genes
all_genes_data <- list()

# Loop over each gene folder
for (gene in gene_folders) {
  
  # Define the directory for the current gene's expression and SNP data
  expression_dir <- file.path(base_expr_dir, gene)
  snp_dir <- file.path(base_snp_dir, gene)
  
  # List expression and SNP matched files for the gene
  expression_files <- list.files(expression_dir, pattern = "\\.rds$", full.names = TRUE)
  snp_files <- list.files(snp_dir, pattern = "\\.rds$", full.names = TRUE)
  
  # Extract tissue names from file names
  expression_tissues <- gsub("^expr_|\\.rds$", "", basename(expression_files))
  snp_tissues <- gsub("^SNPm_|\\.rds$", "", basename(snp_files))
  
  # Initialize an empty list to store combined data for the current gene
  gene_data_list <- list()
  
  # Loop through each tissue to combine expression and SNP data
  for (tissue in expression_tissues) {
    
    # Check if matching SNP file exists for the tissue
    if (tissue %in% snp_tissues) {
      
      # Find the corresponding expression and SNP file for the tissue
      expr_file <- expression_files[which(expression_tissues == tissue)]
      snp_file <- snp_files[which(snp_tissues == tissue)]
      
      # Read the expression and SNP data
      expr_data <- readRDS(expr_file)
      snp_data <- readRDS(snp_file)
      
      # Ensure SNP data is a data frame
      snp_data <- as.data.frame(snp_data)
      
      # Handle single SNP case: ensure the correct SNP column name is retained from the actual SNP data
      if (ncol(snp_data) == 1) {
        colnames(snp_data) <- colnames(snp_data)  # Ensure the original column name (rsID) is retained
      }
      
      # Transpose the expression data and convert to data frame
      expr_data <- as.data.frame(t(expr_data))
      colnames(expr_data) <- "y"  # Rename expression column to 'y'
      
      # Combine the expression and SNP data
      combined_data <- cbind(expr_data, snp_data)  # SNP column names should be retained
      
      # Append the combined data to the gene's data list
      gene_data_list[[tissue]] <- combined_data
    } else {
      message(paste("No SNP data found for tissue:", tissue, "in gene:", gene))
    }
  }
  
  # Add the gene's data list to the overall list of all genes
  all_genes_data[[gene]] <- gene_data_list
}

# Print the structure of the data for all genes
#str(all_genes_data)

length(all_genes_data) #8863
#head(all_genes_data$ENSG00000000457$adipose_subcutaneous)
# Display the head of one of the gene data lists to check the result
#head(all_genes_data[[gene_folders[1]]])

# Check dimensions of combined data for a specific tissue in each gene
#for (gene in gene_folders) {
#  print(paste("Gene:", gene))
#  print(dim(all_genes_data[[gene]]$brain_accumbensganglia))
#  print(dim(all_genes_data[[gene]]$brain_cortex))
#  print(dim(all_genes_data[[gene]]$esophagus_mucosa))
#}


# [1] "Genes with constant expression across all tissues (real data): 
# ENSG00000232636, ENSG00000244491, ENSG00000251380, ENSG00000264031"

# List of genes with constant expression
constant_genes <- c("ENSG00000232636", "ENSG00000244491", "ENSG00000251380", "ENSG00000264031")

# Remove these genes from all_genes_data
all_genes_data <- all_genes_data[!names(all_genes_data) %in% constant_genes]

# Check the new length of all_genes_data after removal
length(all_genes_data)  # Should be 8859 genes

cat("All gene expression training data successfully loaded. \n")
# =============================== #
#   Reading GWAS Data             #
# =============================== #

############### Checking GOKIND Data: No need; move to the next one ####################
# Define the path to the GOKIND .rds file for a specific gene (replace 'gene_name' with the actual gene name)
#gokind_file <- "~/Desktop/Stat Research Group/Project 03-Assoc.Dataset/GOKIND/All_gene/ENSG00000000460_gokind.rds"

# Read the GOKIND data for the given gene
#gokind_data <- readRDS(gokind_file)
#head(gokind_data)
############### Checking GOKIND Data: No need; move to the next one ####################



# List all .rds files in the output directory
#gokind_rds_files <- list.files(gokind_dir, pattern = "\\.rds$", full.names = TRUE)
#head(gokind_rds_files)




####################################

#control <- trainControl(
#  method = "cv", 
#  number = 2,    # Number of folds
#  verboseIter = FALSE,
#  allowParallel = TRUE
#)
ctrl <- trainControl(method = "none")
# Let's clean the data upon which we predict gene expression.
gokind_folder_path <- "~/Desktop/Stat Research Group/Project 03-Assoc.Dataset/GOKIND/All_gene"
gokind_rds_files   <- list.files(gokind_folder_path, pattern = "\\.rds$", full.names = TRUE)
gokind_gene_list   <- lapply(gokind_rds_files, readRDS)
names(gokind_gene_list) <- tools::file_path_sans_ext(basename(gokind_rds_files))
names(gokind_gene_list) <- sub("_gokind$", "", names(gokind_gene_list))
gokind_gene_list        <- lapply(gokind_gene_list, function(df) df[, -(1:6)])
# best set of hyperparameters for each tissue loaded from file as follow.
params_matrix <- read.csv('~/Desktop/Stat Research Group/parameter_values_all_49_tissues_xgboost.csv')
rownames(params_matrix) <- params_matrix$X
params_matrix$X <- NULL
params_matrix <- params_matrix[-c(50:51),]
gene_names <- names(all_genes_data)
cat("All prediction (gokind) data successfully loaded. \n")
# =============================== #
#     Training and prediction     #
# =============================== #
set.seed(seed)
# Initializing the list of output matrices. 
batch_list <- list()
batch_counter <- 0
file_counter <- 1
problematic_gene <- c()
##
for (gene in gene_names){
  all_tissues <-  all_genes_data[[gene]]
  gokind_data <- gokind_gene_list[[gene]]
  
  predicted_matrix <- matrix(0, nrow(gokind_data),  length(expression_tissues))
  colnames(predicted_matrix) <- expression_tissues
  problem_in_gene <- FALSE 
    
  for (tissue in expression_tissues){
    tryCatch({
    best_params <- params_matrix[tissue,]
    rownames(best_params) <- c()
    train_data <- all_tissues[[tissue]]
    
    xgb_model <-  train(
      x = train_data[,-1] ,
      y = train_data[,1],
      method = "xgbTree",
      trControl = ctrl, #control,
      tuneGrid = best_params
    )
    # Now let's predict
    predicted_matrix[, tissue] <- predict(xgb_model, gokind_data)
    
    }, error = function(e) {
      cat("Problem encountered for gene:", gene, "at tissue:", tissue, "\n")
      problem_in_gene <<- TRUE  # Mark this gene as problematic
    })
    if (problem_in_gene) {
      problematic_tissue = tissue
      break
    }
  } # loop out of tissues
  if (!problem_in_gene) {
    batch_list[[gene]] <- predicted_matrix
    
    if (batch_counter %% 10 == 1) {
      cat("Training/Prediction for gene file number=", file_counter, 
          "and gene batch=", batch_counter, 
          "recorded. Current gene:", gene, "being trained.\n")
    }
    batch_counter <- batch_counter + 1
    
    if (batch_counter == batch_size) {
      saveRDS(batch_list, file = paste0("predicted_matrices_Xgboost_batch_", file_counter, ".rds"))
      
      cat("Saved batch", file_counter, "with genes:", paste(names(batch_list)[1:3], "..."), "\n")
      
      # Reset
      batch_list <- list()
      batch_counter <- 0
      file_counter <- file_counter + 1
    }
    
  } else {
    # If there was a problem, record the gene and tissue
    problematic_gene_tissue = paste0(gene, "_", problematic_tissue)
    problematic_genes <- c(problematic_genes, problematic_gene_tissue)
  }
##################
} #loop out of genes

if (batch_counter > 0) {
  saveRDS(batch_list, file = paste0("predicted_matrices_Xgboost_batch_", file_counter, ".rds"))
  cat("Saved final batch", file_counter, "with genes:", paste(names(batch_list)[1:3], "..."), "\n")
}

# Save the problematic genes
saveRDS(problematic_genes, file = "problematic_genes_list.rds")
cat("Saved list of problematic genes.\n")

stopCluster(cl)
total_time <- proc.time()[3]- init_time
cat("Total time taken =",total_time/(60*60), "hours. \n")





## new possible tune grid for minimal xgboost tuning .
#tuneGrid = expand.grid(
#  nrounds = 100,             
#  max_depth = c(3, 5, 7),
#  eta = c(0.01, 0.05, 0.1),
#  gamma = 0,                 # fixed
#  colsample_bytree = 0.7,    # fixed
#  min_child_weight = 1,      # fixed
#  subsample = 0.8            # fixed
#)

# replace file path with your own path within your computer
#file_path <- "~/Desktop/Stat Research Group/Xgboost_predicted_matrices_all_genes_real_data/"
#file_names <- paste0(file_path, "predicted_matrices_Xgboost_batch_", 1:9, ".rds")
#list_of_lists <- lapply(file_names, readRDS)

#final_list <- do.call(c, list_of_lists)

#length(final_list)   
#names(final_list)[1:5]
# rows correspond to different observations, columns corresponds to the various tissues. 
#dim(final_list$ENSG00000000457)
