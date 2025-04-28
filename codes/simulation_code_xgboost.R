############################### READ ME ########################################################################################
###  Download the datasets from 'chr22-Genotype_SNP' & 'chr22-Expressions' folders that have been shared.
### Then save those SNPm_tissuenmae datasets (inside 'chr22-Genotype_SNP') to the location similar to "snp_dr" (down below)
### & save those expr_tissuename datasets (inside 'chr22-Expressions') to the location similar to "expression_dir" (down below)
### Don't change the datasets name by mistake- that will ruin the shared codes!!!
### Once you are done with the coding, please write at the top "Project on TWAS-'MethodName'" & prepared by 'yourname' in both the heading
# & file name so that we can follow who is working on what'.
### Thank you ver much for your patience and reading these 'boring read-me'- :). Let us say us good luck!
#############################################################################################################################



# Load necessary libraries or install if needed
library(tibble)
library(dplyr)
library(xgboost)
library(caret)
library(doParallel)
# Define the directories- change as needed
expression_dir <- "~/Desktop/Stat Research Group/chr22-Expressions 2/"
snp_dir <- "~/Desktop/Stat Research Group/chr22- Genotype_SNP 2/"
# Check, no need for analysis Starts #
SNPmAdS <- readRDS('~/Desktop/Stat Research Group/chr22- Genotype_SNP 2/SNPm_adipose_subcutaneous.rds')
#print(as_tibble(SNPmAdS)) #577 × 87
exprAdS <- readRDS("~/Desktop/Stat Research Group/chr22-Expressions 2/expr_adipose_subcutaneous.rds")
#print(as_tibble(exprAdS)) # 1 × 577
# Check, no need for analysis Ends #

# List expression and SNP matched files
expression_files <- list.files(expression_dir, pattern = "\\.rds$", full.names = TRUE)
snp_files <- list.files(snp_dir, pattern = "\\.rds$", full.names = TRUE)

# Extract tissue names from file names
expression_tissues <- gsub("^expr_|\\.rds$", "", basename(expression_files))
snp_tissues <- gsub("^SNPm_|\\.rds$", "", basename(snp_files))
length(snp_tissues) # 49 tissues
#snp_tissues[26] # "esophagus_mucosa"


# Initialize an empty list to store the combined data
data_list <- list()

# Loop through each tissue to combine expression and SNP data
for (tissue in expression_tissues) {
  # Find the corresponding expression and SNP file for the tissue
  expr_file <- expression_files[which(expression_tissues == tissue)]
  snp_file <- snp_files[which(snp_tissues == tissue)]
  
  # Read the expression and SNP data
  expr_data <- readRDS(expr_file)
  snp_data <- readRDS(snp_file)
  
  # Transpose the expression data and convert to data frame
  expr_data <- as.data.frame(t(expr_data))
  colnames(expr_data) <- "y"
  
  # Combine the expression and SNP data
  combined_data <- cbind(expr_data, snp_data)
  
  # Append the combined data to the list
  data_list[[tissue]] <- combined_data
}

# Print the structure of the data_list
#str(data_list)

# Display the head of the data list to check the result
#head(data_list)
# prediction data: GWAS
gokind_final.matched <- read.table("~/Desktop/Stat Research Group/GWAS-Dataset/gokind37_final.M_rs.txt", header = TRUE, sep = "\t", quote = "")
head(gokind_final.matched)
dim(gokind_final.matched)
gwas <- gokind_final.matched[,-1]




#############################################################################################################################
# Output needed after Fitting Deep Learning model:Create a rds file, named as "saveRDS(weight_deep.methodname, file=location)
#*i is the i-th tissue name
# head(weight_deep)
# [[i]]
# rsid   weight   
############################################################################################################################

# Codes to check whether correct data format
# Number of tissues
#P <- length(weight_deep.methodname) # 49
# Get tissue-specific weights (B_hat) from results_list
#for (i in 1:P) {
#  B_hat_list[[i]] <- weight_deep.methodname[[i]]$weight
#}

#str(B_hat_list) # 1*87 vector of P tissues
#length(B_hat_list[[1]]) # 87

#head(gwas)

# Load libraries
library(xgboost)
library(caret)
ppp <- proc.time() 
#lungs <- data_list$lung_tissue
#lungs <- lungs

#train_indices <- createDataPartition(lungs[,1], p = 0.7, list = FALSE)
#train_data <- lungs[train_indices, ]
#test_data <- lungs[-train_indices, ]

# Define parameter grid
#param_grid <- expand.grid(
#  nrounds = c(100, 200, 250, 300, 350, 400),
#  max_depth = c(3, 4,5,6, 7,8,  9),
#  eta = c(0.01,0.025,0.05, 0.1, 0.2, 0.3, 0.4),
#  gamma = c(0, 0.1, 0.2, 0.3, 0.4),
#  colsample_bytree = c(0.7, 0.8, 1.0),
#  min_child_weight = c(1, 2, 3,4, 5),
#  subsample = c(0.7, 0.8, 1.0)
#)
#parameter_values <- matrix(0, nrow=49, ncol= 7)
# obtain from tuning parameters the min and max values of each parameters. Make this the range
# across all the tissues. 

#control <- trainControl(
#  method = "cv", 
#  number = 3,    # Number of folds
#  verboseIter = TRUE,
#  allowParallel = TRUE
#)

#set.seed(42)
#xgb_model <- train(
#  x = train_data[,-1] ,
#  y = train_data[,1],
#  method = "xgbTree",
#  tuneGrid = param_grid,
#  trControl = control
#)

#print(xgb_model$bestTune)
#best_model <- xgb_model$finalModel
#print(best_model)
#test_pred <- predict(xgb_model, test_data[,-1])
#rmse <- sqrt(mean((test_pred - test_data[,1])^2)); rmse
#gwas_pred <- predict(xgb_model, gwas)

############## The loop for all tissue.  ####################################
#############################################################################

cc <- detectCores()
cl <- makePSOCKcluster(cc - 1)
registerDoParallel(cl)

predicted_matrix <- matrix(0, dim(gwas)[1],  length(expression_tissues) )

param_matrix<- data.frame(nrounds = numeric(0),  max_depth= numeric(0), eta = numeric(0), gamma= numeric(0),
                          colsample_bytree= numeric(0), min_child_weight= numeric(0), subsample= numeric(0))
FID <- rownames(gokind_final.matched)
rownames(predicted_matrix) <- rownames(gokind_final.matched)#FID
colnames(predicted_matrix) <- expression_tissues
length(FID)
# Define parameter grid
param_grid <- expand.grid(
  nrounds = c(100, 200, 250, 300, 350, 400),
  max_depth = c(3, 4, 5, 6, 7, 8,  9),
  eta = c(0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.4),
  gamma = c(0, 0.1, 0.2, 0.3, 0.4),
  colsample_bytree = c(0.7, 0.8, 1.0),
  min_child_weight = c(1, 2, 3, 4, 5),
  subsample = c(0.7, 0.8, 1.0)
)
control <- trainControl(
  method = "cv", 
  number = 3,    
  verboseIter = FALSE,
  allowParallel = TRUE
)

set.seed(4284)
for (tissue in expression_tissues){
  train_data <- data_list[[tissue]]
  ptm <- proc.time()

  xgb_model <- train(
    x = train_data[,-1] ,
    y = train_data[,1],
    method = "xgbTree",
    tuneGrid = param_grid,
    trControl = control
  )
  
  best_param = xgb_model$bestTune
  rownames(best_param)<- c(tissue)
  predicted_matrix[, tissue] <- predict(xgb_model, gwas)
  param_matrix<- rbind(param_matrix, best_param)
  cat("Prediction for Tissue", tissue, "  recorded. Took",(proc.time() - ptm)/60, "minutes \n")
  print(best_param)
}
write.csv(predicted_matrix, paste0("predicted_values_gwas_xgboost_new.csv"))
stopCluster(cl)
##### summary for tuning parameters
min_params <- setNames(as.data.frame(t(apply(param_matrix, 2, min))), names(param_matrix))
max_params <- setNames(as.data.frame(t(apply(param_matrix, 2, max))), names(param_matrix))
rownames(min_params) <- "min_params"
rownames(max_params) <- "max_params"
param_matrix <- rbind(param_matrix, min_params, max_params)
write.csv(param_matrix, paste0("parameter_values_all_49_tissues_xgboost.csv"))
total_duration <- (proc.time()-ppp)[3]/(60*60)
cat("total duration is ",total_duration,"hours." ) 
