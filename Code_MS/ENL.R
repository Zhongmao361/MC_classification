# Method:
# ENL with elastic net penalty. 
# Model selection criteria can be set in 'SET PARAMETERS' section.

# Output

# S: number of repeats
# vec.test.auc: auc on test data for each repeat
# tab.test.roc: roc curve for each repeat
# tab.test.accu: accuracy table with cutoff p=0.5
# tab.train.importance: feature importance table
# test.rows: sample id in test data for each repeat

library(pander)
library(caret)
library(plyr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(MASS)
library(mvtnorm)
library(caret)
library(glmnet)
library(e1071)
library(randomForest)
library(stringr)
library(tibble)
library(Amelia)
library(coefplot)
library(data.table)
library(sparseSVM)
library(AppliedPredictiveModeling)
library(DT)
library(vegan)
library(xgboost)
library(magrittr)
library(Matrix)
library(ROCR)
library(gridExtra)
library(VennDiagram)

source('Supplement_functions.R')


# ////////////// SET PARAMETERS ////////////// #

seed_global = 123
S = 100            # number of repeats
Training.div = c('p_train', 'boot')[1] # dividing training & validation data based on [1] given proportion or [2] bootstrap
p_train = 0.8 # proportion of training if Training.div == 'p_train'
K_cv = 42 # K_cv fold CV for parameter tuning
p_blood = 0.05; p_other = 0.1  # threshold for marginal screening

## Parameters for ENL tuning 
criteria = c('class', 'auc')[1] # tuning by classification error; auc


# ////////////// Load Data ////////////// #
data.full = get(load("../Data_MS/Example/Data_multiomics.rda"))
dat = data.full[, -1]
dim(dat) # 52 9727
levels(dat$Diagnosis) = c('0', '1')

# ////////////// Repeat training & testing for S replicates ////////////// #

# Store results
accu = data.frame(matrix(NA, S, 5))
colnames(accu) = c('TPR', 'TNR', 'FNR', 'FPR', 'Accuracy')
Selected.features <- list()
test.rows <- list()
tab.test.roc <- list() # for roc plot
list.params <- list()
vec.test.auc = rep(NA, S)
nvar = matrix(NA, S, 1)

n = nrow(dat)
set.seed(seed_global)
seed <- sample.int(1000, S) 

for (s in 1:S){ # iter of repeats
  
  cat(paste0('s = ', s, ';\n'))
  set.seed(seed[s])
  
  
  ## Sampling train & test data
  if(Training.div == 'p_train'){
    train_index <- as.vector(createDataPartition(dat[, 1],
                                                 p = p_train,
                                                 list = FALSE,
                                                 times = 1))
    test_index = (1:n)[-train_index] 
  }else if(Training.div == 'boot'){
    train_index <- sort(sample.int(n = n, size = n, replace = TRUE))
    test_index = (1:n)[unique(train_index)] 
  }
  test.rows[[s]] <- test_index
  
  train <- dat[train_index, ]
  test  <- dat[test_index, ] 
  
  
  ## Marginal screening
  # Caution: should modify the 'f.marginal.test' and 'f.selection.inte' before usage. They are specific to this project.
  p.marginal.table <- f.marginal.test(train, thre.normal = 0.05)
  col.sele = f.selection.inte(p.marginal.table, p_blood, p_other)
  select_col = col.sele$col_number
  # Always keep "Sex" in models
  tmp <- match("Sex",names(train))
  tmp <- if (tmp %in% select_col) {c(1, select_col)} else {sort(c(1, select_col, tmp))}
  train <- train[, tmp]
  test <- test[, tmp]
  
  
  ## Fill-in NAs
  # replace numerical NAs with group mean in train; mean in test
  train <- train %>%  group_by(Diagnosis) %>%
    mutate_at(vars(-Diagnosis), f.num_na.replace) %>% ungroup()
  train <- train %>% mutate_all(f.factor_na.replace)
  # replace categorical NAs with G_NA in train & test
  test <- test %>% mutate_at(vars(-Diagnosis), f.num_na.replace)
  test <- test %>% mutate_all(f.factor_na.replace)
  train=do.call(data.frame, train)
  test=do.call(data.frame, test)
  
  
  ## Dummy
  n.train <- nrow(train)
  tmp <- data.frame(rbind(train, test))
  nvar[s] <- ncol(tmp) - 1
  t <- model.matrix(Diagnosis ~., data = tmp)[, -1]
  t <- data.frame(tmp[, 1], t)
  colnames(t)[1] = colnames(train)[1]
  train <- t[1:n.train, ]
  test <- t[-(1:n.train), ]
  
  
  # ## Generate weights for training data by relative frequency of response
  # weights <- rep(NA, nrow(train))
  # loc.lev1 = (train[, 1] == levels(train[, 1])[1])
  # wt.lev1 <- 1 - mean(loc.lev1)
  # wt.lev2 <- mean(loc.lev1)
  # weights[loc.lev1] <- wt.lev1
  # weights[!loc.lev1] <- wt.lev2
  
  
  # //////// Parameter Tuning by CV//////// #
  
  ## [1] Draw sample ids for K-fold CV 
  foldid = createFolds(train[, 1],
                       k = K_cv,list = FALSE, returnTrain = FALSE)
  vec_tmp = unique(foldid)
  index.cv = list()
  for(ii in 1:K_cv){
    index.cv[[ii]] = which(foldid == ii)
  }
  
  ## [2] Tune model by CV & Fit model on training data with selected parameters
  set.seed(seed[s])

  ENL.train <- cv.glmnet(as.matrix(train[, -1]), train[, 1], 
                         # nfolds = nrow(train), 
                         foldid = foldid,
                         family = "binomial", alpha = 0.5, 
                         # weights = weights, # add weigths to training samples 
                         type.measure = criteria # c("default", "mse", "deviance", "class", "auc", "mae", "C")
                         ) 
  # plot(ENL.train)
  
  ## [3] Validate on test data & record roc, auc, feature importance
  pred <- as.numeric(predict(ENL.train, as.matrix(test[, -1]), type = "response"))
  true.lab = test[, 1]
  tmp = f.confusion(true.lab, pred)
  vec.test.auc[s] = tmp$val.auc     # test auc
  accu[s, ] = tmp$accuracy     # test accuracy with cutoff p=0.5 (may remove)
  tab.test.roc[[s]] = tmp$data.roc # test ROC curves
  
  ## Features from train
  ENL_importance <- as.character(extract.coef(ENL.train, lambda = ENL.train$lambda.min)$Coefficient[-1])
  ENL_importance <- data.frame(Feature=ENL_importance, Freq = 1) 
  colnames(ENL_importance) <- c("Feature", "ENL.Frequency")
  Selected.features[[s]] = ENL_importance

}


# //////// Aggregate results //////// #

## Train feature importance table
tab.feature_all <- do.call("rbind", Selected.features)
tab.feature_all_Merge <- plyr::ddply(tab.feature_all, .(Feature), summarize, 
                                     Marginal.Frequency = length(Feature))
tab.train.importance <- tab.feature_all_Merge[order(-tab.feature_all_Merge$Marginal.Frequency), ] # RF Features

## Test accuracy table with cutoff p = 0.5 
tab.test.accu <- data.frame(
  matrix(c(apply(accu, 2, function(x) mean(x, na.rm=TRUE)), sd(accu[,5], na.rm=TRUE)), nrow = 1)
)
colnames(tab.test.accu) = c("AVE.TPR", "AVE.TNR", "AVE.FNR", "AVE.FPR", "Accuracy.AVE", "Accuracy.sd")
row.names(tab.test.accu) <- c("ENL")

## Test roc
tab.test.roc <- lapply(tab.test.roc, function(x) data.frame(type="ENL", x))


## Save Results
res <- list()
res$S = S
res$vec.test.auc = vec.test.auc
res$tab.test.roc <- tab.test.roc
res$tab.test.accu <- tab.test.accu
res$tab.train.importance <- tab.train.importance
# supplemental info of tuning & samples
# res$list.params <- list.params
res$test.rows <- test.rows

file_name = paste0("../Result_MS/ENL.rda")
save(res, file = file_name)







