# Method: 
# XGBoost. 
# Tuning grids and model selection criteria can be set in 'SET PARAMETERS' section.

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

## Parameters for XGBoost tuning
criteria = c('error', 'auc')[1]

vec_eta = c(0.01, 0.05, 0.1, 0.3) # learning rate (Big values of eta result in a faster convergence and more over-fitting problems. Small values may need to many trees to converge.)
vec_max_depth = seq(2, 10, by = 2) # maximum depth of the trees
vec_subsample = c(0.5, 1) # subsample ratio of the training instance
vec_colsample_bytree = c(0.5, 1) # subsample ratio of columns when constructing each tree
min_nrounds = 10 # when tuning nrounds, select nrounds >= min_nrounds




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

for (s in 1:S){ 
  
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
  
  
  # //////// Parameter Tuning by CV//////// #
  
  ## [1] Draw sample ids for K-fold CV 
  foldid = createFolds(train[, 1],
                       k = K_cv,list = FALSE, returnTrain = FALSE)
  vec_tmp = unique(foldid)
  index.cv = list()
  for(ii in 1:K_cv){
    index.cv[[ii]] = which(foldid == ii)
  }
  
  ## [2] Tune model by CV
  set.seed(seed[s])
  
  mat_cv = data.frame(matrix(NA, 
       nrow = length(vec_eta)*length(vec_max_depth)*length(vec_subsample)*length(vec_colsample_bytree),
       ncol = 7))
  colnames(mat_cv)=c('para', 'eta', 'max_depth', 'subsample', 'colsample_bytree', 'selected_nrounds', 'value')
  
  cou = 0
  for(ii in 1:length(vec_eta)){
    for(jj in 1:length(vec_max_depth)){
      for(kk in 1:length(vec_subsample)){
        for(ll in 1:length(vec_colsample_bytree)){
          cou = cou + 1
          params = list(eta = vec_eta[ii], 
                        max_depth = vec_max_depth[jj],
                        subsample = vec_subsample[kk], 
                        colsample_bytree = vec_colsample_bytree[ll]
          )
          bstDense <- xgb.cv(data = as.matrix(train[, -1]), 
                             label = as.integer(as.character(train[, 1])), 
                             folds = index.cv,
                             objective = "binary:logistic",
                             metrics = criteria,
                             booster = 'gbtree',
                             params = params,
                             nrounds = 100,
                             verbose = 0
          )
          mat_cv[cou, 1] = cou
          mat_cv[cou, 2:5] = unlist(params)
          if(criteria == 'auc'){
            mat_cv[cou, 6] = which.max(bstDense$evaluation_log$test_auc_mean[-(1:(min_nrounds-1))])[1] + (min_nrounds-1)
            mat_cv[cou, 7] = bstDense$evaluation_log$test_auc_mean[mat_cv[cou, 6]]
          }else if(criteria == 'error'){
            mat_cv[cou, 6] = which.min(bstDense$evaluation_log$test_error_mean[-(1:(min_nrounds-1))])[1] + (min_nrounds-1)
            mat_cv[cou, 7] = bstDense$evaluation_log$test_error_mean[mat_cv[cou, 6]]
          }
          
        }
      }
    }
  }
  
  # plot(1:nrow(mat_cv), mat_cv$value)
  if(criteria == 'auc'){
    sele = which.max(mat_cv$value)[1]
  }else if(criteria == 'error'){
    sele = which.min(mat_cv$value)[1]
  }
  select_nrounds = mat_cv$selected_nrounds[sele]
  params = list(eta = mat_cv$eta[sele], 
                max_depth = mat_cv$max_depth[sele],
                subsample = mat_cv$subsample[sele], 
                colsample_bytree = mat_cv$colsample_bytree[sele]
  )
  
  ## [3] Fit model on training data with selected parameters
  XGBoost.train <- xgboost(data = as.matrix(train[, -1]), 
                           label = as.integer(as.character(train[, 1])), 
                           objective = "binary:logistic",
                           eval_metric = criteria,
                           booster = 'gbtree',
                           params = params,
                           nrounds = select_nrounds,
                           verbose = 0
  )
  params$nrounds = select_nrounds
  list.params[[s]] = params
  
  ## [4] Validate on test data & record roc, auc, feature importance
  pred = predict(XGBoost.train, as.matrix(test[, -1]))
  true.lab = test[, 1]
  tmp = f.confusion(true.lab, pred)
  vec.test.auc[s] = tmp$val.auc     # test auc
  accu[s, ] = tmp$accuracy     # test accuracy with cutoff p=0.5 
  tab.test.roc[[s]] = tmp$data.roc # test ROC curves
  Selected.features[[s]] <- xgb.importance(model = XGBoost.train) # train Feature importance
  
}


# //////// Aggregate results //////// #

## Train feature importance table
tab.feature_all <- do.call("rbind", Selected.features)[, c(1,2)]
tab.feature_all_Merge <- plyr::ddply(tab.feature_all, .(Feature), summarize, 
                                     Sum.Gain = round(sum(Gain), 3), 
                                     Marginal.Frequency = length(Feature))
tab.train.importance <- tab.feature_all_Merge[order(-tab.feature_all_Merge$Sum.Gain), ] # RF Features

## Test accuracy table with cutoff p = 0.5 
tab.test.accu <- data.frame(
  matrix(c(apply(accu, 2, function(x) mean(x, na.rm=TRUE)), sd(accu[,5], na.rm=TRUE)), nrow = 1)
)
colnames(tab.test.accu) = c("AVE.TPR", "AVE.TNR", "AVE.FNR", "AVE.FPR", "Accuracy.AVE", "Accuracy.sd")
row.names(tab.test.accu) <- c("XGBoost")

## Test roc
tab.test.roc <- lapply(tab.test.roc, function(x) data.frame(type="XGBoost", x))


## Save Results
res <- list()
res$S = S
res$vec.test.auc = vec.test.auc
res$tab.test.roc <- tab.test.roc
res$tab.test.accu <- tab.test.accu
res$tab.train.importance <- tab.train.importance
# supplemental info of tuning & samples 
res$list.params <- list.params
res$test.rows <- test.rows
res$nvar <- nvar

file_name = paste0("../Result_MS/XGBoost.rda")
save(res, file = file_name)









