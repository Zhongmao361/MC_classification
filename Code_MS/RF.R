# Method: 
# RF. 
# Tuning grids & model selection criteria can be set in 'SET PARAMETERS' section and line 164 'vec_mtry = '.

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

## Parameters for ramdom forest tuning 
#  Note: the tuning grids 'vec_ntree' and 'vec_mtry' should be set based on p (number of regressors).
#        If p is large, larger ntree is needed; mtry does not need to be large.
# Ex:
criteria = c('Accuracy', 'auc')[1]
vec_ntree = c(5000)
# p = 50 # number of regressors after marginal screening
# vec_mtry = sort(
#   unique(
#     c(floor(seq(10, floor(p/3), length.out = 6)), floor(sqrt(p)))
#   ))



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
  #   Since 'auc' is not supported by other pacakges, write a loop. 
  set.seed(seed[s])
  
  p = ncol(train) - 1 # number of regressors after marginal screening
  vec_mtry = sort(
    unique(
      c(floor(seq(10, max(floor(p/3), 10), length.out = 6)), floor(sqrt(p)))
    ))

  mat_cv = data.frame(matrix(NA, 
       nrow = length(vec_ntree)*length(vec_mtry),
       ncol = 4))
  colnames(mat_cv)=c('para', 'ntree', 'vec_mtry', 'value')
  
  cou = 0
  for(ii in 1:length(vec_ntree)){
    for(jj in 1:length(vec_mtry)){
      cou = cou + 1
      vec_value = rep(NA, K_cv)
      for(kk in 1:length(index.cv)){
        true_values <- train[index.cv[[kk]], 1] 
        RF.train <- randomForest(x = as.matrix(train[-index.cv[[kk]], -1]),
                                 y = train[-index.cv[[kk]], 1],
                                 mtry = vec_mtry[jj],
                                 ntree = vec_ntree[ii]
        )
        if(criteria == 'auc'){
          RF.prediction.roc <- data.frame(predict(RF.train, train[index.cv[[kk]], -1], type="prob"))[, 2]
          pred <- prediction(RF.prediction.roc, true_values)
          rf.perf <- performance(pred, "tpr", "fpr")
          rf.fpr <- rf.perf@x.values[[1]]
          rf.tpr <- rf.perf@y.values[[1]]
          # pROC::auc(train[index.cv[[kk]], 1], RF.prediction.roc) # is the same
          vec_value[kk] = f.auc(data.frame(fpr = rf.fpr, tpr = rf.tpr))
        }else if(criteria == 'Accuracy'){
          pred <- predict(RF.train, train[index.cv[[kk]], -1])
          vec_value[kk] = accuracy(pred, true_values)
        }
        
      }
      mat_cv[cou, ] = c(cou, vec_ntree[ii], vec_mtry[jj], mean(vec_value))
    }
  }
  sele = which.max(mat_cv$value)[1]
  params = list(ntree = mat_cv$ntree[sele], 
                mtry = mat_cv$vec_mtry[sele]
  )
  
  ## [3] Fit model on training data with selected parameters
  RF.train <- randomForest(as.matrix(train[, -1]),
                           train[, 1],
                           ntree = params$ntree,
                           mtry = params$mtry
  )
  list.params[[s]] = params
  
  ## [4] Validate on test data & record roc, auc, feature importance
  pred <- data.frame(predict(RF.train, as.matrix(test[, -1]), type="prob"))[, 2]
  true.lab = test[, 1]
  tmp = f.confusion(true.lab, pred)
  vec.test.auc[s] = tmp$val.auc     # test auc
  accu[s, ] = tmp$accuracy     # test accuracy with cutoff p=0.5 
  tab.test.roc[[s]] = tmp$data.roc # test ROC curves
  
  tmp = data.frame(RF.train$importance)
  tmp <- tmp %>%
    add_column(Feature = rownames(tmp), .before = 1)
  rownames(tmp) = 1:nrow(tmp)
  Selected.features[[s]] <- tmp # train Feature importance (mean decrease gini)
  
}


# //////// Aggregate results //////// #

## Train feature importance table
tab.feature_all <- do.call("rbind", Selected.features)
tab.feature_all_Merge <- plyr::ddply(tab.feature_all, .(Feature), summarize, 
                                     Sum.MeanDecreaseGini = round(sum(MeanDecreaseGini), 3), 
                                     Marginal.Frequency = length(Feature))
tab.train.importance <- tab.feature_all_Merge[order(-tab.feature_all_Merge$Sum.MeanDecreaseGini), ] # RF Features

## Test accuracy table with cutoff p = 0.5 
tab.test.accu <- data.frame(
  matrix(c(apply(accu, 2, function(x) mean(x, na.rm=TRUE)), sd(accu[,5], na.rm=TRUE)), nrow = 1)
)
colnames(tab.test.accu) = c("AVE.TPR", "AVE.TNR", "AVE.FNR", "AVE.FPR", "Accuracy.AVE", "Accuracy.sd")
row.names(tab.test.accu) <- c("RF")

## Test roc
tab.test.roc <- lapply(tab.test.roc, function(x) data.frame(type="RF", x))


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

file_name = paste0("../Result_MS/RF.rda")
save(res, file = file_name)









