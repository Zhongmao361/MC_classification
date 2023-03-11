##Useful Functions

# ------------------ #
#   Marginal Tests   #
# ------------------ #

## Function "f.marginal.test" conduct columnwise two sample testing (MS group and Control group).

# return a data.frame where
#    "variable": feature name, 
#    "Npvalue": shapiro normality test p value; is "NA" if feature type is not numeric
#    "Tpvalue": p value of two sample t-test / wilcoxon / fisher's exact test depending on data type & shapiro test result.

f.marginal.test <- function(meta, thre.normal) {
  
  class <- meta[, 1]
  cms <- which(meta[, 1] == "1")  # rows of group MS
  ccol <- which(meta[, 1] == "0")  # rows of group Control
  meta <- meta[, -1]  # Delete the Diagnosis col, 9726 cols remain
  
  test.meta <- data.frame(variable = names(meta),
                          Npvalue = rep(NA, ncol(meta)),
                          Tpvalue = rep(NA, ncol(meta)))
  
  for (i in 1:ncol(meta)) {
    
    if (colnames(meta)[i] == "Date.of.collection" | colnames(meta)[i] == "Sex") {
      test.meta[i,] <- c(colnames(meta)[i], NA, NA)
      
    }else if(is.factor(meta[,i])){ 
      ## when col is factor, drop col with < 2 obs in "Control" group
      test.meta[i, 2] <- NA
      test.meta[i, 3] <- if( sum(!is.na(meta[ccol,i])) < 2 ) { 
        NA 
      } else {
        tab <- table(data.frame(class = class, data = meta[, i]))
        fisher.test(tab, alternative = "two.sided")$p.value
      } 
      
    }else if(is.integer(meta[,i])){
      ## when col is integer, drop col with < 2 obs in "Control" group
      if(sum(!is.na(meta[ccol,i])) < 2 ) {
        test.meta[i, 2:3] = c(NA, NA) 
      } else {
        test.meta[i, 2] <- NA
        test.meta[i, 3] <- wilcox.test(meta[cms, i], meta[ccol, i], paired = FALSE, 
                                       correct = TRUE, exact = FALSE, alternative = "two.sided")[3]
      }
      
    }else{
      ## when col is numeric, drop col with < 2 obs in "Control" group or col whose measurements are the same 
      if( sum(!is.na(meta[ccol,i])) < 2 || 
          length(base::unique(meta[ ,i][!is.na(meta[ ,i])])) == 1) {
        test.meta[i, 2:3] = c(NA, NA)
      } else {
        test.meta[i, 2] <- shapiro.test(meta[, i])[2]
        test.meta[i, 3] <- if(test.meta[i, 2] > thre.normal) {
          t.test(meta[cms, i], meta[ccol, i], paired = FALSE, alternative = "two.sided")[3]
        } else {
          wilcox.test(meta[cms, i], meta[ccol, i], paired = FALSE, 
                      correct = FALSE, exact = FALSE, alternative = "two.sided")[3]
        }
      }
    }
  }
  
  return(test.meta)
  
}


## Function "f.selection.inte" conduct FDR control to marginal screening.
#    selection threshold differs in blood and non-blood data.

# return "p.adj": FDR adjusted p value 
#        "select_col_number": select col number in "data.full"

f.selection.inte <- function(p.marginal.table, p_blood, p_other){
  
  p.adjust <- p.adjust(p.marginal.table$Tpvalue, method = "BH")
  
  p.adjust.table <- data.frame(col_number = match(p.marginal.table$variable,colnames(dat)), 
                               feature = p.marginal.table$variable, p.adj = p.adjust)
  
  # FDR selection result: table of col number  + selected features + adjusted p value
  feature.select <- p.adjust.table %>% filter(
    col_number <= 8859 & p.adj < p_blood | 
      col_number > 8859 & p.adj < p_other)
  
  return(feature.select)
}



# -------------------------------- #
#   Some NA Filling-in Functions   #
# -------------------------------- #

## Function "f.num_na.replace" replace numerical NA with column mean

f.num_na.replace <- function(x) {
  if(is.numeric(x)){x[is.na(x)] = mean(x, na.rm =TRUE)}
  x
}


## Function "f.factor_na.replace" replace categorical NA with new group "G_NA".

f.factor_na.replace <- function(x) {
  if (is.factor(x)) {
    x <- as.character(x)
    x[is.na(x)] <- "G_NA"
    x <- factor(x)
  }
  x
}

# ------------------------------------------------------ #
# Adjust train & test data factor feature to same level
# ------------------------------------------------------ #
f.factor.level <- function(data.train, data.test){
  ntr <- 1:nrow(data.train)
  nte <- (nrow(data.train)+1):(nrow(data.train) +nrow(test))
  for (i in 2:ncol(data.train)){
    if(is.factor(data.train[, i])){
      data <- c(as.character(data.train[, i]), 
                as.character(data.test[, i]))
      data <- as.factor(data)
      data.train[, i] <- data[ntr]
      data.test[, i] <- data[nte]
    }
  }
  return(list(train = data.train, test = data.test))
}

# ------------ #
#   Accuracy   #
# ------------ #

accuracy <- function(x, y) {mean(x == y)}



## Calculate median & CI for each model
f.aucci <- function(data.roc, model){
  f.auc <- function(oneroc, model){
    x=oneroc[oneroc$type==model, ]$fpr
    y=oneroc[oneroc$type==model, ]$tpr
    # DescTools::AUC(x, y, method = "trapezoid")
    h <- x[-1] - x[-length(x)]
    sum((y[-1] + y[-length(y)]) * h) / 2
  }
  vec.auc <- as.vector(unlist(lapply(data.roc, f.auc, model= model)))
  auc.mean <- round(mean(vec.auc, na.rm=T), 2)
  auc.90ci <- round(as.vector(quantile(vec.auc, p=c(5, 95)/100, type=4)), 2)
  return(c(auc.mean, auc.90ci))
}


# 11/19/2021 cv for sparsesvm with auc
f.svm_tpr_fpr <- function(eta, SVM.yhat=SVM.yhat, true_values=true_values){
  svm.pred <- factor(1 * (SVM.yhat < eta), levels = c('0', '1'))
  levels(svm.pred) = levels(true_values)
  tmp <- caret::confusionMatrix(data=svm.pred, reference=true_values, positive = levels(true_values)[2])
  svm.tpr <- as.numeric(tmp$byClass[1])  # sensitivity
  svm.fpr <- as.numeric(1 - tmp$byClass[2])  # 1-specifity
  return(c(tpr=svm.tpr, fpr=svm.fpr))
}
f.auc <- function(oneroc){
  x=oneroc$fpr
  y=oneroc$tpr
  # DescTools::AUC(x, y, method = "trapezoid")
  h <- x[-1] - x[-length(x)]
  sum((y[-1] + y[-length(y)]) * h) / 2
}

f.sparseSVM_cv = function(X_all, y_all,   # SVM CV by AUC
                          foldid_one = foldid,
                          preprocess = "standardize", alpha_val = 0.5){
  
  fit = sparseSVM(X_all, y_all, preprocess = preprocess, alpha = alpha_val)
  vec.lambda = fit$lambda
  out_tab = data.frame(lambda = vec.lambda, auc = rep(0, length(vec.lambda)))
  
  for(i in 1:max(foldid_one)){
    
    X_train = X_all[foldid_one != i,]
    y_train = y_all[foldid_one != i]
    X_test = X_all[foldid_one == i,]
    y_test = y_all[foldid_one == i]
    fit_i = sparseSVM(X_train,
                      y_train,
                      preprocess = preprocess,
                      alpha = alpha_val
    )
    SVM.coef.mat = matrix(
      predict(fit_i, X_test, type = 'coefficients'),
      ncol = length(vec.lambda)
    )
    vec_auc = c()
    for(each in 1:ncol(SVM.coef.mat)){
      SVM.coef <- SVM.coef.mat[, each]
      SVM.yhat <-   as.matrix(X_test) %*% SVM.coef[-1] + SVM.coef[1]
      eta <- unique(sort(c(SVM.yhat, (max(SVM.yhat)+1))))
      true_values <- y_test 
      svm.perf <- data.frame(t(sapply(eta, f.svm_tpr_fpr, SVM.yhat=SVM.yhat, true_values=true_values)))
      vec_auc = c(vec_auc, f.auc(svm.perf))
    }
    out_tab$auc = out_tab$auc + vec_auc
    
    
  }
  out_tab$auc = out_tab$auc/max(foldid_one)
  lambda.select = out_tab$lambda[which.max(out_tab$auc)[1]]
  return(lambda.select)
  
}




f.confusion = function(
    true.lab, # true response label
    pred){    # predicted probabilities
  
  roc_test <- pROC::roc(true.lab, pred) # positive = levels(test[, 1])[2]
  # plot(roc_test ) 
  rf.fpr <- rev(1-roc_test$specificities)
  rf.tpr <- rev(roc_test$sensitivities)
  val.auc = f.auc(data.frame(fpr = rf.fpr, tpr = rf.tpr))
  data.roc <- data.frame(fpr=rf.fpr, tpr=rf.tpr)
  
  # report pred accuracy with cutoff p=0.5 (may remove)
  pred.lab = factor((pred>0.5)*1, levels = c(0, 1))
  levels(pred.lab) = levels(true.lab)
  tab = table(pred.lab, true.lab)
  TPR <- tab[2, 2] / (tab[1, 2] + tab[2, 2])
  TNR <- tab[1, 1] / (tab[1, 1] + tab[2, 1])
  FNR <- 1 - TPR
  FPR <- 1 - TNR
  accu = c(TPR, TNR, FNR, FPR, accuracy(pred.lab, true.lab) )
  
  return(
    list(
      val.auc = val.auc,   # auc
      data.roc = data.roc, # roc curve
      accuracy = accu          # accuracy with cutoff p=0.5
    )
  )
  
  
}


