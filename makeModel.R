setwd("~/kag/london")
library(caret)
library(doMC)

load("trainData.RData")
##################
## Build Models ##
##################
#model parameters
registerDoMC(7)
pp <- c("center","scale")
sumFunc <- function (data, lev = NULL, model = NULL) {
  require(pROC)
  if (!all(levels(data[, "pred"]) == levels(data[, "obs"]))) 
    stop("levels of observed and predicted data do not match")
  rocObject <- try(pROC:::roc(data$obs, data[, lev[1]]), silent = TRUE)
  rocAUC <- if (class(rocObject)[1] == "try-error") 
    NA
  else rocObject$auc
  out <- c(mean(data[,"pred"]==data[,"obs"]),rocAUC, sensitivity(data[, "pred"], data[, "obs"], lev[1]), specificity(data[, "pred"], data[, "obs"], lev[2]))
  names(out) <- c("ACC","ROC", "Sens", "Spec")
  out
}
tc <- trainControl(method="repeatedcv",number=10,repeats=4,classProbs=T,savePred=T,index=createMultiFolds(train$label, k=10, times=5),summaryFunction=sumFunc)
(model <- train(label~.,data=train,method="avNNet",trControl=tc,preProcess=pp,metric="ROC",tuneGrid=expand.grid(.bag=c(F),.size=c(27:28),.decay=c(0.17)),allowParallel=F))
test.pred <- predict(model,test,type="prob")[,"one"]

##################################
## Semisupervised Self Training ##
##################################
#add cases
newtraini <- which(test.pred >= 0.98 | test.pred <= 0.02)
newtrain <- test[newtraini,]
newtrain$label <- as.factor(ifelse(test.pred[newtraini] < 0.5, "zero","one"))
train <- rbind(train,newtrain[,names(train)])

#make new model
tc$index <- lapply(tc$index,function(fold) c(fold,1001:nrow(train)))
(model.semi <- train(label~.,data=train,method="avNNet",trControl=tc,preProcess=pp,metric="ROC",tuneGrid=expand.grid(.bag=c(F),.size=c(27:28),.decay=c(0.17)),allowParallel=F))
test.pred.semi <- predict(model.semi,test,type="prob")[,"one"]

######################
## Save Predictions ##
######################
test.pred <- (test.pred+test.pred.semi)/2
write.csv(data.frame(id=1:length(test.pred),solution=ifelse(test.pred < 0.5, 0 ,1)),file="avNNet.csv",row.names=F)