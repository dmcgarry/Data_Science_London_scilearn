setwd("~/kag/london")
library(caret)
library(doMC)

#load data
train <- read.csv(file="train.csv",header=F)
names(train)[-1] <- paste0("V",1:40)
test <- read.csv("test.csv",header=F)

#########
## PCA ##
#########
pca <- prcomp(rbind(train[,-1],test))
train <- as.data.frame(cbind(train[,1],pca$x[1:1000,]))
train[,1] <- as.factor(ifelse(train[,1] == 0, "zero", "one"))
names(train)[1] <- "label"
test <- as.data.frame(pca$x[1001:10000,])

#######################
## Feature Selection ##
#######################
registerDoMC(10)
rfeFuncs <- rfFuncs
rfeFuncs$summary <- twoClassSummary
rfe.control <- rfeControl(rfeFuncs, method = "repeatedcv", number=10 ,repeats = 4, verbose = FALSE, returnResamp = "final")
rfe.rf <- rfe(train[,-1], train[,1], sizes = 10:15, rfeControl = rfe.control,metric="ROC")

###############
## Save Data ##
###############
train <- train[,c("label",predictors(rfe.rf))]
test <- test[,predictors(rfe.rf)]
save(train,test,file="trainData.RData")