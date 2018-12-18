#Prepare
#rm(list=ls())
#setwd("~/Desktop/Coursera/8. Project")

#Import
training<-read.csv(file="pml-training.csv", na.strings = c("NA","#DIV/0!", ""))
testing<-read.csv(file="pml-testing.csv", na.strings = c("NA","#DIV/0!", ""))

#understand data
names(training)
table(training$classe)
dim(training)

#find data with missing values
NAvalues<-sapply(training, function(x) sum(is.na(x)))
NAvalues[NAvalues>0]
summary(NAvalues[NAvalues>0])
NAcolumns<-names(NAvalues[NAvalues>0])
length(NAcolumns)
#note: all columns that have NAs are irrelevant as number of rows with NAs are from 19216 to 19622.

#delete columns with missing values
training<-training[,!names(training) %in% NAcolumns]

#delete columns 1:7 which are not required
training<-training[,-c(1:7)]

#cross validation preparation
set.seed(123)
library(caret)
intrain<-createDataPartition(y=training$classe,p=.75,list=FALSE)
trainingA<-training[intrain,]
trainingB<-training[-intrain,]
dim(trainingA)
dim(trainingB)

#Note: Next few steps are taken from the following site where Len Greski explains how to improve system performance when making rf model.
#https://github.com/lgreski/datasciencectacontent/blob/master/markdown/pml-randomForestPerformance.md

#configure parrallel processing() 
library(parallel)
install.packages("doParallel")
library(doParallel)
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)

#Configure trainControl object
#The most critical arguments for the trainControl function are the resampling method method, the number that specifies the quantity of folds for k-fold cross-validation, and allowParallel which tells caret to use the cluster that we've registered in the previous step.
fitControl <- trainControl(method = "cv", number = 5,allowParallel = TRUE)

#Develop training model rf
#Next, we use caret::train() to train the model, using the trainControl() object that we just created.
x<-trainingA[,-53]
y<-trainingA[,"classe"]

#Develop models
modRF <- train(x, y, method="rf",data=trainingA,trControl = fitControl)
modGBM<-train(x,y, data=trainingA, method="gbm", verbose=FALSE, trControl=fitControl)
modLDA<-train(x,y, data=trainingA,method="lda", verbose=FALSE, trControl=fitControl)

#De-register parallel processing cluster
stopCluster(cluster)
registerDoSEQ()


#Do prediction off trainingB data using 2 methods
predRF<-predict(modRF,trainingB)
predLDA<-predict(modLDA,trainingB)


#Create confusion matrix
cfmRF <- confusionMatrix(trainingB$classe, predRF)
cfmLDA <- confusionMatrix(trainingB$classe, predLDA)

#Check the accuracy
cfmRF$overall["Accuracy"]
cfmLDA$overall["Accuracy"]
#Conclusion: We will select RF

#apply model to test dataset to attempt quiz
predRF<-predict(modRF,testing)

