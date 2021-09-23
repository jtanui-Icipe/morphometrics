
## ML algorithms: kNN, RF, SVM, ANN ===========
#required packages

install.packages("caret")
install.packages("(kernlab")
library(caret)
library(kernlab)

fmd<-read.csv("fruitfly morphometry data.csv") #reading the data 
class(fmd$species)
levels(fmd$species)
head(fmd)

str(fmd)
#change species to factor in train data, 
fmd[["species"]] = factor(fmd[["species"]])
str(fmd)

#split data into training set and testing set createDataPartition() # Random Sampling
set.seed(508)

partition2 = createDataPartition(fmd$species, p=.7, list=F)
train.fmds = fmd[partition2, ]
test.fmds = fmd[-partition2, ]

#checking dimensions of training and testing data frame.
dim(train.fmds) 
dim(test.fmds)

summary(train.fmds$species)
summary(test.fmds$species)

trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

#################################################
#SVM 
#################################################

#)Linear Kernel
set.seed(599)
#setting model control
grid <- expand.grid(C = c(0.00, 0.25, 0.5, 0.75, 1.00,
                          1.00, 1.25, 1.5, 1.75, 2.00,
                          2.00, 2.25, 2.5, 2.75, 3.00,
                          3.00, 3.25, 3.5, 3.75, 4.00,
                          4.00, 4.25, 4.5, 4.75, 5.00,
                          5.00, 5.25, 5.5, 5.75, 6.00,
                          6.00, 6.25, 6.5, 6.75, 7.00,
                          7.00, 7.25, 7.5, 7.75, 8.00,
                          8.00, 8.25, 8.5, 8.75, 9.00,
                          9.00, 9.25, 9.5, 9.75, 10.00))
svm_linears <- train(species ~., data = train.fmds, method = "svmLinear",
                     trControl=trctrl,
                     preProcess = c("center", "scale"),
                     tuneGrid = grid,
                     tuneLength = 10)

print(svm_linears)
plot(svm_linears, main = "Linear SVM kernel: Cost verses Repeated CV Accuracy ")
svm_linears$bestTune

#testing performance of test data
prediction_svmlinears <- predict(svm_linears, newdata = test.fmds[,-1])
confusionMatrix(table(observed=test.fmds[,1], predicted=prediction_svmlinears))

# Radial and polynomial

install.packages("ROCR")
library(ROCR)
install.packages("e1071")
library(e1071)

# b) radial
set.seed(511)
#optimal paratemeters
tune <- tune.svm(species ~., data = train.fmds, 
                 gamma =seq(.01, 0.1, by = .01), 
                 cost = seq(0.01, 10.0, by = 0.25))

tune$best.parameters
tune$performances
plot(tune)
names(tune)

gamma = tune$best.parameters$gamma 
cost = tune$best.parameters$cost

### building model
svmRadials <- train(species ~., data = train.fmds, 
                    method = "svmRadial",
                    trControl=trctrl,
                    preProcess = c("center", "scale"),
                    tuneGrid = expand.grid(C = tune$best.parameters$cost,
                                           sigma= tune$best.parameters$gamma),
                    tuneLength = 10)

print(svmRadials)

#testing data Radial
prediction_svmRadials <- predict(svmRadials, newdata = test.fmds[,-1])
confusionMatrix(table(observed=test.fmds[,1], predicted=prediction_svmRadials))


# c) Polynomial

# Fit the model on the training set
set.seed(522)
C <- c(0.1,1,2)
degree <- c(1,2,3)
scale <- c(1,2,3)

gr.poly <- expand.grid(C=C,degree=degree,scale=scale)
smvpolynomials <- train(
  species ~., data = train.fmds, method = "svmPoly",
  trControl = trctrl,
  preProcess = c("center","scale"),
  #tuneLength = 10
  tuneGrid=gr.poly
)

print(smvpolynomials)
smvpolynomials
# Print the best tuning parameter sigma and C that
# maximizes model accuracy
smvpolynomials$bestTune

# Make predictions on the test data
test_pred_polynomials<- predict(smvpolynomials, newdata = test.fmds)
confusionMatrix(table(observed= test.fmds$species, predicted=test_pred_polynomials))

##################################################################
##################################################################
#Random Forest classifier
set.seed(533)
trctrlrf <- trainControl(method = "cv", number = 10)

#trctrl <- trainControl(method = "repeatedcv", number = 10, repeats = 5)

rf <- train(species ~ ., data = train.fmds, 
            method = "rf",
            trControl=trctrlrf,
            preProcess = c("center", "scale"),
            tuneLength = 10)

print(rf)
plot(rf, main ="Variation in CV error with no of randomly selected trees")
plot(rf)

#b) performance on test data
# Validation set assessment #1: looking at confusion matrix
predictionrfs <- predict(rf, test.fmds[,-1])
confusionMatrix(table(observed=test.fmds[,1], predicted=predictionrfs))

#################################
#KNN

install.packages("Knn")
library(knn)
install.packages('class')
library(class)

###
set.seed(544)
knns <- train(species ~ ., data = train.fmds, method = "knn",
              trControl=trctrl,
              preProcess = c("center", "scale"),
              tuneLength = 10)
print(knns)

plot(knns, main="Accuracy with changes in K")
plot(knns)

# Validation set assessment #1: looking at confusion matrix
prediction_knns <- predict(knns, test.fmds[,-1])
confusionMatrix(table(observed = test.fmds[,1], predicted=prediction_knns))

##############################
##ANN

library(neuralnet)
library(nnet)
require(ggplot2)

set.seed(555)

#setting model control
trctrlann <- trainControl(method = "cv", number = 10)

anns <- train(species ~ ., 
              data = train.fmds,
              method = "nnet",
              trControl= trctrlann,
              preProcess=c("scale","center"),
              tuneLength = 10)

print(anns)
plot(anns)
# Validation set assessment #1: looking at confusion matrix
predictionanns <- predict(anns, test.fmds[,-1])
confusionMatrix(table(observed = test.fmds[,1], predicted=predictionanns))
names(anns)
anns$bestTune
anns$preProcess
anns$results # gives the size, decay, and accuracy
anns$finalModel ## final model fitted






























