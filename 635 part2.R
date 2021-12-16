#importing and cleaning data
library(dplyr)
library(corrr)
library(ggplot2)
library(tidyr)
heart<-read.csv("heart.csv")
View(heart)
heart[!complete.cases(heart),]
#no missing values

df<- heart
View(df)
for(i in 1:14){df[,i]=as.factor((df[,i]))}
-------------------------------------------------------------------------------------------
  # Initial Visualizations of the data
  #density_value
  library(ggplot2)
library(plotly)
library(tidyr)
df %>% 
  select(age,sex, cp, trestbps, chol, fbs,restecg,thalach,exang,oldpeak,slope,ca,thal,target) %>% 
  gather(metric, value) %>% 
  ggplot(aes(value, fill = metric)) + 
  geom_density(show.legend = FALSE) + 
  facet_wrap(~ metric, scales = "free")

ggplot(df, aes(x = age)) +
  geom_density(fill = 'cyan')

df %>% 
  select(age, ethnic, income, marital, occGroup, gender,weight,height,heartRate,testA,testB,testC,testD,testE) %>% 
  gather(metric, value) %>% 
  ggplot(aes(value, fill = metric)) + 
  geom_density(show.legend = FALSE) + 
  facet_wrap(~ metric, scales = "free")

ggplot(data = df) +
  geom_bar(mapping = aes(x = trestbps,color=target))

library(ggplot2)
ggplot(df, aes(x = thalach, y = cp,color=target)) +
  geom_point()


ggplot(df, aes(x = thalach, y = target)) +
  geom_point()

ggplot(df, aes(x = thalach, y = ca,color=target)) +
  geom_point()

library(ggplot2)

ggplot(df, aes(x = restecg, y = thalach,color=target)) +
  geom_point()


#df$disease=as.factor(df$disease)

library(MASS)
library(corrr)
corelation=cor(df)
corelation

library(ggcorrplot)
ggcorrplot(corelation)

-------------------------------------------------------------------------------------------
  ------------------------------------------------------------------------------------------
  #begin naive_bayes modeling
  #training the model using naive bayes using 10 cross fold validation
  for(i in 1:14){df[,i]=as.factor((df[,i]))}
X <- subset(df, select=-target)
Y <- df$target
#Y <- as.factor(Y)
library(e1071)
naive_bayes <- naiveBayes(X,Y)
library(caret)
library(klaR)
naiveBayes_model <- train(X, Y, method = "nb", trControl = trainControl(method = "cv", number = 10))

plot(naiveBayes_model)
print(naiveBayes_model)
summary(naiveBayes_model)
confusionMatrix(naiveBayes_model)
importance <- varImp(naiveBayes_model)
plot(importance)

-------------------------------------------------------------------------------------------
  #for the prediction and other statistics details using naive bayes
  library(rsample)  # data splitting 
library(dplyr)    # data transformation
library(ggplot2)  # data visualization
library(caret)
set.seed(123)
split <- initial_split(df, prop = .8, strata = "target")
train <- training(split)
test  <- testing(split)
pred_nb <- predict(naiveBayes_model, newdata = test)
test$target <- as.factor(test$target)
confusionMatrix(pred_nb,test$target)
summary(pred_nb)
#ending naive bayes modeling
-----------------------------------------------------------------------------------------------
  ------------------------------------------------------------------------------------------
  
  #decision tree modeling
  for(i in 1:15){df[,i]=as.factor((df[,i]))}
X <- subset(df, select=-target)
Y <- df$target
#Y <- as.factor(Y)
library(e1071)
library(caret)
library(klaR)
decision_tree <- train(X, Y, method = "rpart", trControl = trainControl(method = "cv", number = 10))

decision_tree
plot(decision_tree)
print(decision_tree)
summary(decision_tree)
confusionMatrix(decision_tree)
importance_dt <- varImp(decision_tree)
plot(importance_dt) 

#for the prediction and other statistics details using decision tree
library(rsample)  # data splitting 
library(dplyr)    # data transformation
library(ggplot2)  # data visualization
library(caret)
set.seed(123)
split <- initial_split(df, prop = .8, strata = "target")
train <- training(split)
test  <- testing(split)
pred_dt <- predict(decision_tree, newdata = test)
test$target <- as.factor(test$target)
confusionMatrix(pred_dt,test$target)
summary(pred_dt)
#end of decision tree
----------------------------------------------------------------------------------------
  --------------------------------------------------------------------------------------
  #support vector machines
  for(i in 1:15){df[,i]=as.factor((df[,i]))}
X <- subset(df, select=-disease)
Y <- df$disease
#Y <- as.factor(Y)
library(e1071)
library(caret)
library(klaR)
set.seed(123)
#svm <- train(X, Y, method = "svmLinear",
#             trControl = trainControl(method = "cv", number = 10),
#             preProcess = c("center","scale"))

svm_model <- train(
  target ~., data = df, method = "svmLinear",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center","scale")
)

svm
plot(svm)
print(svm)
summary(svm)
confusionMatrix(svm)
importance_svm <- varImp(svm)
plot(importance_svm) 

#for the prediction and other statistics details using svm
library(rsample)  # data splitting 
library(dplyr)    # data transformation
library(ggplot2)  # data visualization
library(caret)
set.seed(123)
split <- initial_split(df, prop = .8, strata = "disease")
train <- training(split)
test  <- testing(split)
pred_svm <- predict(svm, newdata = test)
test$disease <- as.factor(test$disease)
confusionMatrix(pred_svm,test$disease)
summary(pred_svm)
#end of svm
----------------------------------------------------------------------------------
  --------------------------------------------------------------------------------
  #neural network modeling
  for(i in 1:15){df[,i]=as.factor((df[,i]))}
X <- subset(df, select=-target)
Y <- df$target
#Y <- as.factor(Y)
library(e1071)
library(caret)
library(klaR)
set.seed(123)
#svm <- train(X, Y, method = "svmLinear",
#             trControl = trainControl(method = "cv", number = 10),
#             preProcess = c("center","scale"))

nnet_model <- train(
  target ~., data = df, method = "nnet",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center","scale")
)

nnet_model
plot(nnet_model)
print(nnet_model)
summary(nnet_model)
confusionMatrix(nnet_model)
importance_nnet <- varImp(nnet_model)
plot(importance_nnet) 

#for the prediction and other statistics details using nnet
library(rsample)  # data splitting 
library(dplyr)    # data transformation
library(ggplot2)  # data visualization
library(caret)
set.seed(123)
split <- initial_split(df, prop = .8, strata = "target")
train <- training(split)
test  <- testing(split)

pred_nnet <- predict(nnet_model, newdata = test)

confusionMatrix(pred_nnet,test$target)
summary(pred_nnet)
#end of nnet  
---------------------------------------------------------------------------------------
  ---------------------------------------------------------------------------------------
  #knn modeling
  for(i in 1:15){df[,i]=as.factor((df[,i]))}
X <- subset(df, select=-target)
Y <- df$target
#Y <- as.factor(Y)
library(e1071)
library(caret)
library(klaR)
set.seed(123)
#knn <- train(X, Y, method = "knn",
#           trControl = trainControl(method = "cv", number = 10))

knn <- train(
  target ~., data = df, method = "knn",
  trControl = trainControl("cv", number = 10),
  preProcess = c("center","scale")
)

knn
plot(knn)
print(knn)
summary(knn)
confusionMatrix(knn)
importance_knn <- varImp(knn)
plot(importance_knn) 

#for the prediction and other statistics details using knn
library(rsample)  # data splitting 
library(dplyr)    # data transformation
library(ggplot2)  # data visualization
library(caret)
set.seed(123)
split <- initial_split(df, prop = .8, strata = "target")
train <- training(split)
test  <- testing(split)
pred_knn <- predict(knn, newdata = test)

confusionMatrix(pred_knn,test$target)
summary(pred_knn)
#end of knn
---------------------------------------------------------------------------------------
  ---------------------------------------------------------------------------------------
  #kmeans clustering
  dfs<-df
for(i in 1:14){dfs[,i]=as.numeric((dfs[,i]))}
library(caret)
library(ggplot2)
library(dplyr)
df %>% ggplot(aes(thalach, cp, color= target))+
  geom_point()

library(ggpubr)
library(factoextra)
set.seed(123)

km3 <- kmeans(dfs, 3, nstart = 25)
km4 <- kmeans(dfs, 4, nstart = 25)
km5 <- kmeans(dfs, 5, nstart = 25)
km6 <- kmeans(dfs, 6, nstart = 25)
print(km3)
print(km4)
print(km5)
print(km6)
library(factoextra)

fviz_cluster(km3, dfs, ellipse.type = "norm")
fviz_cluster(km4, dfs, ellipse.type = "norm")
fviz_cluster(km5, dfs, ellipse.type = "norm")
fviz_cluster(km6, dfs, ellipse.type = "norm")


