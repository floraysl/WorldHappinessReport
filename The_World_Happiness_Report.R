library(dplyr)
library(readxl)
library(dplyr)
library(glmnet) #Ridge and Lasso
library(rpart) # decision tree
library(randomForest) # random forest
library(gbm) # generalized boosted models
library(caret) # CV utilities, easily train with cross validation
library(xgboost) # xgboost
library(nnet) # Neural Network

raw_data <- read_excel("Chapter2OnlineData.xls")[, 1:14]

#===Remove Cuba, North Cyprus, Oman, Somalia, Somaliland region, Swaziland===#
data0 <- raw_data[-which(raw_data$`Country name`=='Cuba'
                         | raw_data$`Country name`=='North Cyprus'
                         | raw_data$`Country name`=='Oman'
                         | raw_data$`Country name`=='Somalia'
                         | raw_data$`Country name`=='Somaliland region'
                         | raw_data$`Country name`=='Swaziland'), ]

#===Remove white space from column names===#
names(data0) <- make.names(names(data0), unique=TRUE)

#===Split the data into train/test===#
set.seed(1)
train_ind <- sample(1:nrow(data0), 0.8*nrow(data0))
train0 <- data0[train_ind, ]
test0 <- data0[-train_ind, ]

#===Remove unneccessary columns===#
data1 <- dplyr::select(data0, -c("Year", "Perceptions.of.corruption", "Confidence.in.national.government"))

#===Create a new dataset called country_avg which records the averages through years for each country==#
country_avg <- aggregate(data1[,-1], by = list(as.factor(data1$Country.name)), mean, na.rm = TRUE)
data_avg <- dplyr::select(country_avg, -c("Group.1"))

#===Split the averages dataset into train/test==#
set.seed(1)
train_ind <- sample(1:nrow(data_avg), 0.8*nrow(data_avg))
train_avg <- data_avg[train_ind, ]
test_avg <- data_avg[-train_ind, ]

train_x <- dplyr::select(train_avg, -c("Life.Ladder"))
train_y <- train_avg$Life.Ladder
test_x <- dplyr::select(test_avg, -c("Life.Ladder"))
test_y <- test_avg$Life.Ladder

#==show the correlations==#
#correlation matrix
cor(data1[,-1], use = "complete.obs")
#visulization
library(PerformanceAnalytics)
data_corr<- data1[,-c(1,2)]
df<- data.frame(data_corr,data1[,3])
suppressWarnings(chart.Correlation(df, col = "purple", pch = "*"))


#===Linear Regression==#
lm<-lm(Life.Ladder~.,data=train_avg[,-1])
summary(lm)
pred_lm<-predict(lm,test_avg[,-1])
pred_lm
MSE_lm<-mean((test_avg$Life.Ladder-pred_lm)^2)
MSE_lm #0.1282466


#===Linear Regression with AIC===#
lm_AIC<-step(lm)
summary(lm_AIC)
pred_AIC<-predict(lm_AIC,test_avg[,-1])
MSE_lm_AIC<-mean((test_avg$Life.Ladder-pred_AIC)^2)
MSE_lm_AIC #0.1303666


#===Linear Regression with BIC===#
lm_BIC<- step(lm, direction = "both" ,k = log(nrow(train_avg)), trace = 0)
summary(lm_BIC)
pred_BIC <- predict(lm_BIC, newdata = test_avg[,-1])
MSE_lm_BIC<- mean((pred_BIC - test_avg$Life.Ladder)^2)
MSE_lm_BIC #0.1392965


#===Ridge(nfold = 5)===#
library(glmnet)
set.seed(1)
train_num<-train_avg[,-1]
lambdas <- cv.glmnet(x = as.matrix(train_num[,-1]), y = train_num[,1], nfolds = 5,alpha = 0)
lambdas$lambda.min 
lm_ridge <- glmnet(x = as.matrix(train_num[,-1]), y = train_num[,1],lambda = lambdas$lambda.min)
coef(lm_ridge)
pred_ridge <- predict(lm_ridge, newx = as.matrix(test_avg[,c(-1,-2)]))
MSE_lm_ridge<- mean((pred_ridge - test_avg$Life.Ladder)^2)
MSE_lm_ridge  # 0.2122564



#===LASSO===#
set.seed(1)
train.x=as.matrix(train_avg[,c(-1,-2)])
train.y=as.matrix(train_avg[,2])
test.x=as.matrix(test_avg[,c(-1,-2)])
test.y=as.matrix(test_avg[,2])

lasso = cv.glmnet(train.x, train.y, alpha =1, nfolds = 5)
opt_lam<-lasso$lambda.min
CF_lasso <- as.matrix(coef(lasso, lasso$lambda.1se))
CF_lasso
pred_lasso<-as.matrix(cbind(const=1,test.x)) %*% as.vector(coef(lasso, s = "lambda.min"))
MSE_lasso<-mean((pred_lasso-test_avg$Life.Ladder)^2)
MSE_lasso #0.1254919



#===Decision Tree===#
complexity.parameter <- seq(.005, .025, .001)
rpart_grid <- expand.grid(cp = complexity.parameter)
rpart_control <- trainControl(method = "cv", number = 10)
rpart_fit <- train(Life.Ladder ~ ., data = data_avg,
                   method = "rpart",
                   tuneGrid = rpart_grid,
                   trControl = rpart_control)
rpart_fit
plot(rpart_fit)
#par(mfrow = c(1,2), xpd = NA)
plot(rpart_fit$finalModel)
text(rpart_fit$finalModel, use.n = TRUE)
mse(actual = test_y, predicted = predict(rpart_fit, newdata = test_x))
#par(mfrow = c(1,1))



#===Random Forest===#
mtry <- seq(2,8)
nodesize <- seq(3,10)
rf <- list()
rf$grid <- expand.grid(mtry = mtry)
rf$control <- trainControl(method = "oob")
rf$fit <- train(Life.Ladder ~ ., data = data_avg,
                method = "rf",
                tuneGrid = rf$grid,
                trControl = rf$control,
                importance = TRUE)

rf$fit$finalModel
plot(rf$fit)
ggplot(data = rf$fit$results) +
  geom_line(aes(x = mtry, y = RMSE), color = "blue") +
  geom_point(aes(x = mtry, y = RMSE), color = "blue") +
  labs(title = "Cross Validation Errors for Random Forest",
       x = "Number of Randomly Selected Predictors (mtry)",
       y = "RMSE Out of Bag Resampling")

rf$plot_data_final <- data.frame(ntree = 1:500, Error = rf$fit$finalModel$mse)
plot(rf$fit$finalModel)

ggplot(data = rf$plot_data_final) +
  geom_line(aes(x = ntree, y = Error)) +
  labs(title = "Learning Rate of Final Random Forest Model", x = "Number of Trees", y = "Error")

rf$fit
rftmp <- randomForest(Life.Ladder ~ ., data = data_avg, importance = TRUE, mtry=5)
rftmp
rf_imp <- importance(rftmp)
attr(rf_imp)
rf_imp[2,1]
imp_data <- as.data.frame(rf_imp)
imp_data$predictor <- factor(rownames(imp_data),
                             levels = names(sort(rf_imp[,2], decreasing = TRUE)))
ggplot(data = imp_data) +
  geom_col(aes(x = predictor, y = IncNodePurity, fill = predictor)) +
  scale_colour_manual(values=cbPalette) + scale_fill_manual(values=cbPalette) +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Predictor Importance from Random Forest", x = "predictor")

ggplot(data = imp_data) +
  geom_col(aes(x = predictor, y = IncNodePurity, fill = predictor)) +
  scale_color_viridis_d() + scale_fill_viridis_d() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  labs(title = "Predictor Importance from Random Forest", x = "predictor")



plot(rftmp)
importance(rf$fit)
mse(predict(rftmp, test_x), test_y)
mse(predict(rf$fit, test_x), test_y)



#===Gradient Boosting===#
num_trees <- c(50, 100, 200, 400, 500, 1000)
shrinkage <- c(.0001, .001, .01, .05, .075, .1,  .125, .15, .2)
int.depth <- 1
n.mino <- 10

gb <- list()
gb$grid <- expand.grid(n.trees = num_trees, interaction.depth = int.depth,  
                       shrinkage = shrinkage, n.minobsinnode = n.mino)
gb$control <- trainControl(method = "cv", number = 10)
gb$fit <- train(Life.Ladder ~ ., data = data_avg,
                method = "gbm",
                tuneGrid = gb$grid,
                trControl = gb$control)
gb$fit
plot(gb$fit)

# error plot of final model
gb$plot_data_final <- data.frame(ntree = 1:length(gb$fit$finalModel$train.error))
plot(gb$fit$finalModel)
mse(predict(gb$fit, newdata = test_x), test_y)

## Neural Network ##

nfold = 5
H_matrix <- matrix(NA, 20, 5)
infold <- sample(rep(1:nfold, length.out = 127))
for (i in 1:nfold){
  count <- 0
  for (j in seq(1,20,1)){
    count <- count +1
    train_avg_temp <- train_avg[infold != i, 2:11]
    val_avg_temp <- train_avg[infold == i, 2:11]
    Happiness_nn <- nnet(Life.Ladder/max(abs(train_avg_temp$Life.Ladder))~., data = train_avg_temp, size = j*4, decay = 0.01)
    Happiness_predict <- predict(Happiness_nn, val_avg_temp[,2:10]) * max(train_avg_temp$Life.Ladder)
    
    H_matrix[count, i] <- mean((val_avg_temp$Life.Ladder - Happiness_predict)^2)
  }
}

size <- rowMeans(H_matrix)


MSPE_m <- mean((test_avg$Life.Ladder - Happiness_predict_t)^2)
MSPE <- (test_avg$Life.Ladder - Happiness_predict_t)^2

Happiness_nn <- nnet(Life.Ladder/max(train_avg_temp$Life.Ladder)~., data = train_avg_temp, size = 60, decay = 0.01)
Happiness_predict_t <- predict(Happiness_nn, test_avg[,3:11]) * max(abs(train_avg_temp$Life.Ladder))

## Neural Network Plots 
p1 <- qplot(seq(1,20,1)*4, size, main = 'Cross Validation Error', xlab = 'nNode', ylab = 'MSE')
p1 + theme(axis.title=element_text(face="bold.italic",
                                   size="25", color="black"),axis.text.x = element_text(size = '20'), plot.title = element_text(size = '25'), legend.position="top") + geom_point(size = 5,colour = I("blue"))

p2 <- qplot(test_avg$Group.1, MSPE, xlab = 'Nations', ylab = 'R^2', main = 'Squared Prediction Error for Each Nation in Test Set')
p2 + theme(axis.title = element_text(face = 'bold', size = '25', color = 'black'), axis.text.x = element_text(angle=90, hjust=1, size = '20'), plot.title = element_text(size = '25'))+ geom_point(size = 5,colour = I('darkblue'))
