rm(list=ls())
library(car)
library(tidyverse)
library(corrplot)
library(spatstat)
library(ncpen)
library(ROSE)
library(glmnet)
library(nnet)
library(rpart)
library(rpart.plot)
library(MASS)
library(randomForest)
setwd("C:\\Users\\user\\Desktop\\My\\R program\\데이터마이닝\\practice data")
source("data.mining.functions.0419.R")
source("data.mining.functions.2020.0602.R")

bank = read.csv(file.choose())[-c(1,2,3)]

#### EDA ####
summary(bank)
summary(bank[,c(1,4,5,6,7,10)])
boxplot(bank[,1]) #하위 17개 이상치 발견 383이하
boxplot(bank[,4]) #상위 458개 이상치 발견 62세 이상
boxplot(bank[,5]) 
boxplot(bank[,6])
boxplot(bank[,7]) #4일 때 이상치
boxplot(bank[,10])
colSums(is.na(bank)) #결측치 없음
cor(bank[,c(1,4,5,6,7,10)])
corrplot(cor(bank[,c(1,4,5,6,7,10)]),method='color') #상관관계 거의 존재하지 않음
sum(bank[,11]==1)/length(bank[,11]) #Exited의 비율이 약 8:2


#### dummify ####
bank[,8] = as.factor(bank[,8])
bank[,9] = as.factor(bank[,9])

dummify(bank[,2])[,-1] #France 제외
dummify(bank[,3])[,-1] #Female 제외
dummify(bank[,8])[,-1] #Nothascrcard 제외
dummify(bank[,9])[,-1] #NotActiveMember 제외

bank.df = cbind(bank[,-c(2,3,8,9)],dummify(bank[,2])[,-1],dummify(bank[,3])[,-1],dummify(bank[,8])[,-1],dummify(bank[,9])[,-1])
bank.df
names(bank.df)[10:12]=c("M","HC","AC")

#### outlier 제거 , 총 527개 제거####
sum(bank.df[,1]<=383)
order(bank.df[,1]<=383,decreasing = T)[1:17]
bank.df = bank.df[-order(bank.df[,1]<=383,decreasing = T)[1:17],]

sum(bank.df[,2]>=62)
order(bank.df[,2]>=62,decreasing = T)
bank.df = bank.df[-order(bank.df[,2]>=62,decreasing = T)[1:458],]

sum(bank.df[,5]==4)
order(bank.df[,5]==4,decreasing = T)
bank.df = bank.df[-order(bank.df[,5]==4,decreasing = T)[1:69],]

#### continuous variable scaling ####
bank.df[,1:6]=scale(bank.df[,1:6],center=T,scale=T)

#### training and test set 10:1 비율로####

xy.df = bank.df[1:9521,]
y.vec = as.vector(xy.df[,7])
x.mat = as.matrix(xy.df[,-7])
xy.df = data.frame(y.vec,x.mat)

nxy.df = bank.df[9522:10456,]
ny.vec = as.vector(nxy.df[,7])
nx.mat = as.matrix(nxy.df[,-7])
nxy.df = data.frame(ny.vec,nx.mat)

#### ready ####
m.vec = c("ridge","lasso","scad","mbridge")
m.vec = c("logit","forward",paste("err-",m.vec,sep=""),paste("dev-",m.vec,sep=""))
b.mat = matrix(NA,nrow=1+ncol(x.mat),ncol=length(m.vec))
colnames(b.mat) = m.vec
rownames(b.mat) = c("intercept",colnames(x.mat))
cv.id = cv.index.fun(y.vec,k.val=10)

#### logistic regression ####
log.fit = glm(y.vec~.,family=binomial,data=xy.df)
summary(log.fit)
b.mat[,"logit"] = coef(log.fit)

#### forward selection ####
null = glm(y.vec~1,family=binomial,data=xy.df)
forw.fit = step(null,direction="forward",scope=list(lower=null,upper=log.fit))
summary(forw.fit)
f.coef = coef(forw.fit)
coef.vec = c(f.coef[1],0,f.coef[2],f.coef[8],f.coef[7],f.coef[6],0,f.coef[4],0,f.coef[5],0,f.coef[3])
b.mat[,"forward"] = coef.vec

#### ridge ####
cv.fit = cv.glmnet(x.mat,y.vec,family="binomial",alpha=0,foldid=cv.id,type.measure = "deviance")
opt = which.min(cv.fit$cvm)
plot(cv.fit$cvm)
b.mat[,"err-ridge"] = coef(cv.fit$glmnet.fit)[,opt]
cv.fit = cv.glmnet(x.mat,y.vec,family="binomial",alpha=0,foldid=cv.id,type.measure = "class")
opt = which.min(cv.fit$cvm)
plot(cv.fit$cvm)
b.mat[,"dev-ridge"] = coef(cv.fit$glmnet.fit)[,opt]

#### lasso ####
cv.fit = cv.glmnet(x.mat,y.vec,family="binomial",alpha=1,foldid=cv.id,type.measure = "deviance")
opt = which.min(cv.fit$cvm)
plot(cv.fit$cvm)
b.mat[,"err-lasso"] = coef(cv.fit$glmnet.fit)[,opt]
cv.fit = cv.glmnet(x.mat,y.vec,family="binomial",alpha=1,foldid=cv.id,type.measure = "class")
opt = which.min(cv.fit$cvm)
plot(cv.fit$cvm)
b.mat[,"dev-lasso"] = coef(cv.fit$glmnet.fit)[,opt]

#### scad ####
cv.fit = cv.ncpen(y.vec,x.mat,family="binomial",penalty="scad",fold.id=cv.id)
opt = which.min(cv.fit$rmse)
b.mat[,"err-scad"] = coef(cv.fit$ncpen.fit)[,opt]
cv.fit = cv.ncpen(y.vec,x.mat,family="binomial",penalty="scad",fold.id=cv.id)
opt = which.min(cv.fit$like)
b.mat[,"dev-scad"] = coef(cv.fit$ncpen.fit)[,opt]

#### mbridge ####
cv.fit = cv.ncpen(y.vec,x.mat,family="binomial",penalty="mbridge",fold.id=cv.id)
opt = which.min(cv.fit$rmse)
b.mat[,"err-mbridge"] = coef(cv.fit$ncpen.fit)[,opt]
cv.fit = cv.ncpen(y.vec,x.mat,family="binomial",penalty="mbridge",fold.id=cv.id)
opt = which.min(cv.fit$like)
b.mat[,"dev-mbridge"] = coef(cv.fit$ncpen.fit)[,opt]

#### assessment ####
ass = glm.ass.fun(ny.vec,nx.mat,b.mat,mod="binomial")$ass
ass

#### decision tree ####
tree = rpart(y.vec~.,data=xy.df,method="class")
rpart.plot(tree)
pruned.tree = prune(tree,cp=0.01)
rpart.plot(pruned.tree)
pred.tree = predict(pruned.tree,nxy.df,type="class")
auc.tr = sum(pred.tree==ny.vec)/nrow(nxy.df)*100 # 80.6% 정확도 4.5% 증가

#### random forest ####
y.vec = as.factor(y.vec)
y.vec = as.character((y.vec))
xy.df = data.frame(y.vec,x.mat)
bank.rf = randomForest(y.vec~.,xy.df,mtry=11,inportance=T)
pred.rf = predict(bank.rf,newdata = nxy.df,typer="class")
auc.rf = sum(pred.rf==ny.vec)/nrow(nxy.df)*100 # 79.7% 정확도 3.5% 증가
names(bank.rf)
bank.rf$importance

#### LDA ####
library(MASS)
fit = lda(Exited~.,data=txy.df)
pred.lda = predict(fit,newdata = nxy.df)$class
auc.lda = sum(pred.lda==ny.vec)/nrow(nxy.df)

#### generalized gradiendt boosting ####
library(gbm)
set.seed(1234)
bxy.df = txy.df; nbxy.df = nxy.df
bxy.df[,1] = bxy.df[,1]=="1"
nbxy.df[,1] = nbxy.df[,1]=="1"
fit = gbm(Exited~.,data=bxy.df,distribution = "adaboost",n.trees=100,cv.fold=10)
opt = gbm.perf(fit,method="cv",plot.it=FALSE)
pred.b = predict(fit,newdata=nbxy.df[,-1],n.trees=opt)>0
auc.b = sum(pred.b==ny.vec)/nrow(nxy.df)

#### 정확도 비교 ####
auc.vec = c(ass[,10],auc.tr,auc.rf)







