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
source("data.mining.functions.2020.0602.R")

bank = read.csv(file.choose())[-c(1,2,3)]

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

#### performance test by randomization ####
bank.df = bank.df[,c(7,1,2,3,4,5,6,8,9,10,11,12)]
y.vec = as.vector(dummify(bank.df[,1])[,1])
x.mat = as.matrix(bank.df[,-1])

#### ready ####
m.vec = c("ridge","lasso","scad","mbridge")
m.vec = c("logit","forward",paste("err-",m.vec,sep=""),paste("dev-",m.vec,sep=""))
b.mat = matrix(NA,nrow=1+ncol(x.mat),ncol=length(m.vec))
colnames(b.mat) = m.vec
rownames(b.mat) = c("intercept",colnames(x.mat))

#### 30 random test accuracy with cross validation ####
s.num = 30
r.mat = rand.index.fun(y.vec,s.num=s.num)
mod = c("logit","forward","err-ridge","dev-ridge","err-lasso","dev-lasso","err-scad","dev-scad","err-mbridge","dev-mbridge","tree","R.F","LDA","boosting")
e.mat = matrix(NA,s.num,length(mod))
colnames(e.mat) = mod

for(s.id in 1:s.num){
  print(s.id)
  # partition
  set = r.mat[,s.id]
  txy.df = bank.df[set,]
  nxy.df = bank.df[!set,]
  tx.mat = x.mat[set,]
  nx.mat = x.mat[!set,]
  ty.vec = y.vec[set]
  ny.vec = y.vec[!set]
  
  #### logistic regression ####
  log.fit = glm(Exited~.,family=binomial,data=txy.df)
  b.mat[,"logit"] = coef(log.fit)
  
  #### forward selection ####
  null = glm(Exited~1,family=binomial,data=txy.df)
  forw.fit = step(null,direction="forward",scope=list(lower=null,upper=log.fit))
  f.coef = coef(forw.fit)
  coef.vec = c(f.coef[1],0,f.coef[2],f.coef[8],f.coef[7],f.coef[6],0,f.coef[4],0,f.coef[5],0,f.coef[3])
  b.mat[,"forward"] = coef.vec
  
  #### ridge ####
  cv.fit = cv.glmnet(tx.mat,ty.vec,family="binomial",alpha=0,type.measure = "deviance")
  opt = which.min(cv.fit$cvm)
  b.mat[,"err-ridge"] = coef(cv.fit$glmnet.fit)[,opt]
  cv.fit = cv.glmnet(tx.mat,ty.vec,family="binomial",alpha=0,type.measure = "class")
  opt = which.min(cv.fit$cvm)
  b.mat[,"dev-ridge"] = coef(cv.fit$glmnet.fit)[,opt]
  
  #### lasso ####
  cv.fit = cv.glmnet(tx.mat,ty.vec,family="binomial",alpha=1,type.measure = "deviance")
  opt = which.min(cv.fit$cvm)
  b.mat[,"err-lasso"] = coef(cv.fit$glmnet.fit)[,opt]
  cv.fit = cv.glmnet(tx.mat,ty.vec,family="binomial",alpha=1,type.measure = "class")
  opt = which.min(cv.fit$cvm)
  b.mat[,"dev-lasso"] = coef(cv.fit$glmnet.fit)[,opt]
  
  #### scad ####
  set.seed(1234)
  cv.fit = cv.ncpen(ty.vec,tx.mat,family="binomial",penalty="scad")
  opt = which.min(cv.fit$rmse)
  b.mat[,"err-scad"] = coef(cv.fit$ncpen.fit)[,opt]
  cv.fit = cv.ncpen(ty.vec,tx.mat,family="binomial",penalty="scad")
  opt = which.min(cv.fit$like)
  b.mat[,"dev-scad"] = coef(cv.fit$ncpen.fit)[,opt]
  
  #### mbridge ####
  cv.fit = cv.ncpen(ty.vec,tx.mat,family="binomial",penalty="mbridge")
  opt = which.min(cv.fit$rmse)
  b.mat[,"err-mbridge"] = coef(cv.fit$ncpen.fit)[,opt]
  cv.fit = cv.ncpen(ty.vec,tx.mat,family="binomial",penalty="mbridge")
  opt = which.min(cv.fit$like)
  b.mat[,"dev-mbridge"] = coef(cv.fit$ncpen.fit)[,opt]
  
  #### assessment ####
  ass = glm.ass.fun(ny.vec,nx.mat,b.mat,mod="binomial")$ass
  
  #### decision tree ####
  tree = rpart(Exited~.,data=txy.df,method="class")
  pruned.tree = prune(tree,cp=0.01)
  pred.tree = predict(pruned.tree,nxy.df,type="class")
  auc.tr = sum(pred.tree==ny.vec)/nrow(nxy.df) 
  
  #### random forest ####
  set.seed(1234)
  cv.fit = rfcv(trainx = tx.mat,trainy=ty.vec,cv.fold=5)
  opt = cv.fit$n.var[which.min(cv.fit$error.cv)]
  rfy.vec = as.factor(ty.vec)
  rfy.vec = as.character((ty.vec))
  rfxy.df = data.frame(rfy.vec,tx.mat)
  bank.rf = randomForest(rfy.vec~.,rfxy.df,mtry=opt,inportance=T)
  pred.rf = predict(bank.rf,newdata = nxy.df,typer="class")
  auc.rf = sum(pred.rf==ny.vec)/nrow(nxy.df)
  
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
  e.mat[s.id,] = c(ass[,10],auc.tr,auc.rf,auc.lda,auc.b)
}
e.mat
boxplot(e.mat)
colMeans(e.mat)



