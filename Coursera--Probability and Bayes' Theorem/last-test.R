dat <- read.table('F:/pgalpga2008.dat')
datF <- subset(dat, V3==1, select=1:2)
datM <- subset(dat, V3==2, select=1:2)
X = c(datF$V1)
Y = c(datF$V2)
plot(datM$V2~datM$V1)
plot(datF$V2~datF$V1)
lm.sol <- lm(Y~1+X) 
summary(lm.sol)
new <- data.frame(X = 260)     
lm.pred <- predict(lm.sol,new,interval = "prediction",level = 0.95)
lm.pred


X1 = c(dat$V1)
X2 = c(dat$V3)
Y = c(dat$V2)
lm.sol <- lm(Y~1+X1+X2) 
summary(lm.sol)
  