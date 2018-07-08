
data = read.csv("results.csv")
data[-1] = round(data[-1], 5)
colnames(data) = c("Dataset", "SLIM RMSE", "LIME Exp RMSE - 0.1", "SLIM Exp RMSE - 0.1", "LIME Exp RMSE - 0.25", "SLIM Exp RMSE - 0.25")
data = data[order(data$Dataset), ]


write.csv(data, "table.csv", row.names = F)

