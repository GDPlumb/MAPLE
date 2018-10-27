
data = read.csv("results.csv")
data[-1] = round(data[-1], 5)
colnames(data) = c("Dataset", "MAPLE RMSE", "LIME Exp RMSE - 0.1", " APLE Exp RMSE - 0.1", "LIME Exp RMSE - 0.25", "MAPLE Exp RMSE - 0.25")
data = data[order(data$Dataset), ]

data[-1] = round(data[-1], 3)

write.csv(data, "table.csv", row.names = F)

