
data = read.csv("results.csv")
colnames(data) = c("Dataset", "SVR", "LIME - 0.1", "SLIM - 0.1", "LIME - 0.2", "SLIM - 0.2")
data = data[order(data$Dataset), ]

data[-1] = round(data[-1], 3)

write.csv(data, "table.csv", row.names = F)

