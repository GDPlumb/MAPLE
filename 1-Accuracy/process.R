
data = read.csv("results.csv")
data[-1] = round(data[-1], 6)
colnames(data) = c("Dataset", "LM", "RF", "SILO + RF", "MAPLE + RF", "Num Features RF", "GBRT", "SILO + GBRT", "MAPLE + GBRT", "Num Features GBRT")
data = data[order(data$Dataset), ]

accuracy = data[ , c(1,2,3,4,5,7,8,9)]

accuracy[-1] = round(accuracy[-1], 3)

features = data[ ,c(1,6,10)]
features$Dimension = c(8, 103, 103, 15, 8, 12, 70, 12)
features = features[c(1,4,2,3)]

write.csv(accuracy, "accuracy.csv", row.names = FALSE)
write.csv(features, "features.csv", row.names = FALSE)
