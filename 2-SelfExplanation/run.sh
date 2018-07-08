rm results.csv
rm Trials/*
rm table.csv
rm stats.txt

python run.py

Rscript process.R
