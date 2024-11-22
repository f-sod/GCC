library(varrank)
library(dplyr)


# discretization.method with number -> equal binning see FS paper + https://www.math.uzh.ch/pages/varrank/reference/discretization.html
# method: kwak -> see https://www.math.uzh.ch/pages/varrank/reference/varrank.html
# beta (here called ratio) must be 1 when applying the def in my thesis
# we want to optimize for the discretized drug response
# data is the feature matrix + response column
args = commandArgs(TRUE)
receptor = args[1]
X_data = read.table(paste('../Data/Output/', receptor, '_X_train.csv', sep = ''), header = TRUE, sep ="\t")
y_data = read.table(paste('../Data/Output/', receptor, '_y_train.csv', sep = ''), header = TRUE, sep ="\t")
data = bind_cols(X_data, y_data)
data = subset(data, select = -CID)
data = subset(data, select = -X)
resp = data[receptor]

data$discretized.response = resp
out = varrank(data.df = data, variable.important = receptor, method = "kwak", 
                 discretization.method = 6, ratio = 1, verbose = TRUE,scheme = "miq", algorithm="forward")

write.table(out$ordered.var,paste('../Data/Output/', receptor, '_feature_selection.csv', sep = ''), sep = '\t', col.names = FALSE, quote = FALSE)