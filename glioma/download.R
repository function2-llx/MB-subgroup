library(TCGAbiolinks)

library(openxlsx)

lgg_subtype <- TCGAquery_subtype(tumor = "lgg")

write.xlsx(lgg_subtype, file = "lgg_subtype.xlsx")

gbm_subtype <- TCGAquery_subtype(tumor = "gbm")

write.xlsx(gbm_subtype, file = "gbm_subtype.xlsx")
