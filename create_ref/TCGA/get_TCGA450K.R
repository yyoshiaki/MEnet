library(TCGAbiolinks)

query.met <- GDCquery(project = c("TCGA-ESCA", "TCGA-PAAD"),
                      data.category = "DNA Methylation",
                      legacy = FALSE,
                      platform = c("Illumina Human Methylation 450"),
                      sample.type = "Solid Tissue Normal")

met <- getResults(query.met)
met
