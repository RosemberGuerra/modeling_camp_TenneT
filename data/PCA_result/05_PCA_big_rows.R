PECDv4.2_NLregion_TAfeatures <- read.csv('/Users/holee/Desktop/R_test/01_paper_PhD_reading/12_ModellingCamp/data/PECDv4.2_NLregion_TAfeatures.csv',
                                         sep = ",", header = T)
head(PECDv4.2_NLregion_TAfeatures)
PECDv4.2_NLregion_TAfeatures$cy_id %>% table %>% length

pca_result <- PCA(PECDv4.2_NLregion_TAfeatures[, -1], scale.unit = TRUE, graph = F)
pca_scores <- pca_result$ind$coord
PECDv4.2_NLregion_TAfeatures_PC1 <- data.frame(PECDv4.2_NLregion_TAfeatures, PC1=pca_scores[,1])
subset.again <- PECDv4.2_NLregion_TAfeatures_PC1 %>%
  arrange(desc(PC1)) %>%
  slice_head(n = 300)
subset.again


hist(PECDv4.2_NLregion_TAfeatures$integrated_temp_distance)
hist(PECDv4.2_NLregion_TAfeatures$avg_annual_temp)
hist(subset.again$avg_annual_temp)

# Set transparency (alpha) to visualize overlap
hist(PECDv4.2_NLregion_TAfeatures$integrated_temp_distance,
     col = rgb(1, 0, 0, 0.5),  # Red with transparency
     border = "white",
     probability = TRUE,  # Normalize to probability density
     main = "Overlapping Histograms with Density Curves",
     xlab = "Average Annual Temperature")

hist(subset.again$integrated_temp_distance,
     col = rgb(0, 0, 1, 0.5),  # Blue with transparency
     border = "white",
     probability = TRUE,
     add = TRUE)  # Add to existing plot

# Add density curves
lines(density(PECDv4.2_NLregion_TAfeatures$avg_annual_temp), col = "red", lwd = 2)
lines(density(subset.again$avg_annual_temp), col = "blue", lwd = 2)

# Add legend
legend("topright", legend = c("Full Dataset", "Subset"), fill = c("red", "blue"), border = "black")



