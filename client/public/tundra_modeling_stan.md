# Welcome to Wrattler
```markdown
Tundra analysis
```

```r
library(rstan)
library(loo)
library(tidyverse)


rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())

load("filtered_dataset.RData")


df <- subset(filtered_dataset,filtered_dataset$Trait == "Leaf nitrogen (N) content per leaf dry mass")

species <- data.frame(unique(df$AccSpeciesName))

colnames(species) <- c("AccSpeciesName")
rownames(species) <- NULL
species["species_index"] <- c(1:nrow(species))

df <- merge(df, species, by = c("AccSpeciesName"))

# Train/Test split
smp_siz = floor(0.80*nrow(df))
set.seed(42)
train_ind = sample(seq_len(nrow(df)), size = smp_siz)
train = df %>% slice(1:train_ind)
test = df %>% slice(train_ind:nrow(df))


# model
model_data <- list(n = nrow(train),s = nrow(species),y = train$Value,tmp = train$tmp.centered,species = train$species_index)

# model 3 removes the normal distribution generating alpha_sd parameters, best model according to diagnostics
fit_3 <- stan(file = 'stan_m3.stan', data = model_data, iter = 1000, chains = 4)

```
```r

tmp_ranges <- seq(0, 15, length.out = 100)
exp_model <- function(t) {
  return(exp(2.98 + 0.03 * t))
}
general_line <- lapply(tmp_ranges, exp_model)
test_line <- lapply(test$tmp.centered, exp_model)

tmp <- test$tmp

Value <-test_line


predicted_df <- data.frame(col1=tmp, col2=as.numeric((unlist(test_line))))
colnames(predicted_df) <- c("tmp","Value")

figure2 <- ggplot(df) +
    geom_point(aes(tmp, Value), size = 0.2, shape = 1, alpha = 0.7) +
    geom_smooth(aes(tmp, Value), method = lm, se = FALSE) +
    facet_wrap(~AccSpeciesName, scales = "free_y") +
    labs(x = "Temperature (ºC)", y = "Trait")




colnames(predicted_df) <- c("tmp","Value")

# Calculate error
error <- test$Value - predicted_df$Value
mean_error <- test$Value - mean(test$Value)


figure3 <- ggplot(df, aes(x = tmp, y = Value)) +
  geom_point() +
  geom_smooth(data = train, method = lm, aes(color = "green")) + geom_smooth(data = test, method = lm, aes(color = "blue")) +# geom_smooth(data = predicted_df, method = lm, aes(color = "red")) +
  labs(x = "Temperature", y = "Trait value") + ggtitle("Trait example: Leaf nitrogen (N) content per leaf dry mass (mg/g)") +
  theme(axis.text=element_text(size=18), axis.title=element_text(size=21), plot.title = element_text(size=15, hjust = 0.5)) +
  scale_colour_manual(name="Legend", values=c("green", "blue", "red"), labels = c("Train", "Test", "Predicted from Test")) + guides(fill=TRUE)


```
