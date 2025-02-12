# Multiday-Transformer
A short project which investigates whether stitching together training data from many different neural recording sessions improves single day performance of a transformer decoder when compared to training on a single day. 

[[PDF]](https://github.com/omerbe/Multiday-Transformer/blob/main/Research%20report%20Fall%2024.PDF)
### Introduction

Recent successes of large language models have shown the potential of transformer architectures. In particular, these advances have shown that scaling up the size of the training set can drastically improve model performance. Unfortunately, decoding from neurons is more complex than decoding from language. Unlike words in language, neurons shift over time so that subsequent days cannot be aligned. To overcome this, neural transformer models utilize various tokenization techniques to group together like neurons while maintaining temporal and spatial information. Further, these models incorporate fine tuning layers for the specific decoding task. With these additions, large neural transformer models have shown promising results.

As an initial step towards replicating these complex models, this report investigates whether stitching together training data from many different sessions improves single day performance of a transformer decoder when compared training on a single day.

### Results:

![alt text](https://github.com/omerbe/Multiday-Transformer/blob/main/multiday_transformer_fig1.png)

Figure 1. A box plot comparing the 4 multi-day models, evaluated over the first ten days of data, to the corresponding single day model is shown with correlation coefficient as the metric. This appears to show that training across multiple days has the potential to augment decoder performance, although there seems to be worsening performance with larger training sets. This could be due to the multi-day models working to minimize the training loss across all training days, whereas the single day model only minimizes the training loss for the day being tested. 

![alt text](https://github.com/omerbe/Multiday-Transformer/blob/main/multiday_transformer_fig2.png)

Figure 2. Correlation coefficient for each model across all of its testing days. Note that models drop to the bottom of the screen when the day number exceeds their training days, as each model learns a specific input layer only for each individual day in its training set. It is also interesting to note that the correlations of the multi-day models seem to match themselves even when diverging from the single day model. 
