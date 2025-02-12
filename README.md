# Multiday-Transformer
A short project which investigates whether stitching together training data from many different neural recording sessions improves single day performance of a transformer decoder when compared to training on a single day. 

### Introduction

Recent successes of large language models have shown the potential of transformer architectures [1]. In particular, these advances have shown that scaling up the size of the training set can drastically improve model performance. Unfortunately, decoding from neurons is more complex than decoding from language. Unlike words in language, neurons shift over time so that subsequent days cannot be aligned [2]. To overcome this, neural transformer models utilize various tokenization techniques to group together like neurons while maintaining temporal and spatial information [3,4]. Further, these models incorporate fine tuning layers for the specific decoding task. With these additions, large neural transformer models have shown promising results.

As an initial step towards replicating these complex models, this report investigates whether stitching together training data from many different sessions improves single day performance of a transformer decoder when compared training on a single day.

### Results:


As can be seen in Figure 1, it appears that training across multiple days has the  potential to augment decoder performance. The box plot in Figure 1 shows each decoder's performance, relative to the applicable single day model, for the first ten days of the test set.

It is notable that the mean of the first model is positive, showing improvement over the single day model. It is also interesting to note, that can be seen in Figure 2, that the correlations of the multi-day models seem to match themselves even when diverging from the single day model. This, along with the worse performance of larger models in Figure 1, could be due to the multi-day models working to minimize the loss across all training days, whereas the single day model only minimizes the loss for the single day.


Figure 1. A box plot comparing the 4 multi-day models over their first ten days of data to the corresponding single day model is shown with correlation coefficient as the metric. 


Figure 2. Correlation coefficient for each model across all of its testing days. Note that models drop to the bottom of the screen when the day number exceeds their training days, as each model learns a specific input layer for each individual day. 
