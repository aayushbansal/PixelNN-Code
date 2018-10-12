There are multiple parts of the code -- 

1. Getting an intermediate output from a regression model. We use PixelNet to train a model for the intended task using simple l-2 regression. 

2. We then generate the data for the trained model for both the training and test data. Essentially, we generate a triplet in the training set which is then used to do nearest neighbors. An example of this code is available in experiments/faces

3. The second part of the code is about the nearest neighbors. A naive pixel-wise nearest neighbors could take forever. We reduce our search space by finding global nearest neighbors (using conv-5 features from a pretrained AlexNet). An example of this code is available in nn/global. Note that in this code, it is a naive nearest neighbor but efficient methods such as k-d trees could also be used.

4. Finally, the part of the code that helps us to utilize the information from the nearest neighbors in nn/local. I have added example for faces, shoes and bags. 

I have tried to add the required data, but in case something is missing or you want more details, please feel free to mail at aayushb@cs.cmu.edu
