# Covid X-ray ML Competition

## About Competition
- Competition Details: https://r7.ieee.org/montreal-sight/ai-against-covid-19/#Dates
- Dataset (Kaggle): https://www.kaggle.com/andyczhao/covidx-cxr2
- eval.ai (Submission): https://eval.ai/web/challenges/challenge-page/925/evaluation

### Opening Ceremony Notes:

- eval.ai registration of participation starts May 31st
- You may use test (more like validation) and train dataset for training the model
- test dataset is unknown


## Instructions
1. ```$ mkdir data```
2. Prepare to download dataset locally and unzip to "data/*" folder
3. TODO

## Documentation:
### Background:
1. Understanding resnet from scratch: https://jarvislabs.ai/blogs/resnet
2. Checklist on squeezing the shit out of your model: http://karpathy.github.io/2019/04/25/recipe/


### Description:
- There are two approaches to make a better predictions on given dataset:
    1. Use a decent model that works well with the task.
    2. Engineer the dataset to make the model more efficient and effective when learning.
- The base model is a simple and basic **Resnet34** (https://jarvislabs.ai/blogs/resnet), for its lightweight and adaptive properties for the given task on chest COVID detection.
- Due to limitation of my hardware (only have a GTX980Ti 6GB), I was not able to go with a deeper model and pytorch built-in model. The **Resnet34** was selected for the task, resulting a 70-80% accuracies on the evaluation test dataset provided.
- The training dataset was discovered to be quite imbalanced:
    
    ![dataset](img/dataset.png)
- For simplicity, the dataset is randomly downsampled for -ve dataset, with +ve dataset unchanged.
- To further improve the performance, we start to engineer the dataset to better utilize the model we use:
    - The initial thought is that the provided image has RGB channels exactly same to provide a black and white image, hence three channels have duplicated information, which is redundant for **Resnet34**.
    - In classical computer vision, we would use morphological operators (dilation and erosion) to extract features from the image. In addition, we figure out whether patient has COVID-19 based on the abnormal features within the chest scan. As a result, the idea is to provide **Resnet34** a sense of where the the chest region is and where the features are, with dilation and erosion respectively. Hence, we can utilize the three channels with R:(gray image), G:(erosion image), B:(dilation image), and the **Resnet34** can now fully utilize all three channels to produce a better prediction:
        ![dataset](img/rgb.png)
    - Sample training dataset becomes:
        ![Training Sample](output/CUSTOM-MODEL/v6-custom-3/plot_training-sample.png)
- As a result, the performance is quite well:
    ![Training Progress](output/CUSTOM-MODEL/v6-custom-3/training_progress[v6-custom-3].png)
- Lastly, to further push the model performance and robustness, we doubled the dataset with random zoom and rotation. To note, we have also tweaked around the learning rate and stopping criteria to find the best parameters
- To note, we pre-generate the training dataset in advance to improve the run-time efficiency.
- Overall, the best competition scored model (with just 6 epochs): 
    ![confusion matrix](output/CUSTOM-MODEL/v6-custom-with-aug-3/models/confusion_matrix_6:100.jpg)
