# Havells-Assigment-Person-identifier
This app detects if there is a human/s in a picture. If a human is present, it classifies each human on an object level as a Zomato delivery agent or any other normal human. If there are no human/s detected it simply writes No Human detected on console using CV and DL

# Approach 

![image](https://github.com/naga24/Havells-Assigment-Person-identifier/assets/12075514/97653739-4e93-4bef-aac2-74c81c126250)

# Dataset Preparation
Used the bing image downloader to scrape images for particular search keywords to build the dataset. The keywords being 'People walking in metro station India','People walking on street','zomato','zomato delivery boy','zomato delivery partner','zomato logo

After downloading the images and filtering them, we have a total of 133 images belonging to "zomato" category and "other" selected randomly for training the classifier.

# Download Dataset and models

1. Dataset : [Dataset](https://drive.google.com/file/d/1P5vimEYBsyosyJTzGebfrg1qFffFVg3x/view?usp=drive_link)
2. Model : [Trained Model](https://drive.google.com/file/d/1_lS1_FwlIx98HV5ADI6odYi6sUA0krLk/view?usp=drive_link)

After downloading, place model.h5 in base folder. Unzip and copy the "data" folder and paste it in base folder

# How to run the code
1. Install anaconda for Windows (https://www.anaconda.com/download)
2. conda create -n <ENV_NAME> python=3.8
3. conda activate <ENV_NAME>
4. pip install -r requirements.txt

# Hyperparameters Used in training

IMAGE_SIZE: (224, 224) - The dimensions to which input images are resized.
BATCH_SIZE: 4 - The number of samples per batch during training.
EPOCHS: 50 - The number of times the entire training dataset is passed forward and backward through the neural network.
NUM_CLASSES: 2 - The number of classes in your dataset (used for classification).
Additional hyperparameters can be inferred indirectly from the code
Learning Rate: 1e-3 - Defined in the optimizer (Adam) as the learning rate parameter.
Dropout : 0.5

# Run Training

```python train.py```

# Run Evaluate model

```python evaluate.py```

Performance metric : F1-score, For our use case, classifier performs at 67 % F1-score

# Run model on sample image

```python test.py```

# To run the full flow use the Streamlit service

Go to streamlit_app folder, from there run

```streamlit run app.py```

# Limitations & Improvement

The classification model seems to be baised towards the zomato class. This is caused by insufficient data and also the model not able to distinguish the features properly. This can be improved by using logo detection i.e detect the zomato logo, if logo is detected then the person is a zomato delivery agent. 

# References

1. [Person Detection Code](https://pyimagesearch.com/2021/08/02/pytorch-object-detection-with-pre-trained-networks/)
2. [ChatGPT 3.5](https://chat.openai.com/g/g-F00faAwkE-open-a-i-gpt-3-5)
3. [bing-image-downloader](https://pypi.org/project/bing-image-downloader/)

# Queries

<nagarjun.gururaj@gmail.com>