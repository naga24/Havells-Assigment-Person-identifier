# Havells-Assigment-Person-identifier
This app detects if there is a human/s in a picture. If a human is present, it classifies each human on an object level as a Zomato delivery agent or any other normal human. If there are no human/s detected it simply writes No Human detected on console using CV and DL

# Approach 

![image](https://github.com/naga24/Havells-Assigment-Person-identifier/assets/12075514/97653739-4e93-4bef-aac2-74c81c126250)

# Dataset Preparation
Used the bing image downloader to scrape images for particular search keywords to build the dataset. The keywords being 'People walking in metro station India','People walking on street','zomato','zomato delivery boy','zomato delivery partner','zomato logo

After downloading the images and filtering them, we have a total of 133 images belonging to "zomato" category and "other" selected randomly for training the classifier.

# How to run the code
1. Install anaconda for Windows (https://www.anaconda.com/download)
2. conda create -n <ENV_NAME> python=3.8
3. pip install -r requirements.txt

