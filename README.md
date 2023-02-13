# looting-maskrcnn
the steps to run the Colab notebook that trains a custom dataset using Mask R-CNN and data augmentation. The dataset is stored in Google Drive and needs to be mounted to the Colab environment before running the notebook.

# Prerequisites
1. A Google account with access to Google Drive
2. A Google Drive folder containing the dataset to be used for training

# Steps
1. Open the Colab notebook in Google Colab
2. Mount the Google Drive to the Colab environment. To do this, follow these steps:
   1. Click on the "Mount Drive" button in the code block.
   2. Follow the instructions to sign in to your Google account and authorize Colab to access your Google Drive.
   3. Enter the verification code provided.
3. In the code block, replace the string "path/to/dataset" with the path to the Google Drive folder containing your dataset.
4. Run the rest of the code blocks in the Colab notebook to train the Mask R-CNN model with data augmentation

the below sample of the evaluation images result 

![alt text](https://github.com/mshanah/looting-maskrcnn/blob/main/images/6.PNG?raw=true)
![alt text](https://github.com/mshanah/looting-maskrcnn/blob/main/images/7.PNG?raw=true)
![alt text](https://github.com/mshanah/looting-maskrcnn/blob/main/images/8.PNG?raw=true)

