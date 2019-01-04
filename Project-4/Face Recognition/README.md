
## Face recognition through eigenfaces 

- **Generating the data for face recognition**
  There are open sources data available for face recognition but for custom data set to train you can use the script to get your own dataset. Just run the command :
  `python -B dataset.py`
  
 - **Preparing the training data**
 We need to create a CSV file. I have already added the code for it in the `dataset.py` script.
 
 - **Loading images**
 script for the same is written in `load_data.py`. The module in imported in the main file.
 
 - **Performing Eigenfaces recognition**
 This is the main script to train on the dataset that you created using `dataset.py`. 
 
Run the following command : 
argument to be passed in the command- `[file] <path/to/image> [<path/to/store/eigen faces>]`

`python -B main.py "C:/Users/ARPIT JAIN/Desktop/Face Recognition" "C:/Users/ARPIT JAIN/Desktop/Face Recognition/output"`
