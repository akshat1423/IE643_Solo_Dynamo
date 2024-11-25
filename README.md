
# Feature Fusion Attention Network with CycleGAN for Image Dehazing, De-Snowing and De-Raining with limited domain Images

This model presents a novel approach to image

dehazing by combining Feature Fusion Attention (FFA) net-
works with CycleGAN architecture. Our method leverages both supervised and unsupervised learning techniques to effectively
remove haze from images while preserving crucial image details.

The proposed hybrid architecture demonstrates significant im-
provements in image quality metrics, achieving superior PSNR
and SSIM scores compared to traditional dehazing methods.

Through extensive experimentation on the RESIDE and Dense-
Haze CVPR 2019 dataset, we show that our approach effectively
handles both synthetic and real-world hazy images. CycleGAN
handles the unpaired nature of hazy and clean images effectively,
enabling the model to learn mappings even without paired data.


## Report 

- [Feature Fusion Attention Network with CycleGAN for Image Dehazing, De-Snowing and De-Raining ](https://github.com/akshat1423/IE643_Solo_Dynamo/blob/master/Project%20Report.pdf)


## Installation

## Dataset Setup for RESIDE and Related Projects

This guide explains how to set up your environment to download datasets from Kaggle using the Kaggle API and prepare them for use in the project.

---

### Prerequisites

1. **Python Installed**  
   Ensure you have Python 3.6 or later installed.  

2. **Kaggle Account**  
   Create a Kaggle account if you don’t already have one: [https://www.kaggle.com/](https://www.kaggle.com/).

---

### Step 1: Install Kaggle API

1. Install the Kaggle API using `pip`:
   ```bash
   pip install kaggle
    ```
2. Verify the installation:

```bash
    kaggle --version
```

### Step 2: Set Up Kaggle API Credentials
Log in to your Kaggle account and go to the API section of your account settings.

Click Create New API Token. This will download a file named kaggle.json.

Place the kaggle.json file in the following directory:

Linux/Mac: 
``` bash 
 ~/.kaggle/ 
 ```
Windows: 
``` bash 
%HOMEPATH%\.kaggle\
```
Set correct permissions for the file (Linux/Mac only):

```bash
chmod 600 ~/.kaggle/kaggle.json
```

### Step 3: Download and Extract Datasets
Run the provided script download_datasets.py to download and prepare the datasets. It automates the process of downloading, extracting, and organizing datasets.

Run the script:
```bash 
python download_datasets.py 
```

### Step 4: Verify the Datasets

After running the script, the datasets should be available in the following directories:
```bash 
../datasets/hazing-images-dataset-cvpr-2019
../datasets/indoor-training-set-its-residestandard
../datasets/synthetic-objective-testing-set-sots-reside
```
Check the contents of these directories to ensure the datasets are extracted properly.


## Train the Model

Install dependecies
```bash
pip install torch torchvision matplotlib pillow
```

To start training the model 
```bash
python main.py
```

Note: Change the parameter of ```num_paired``` 25, 10, 5 or 0 as per your desired fine tuning condition



## Test Model - Web Interface using Flask

Install the necessary libraries
```bash
pip install flask torch torchvision pillow numpy scikit-image opencv-python
```
### Running the Application
1. **Start the Flask Server**
Run the Flask application from :
```bash
python app.py
```
By default, the application will run on http://127.0.0.1:5000.

2.  **Access the Web Interface**
Open your web browser and navigate to:
```http://127.0.0.1:5000```
You will see a form where you can upload an image, select the number of paired images used for fine-tuning, and process the image.

3. **Processing an Image**
    1. Upload the image: Click on the "Choose File" button and upload a hazy image.

    2. Select fine-tuning parameters:

    Enter the number of paired images (e.g., 0, 5, 10, 15, 20, or 25).
    This parameter affects the model's fine-tuning adjustments.

    3. Click "Process": The application will process the image and return the following:
    Cleaned image (dehazed)
    Metrics: PSNR (Peak Signal-to-Noise Ratio) and SSIM (Structural Similarity Index)


4. **Output**
The output includes:

   1. A dehazed image displayed in the web interface.
   2. The calculated PSNR and SSIM values.

## Code Walkthrough 
- [Code Walkthrough Video ](https://drive.google.com/file/d/1zqBe_orSMsKyb3Mg4WJuIHOkVsCtgY5-/view?usp=drive_link)

## Acknowledgements and References

- [J.-Y. Zhu, T. Park, P. Isola, and A. A. Efros, “Unpaired image-to-image translation using cycle-consistent adversarial networks,” 2020](https://arxiv.org/abs/1703.10593).  
- [B. Cai, X. Xu, K. Jia, C. Qing, and D. Tao, “Dehazenet: An end-to-end system for single image haze removal,” *IEEE Transactions on Image Processing*, vol. 25, no. 11, pp. 5187–5198, 2016](https://arxiv.org/pdf/1601.07661).  
- [W. Yang, R. T. Tan, J. Feng, J. Liu, Z. Guo, and S. Yan, “Deep joint rain detection and removal from a single image,” 2017](https://arxiv.org/abs/1609.07769).  
- [C. O. Ancuti, C. Ancuti, M. Sbert, and R. Timofte, “Dense haze: A benchmark for image dehazing with dense-haze and haze-free images,” in IEEE International Conference on Image Processing (ICIP), 2019](https://www.kaggle.com/datasets/rajat95gupta/hazing-images-dataset-cvpr-2019?select=hazy).  
- [C. O. Ancuti, C. Ancuti, R. Timofte, L. V. Gool, L. Zhang, and M.-H. Yang, “Ntire 2019 image dehazing challenge report,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition Workshops, 2019](https://www.kaggle.com/datasets/rajat95gupta/hazing-images-dataset-cvpr-2019?select=hazy).  
- [B. Li, W. Ren, D. Fu, D. Tao, D. Feng, W. Zeng, and Z. Wang, “Benchmarking single-image dehazing and beyond,” *IEEE Transactions on Image Processing*, vol. 28, no. 1, pp. 492–505, 2019](https://www.kaggle.com/datasets/balraj98/indoor-training-set-its-residestandard).  
- [B. Li, W. Ren, D. Fu, D. Tao, D. Feng, W. Zeng, and Z. Wang, “Benchmarking single-image dehazing and beyond,” *IEEE Transactions on Image Processing*, vol. 28, no. 1, pp. 492–505, 2019](https://www.kaggle.com/datasets/brunobelloni/outdoor-training-set-ots-reside).  
- [A. Hu, Z. Xie, Y. Xu, M. Xie, L. Wu, and Q. Qiu, “Unsupervised haze removal for high-resolution optical remote-sensing images based on improved generative adversarial networks,” *Remote Sensing*, vol. 12, p. 4162, 2020](https://www.researchgate.net/publication/347787080_Unsupervised_Haze_Removal_for_High-Resolution_Optical_Remote-Sensing_Images_Based_on_Improved_Generative_Adversarial_Networks).  
