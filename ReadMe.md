##Requirements:
- CUDA Enabled GPU hardware
- python == 3.6.9
- tensorflow-gpu==1.15.0
- tensorflow_probability == 0.7


Data Preperation:
After preprocessing data will be placed in "data/<dataset>" folder as pickle file format.
1. For CIFAR10 dataset, no pre-processing is required and pickle files can be downloaded directly into "data/cifar" folder.

2. Download ImageNet data directly in "Raw_data/ImageNet" folder. We have used ILSVRC2012_img_train dateset. To buid Custom Restricted ImageNet data, necessary files are already provided in "Raw_data/ImageNet/imagenet_info".
Following Code will build custom imagenet dataset and save it in "data/imagenet"
  - python prepare_imagenet.py

3. Download GTSRB data in "Raw_data/GTSRB" folder and run following code to convert it into pickle files.
  - prepare_GTSRB_data.py



Most of the codes will require dataset name. This value should be passed as cifar, imagenet or gtsrb.

Training:
  1. Classifier:
        Input: Training and Test data files.
        Output: Trained model.
        - python Classifier/train.py -i <data_dir_path> -d <dataset> -c <Num of Classes>
        The trained model will be saved in Classifier/<dataset>_Model/ResNet50_ckpt/ folder.

  2. Pixel-CNN
        Input: Training and Test data files.
        Output: Trained model.
        - python pixel-cnn/train.py -d <dataset>
        Trained model will be saved in pixel-cnn/<dataset>_Model/params_<dataset>.ckpt.

  3. Train GMM
        To train GMM model, first it requires representation vector of all training/validation data.
              - python GMM/prepare_representation.py -d <dataset> -f clean
              Output: Representation data in data/<dataset>/Representation folder

        - python GMM/train_GMM.py -d <dataset>
        Output: Class conditional GMM model saved in GMM/GMM_Models/<dataset>_GMM


Attack:
    Input: Test data and trained classifier model
    Output: Adversarial examples will be saved in "data/<dataset>/Adversarial/" folder.
    To attack all samples give num_samples = -1.

      1. L2 attacks  [PGD,FGSM,MIM]
          - python Attacks/attack_Linf.py -d cifar -n 1000
      2. L-inf attacks [CW,DeepFool]
          - python Attacks/attack_L2.py -d cifar -n 1000
      3. WhiteBox Attack
          - python WhiteBox_attack.py -d <cifar> -ns 1000


View Generation:
      For precise analysis, it's important that benign test files should only contain samples that are correctly classified.
      Following code will save correctly classified samples in data/<dataset>/test_c
      - python Classifier/test.py - d <dataset>

      To generate views for benign and adversarial samples: Following code will save files in data/<dataset>/Generated/
      - python pixel-cnn/generate_views.py -d <dataset> -n <num_samples>
      To generate views for all samples give num_samples = -1. Though view generation is a time consuming step. So either keep num_samples value low or expect this code to finish in 2-3 days.
      The approximate detection rate that are quite close to True detection rate can be obtained using num_samples = 1000 for cifar/gtsrb. For imagenet test samples are already close to 1000.
      The above code generate views for each files independently. Thus they can be generated seperately by modifying line 224.

Detection:
      Step 1: Feature/Descriptor extraction using generated views. Output of the following code will save files in data/<dataset>/Final_Descriptors
      - python Final_Features.py -d <dataset>

      Step 2: Training and Evaluation for Adversarial detector. Output of the following code will be AUC Score for all attack methods.
      - python  Adversarial_Detector.py -d <dataset>
