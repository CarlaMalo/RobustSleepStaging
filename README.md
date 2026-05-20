# RobustSleepStaging

RobustSleepStaging is a research codebase for automatic sleep stage classification using the Sleep-EDF dataset. The primary objective is to evaluate distribution shift between recording conditions (Sleep-Cassette and Sleep-Telemetry) and to measure model robustness across these domains. 
The repository provides preprocessing, feature extraction, model training and evaluation scripts and Jupyter notebooks to visualize the results.

## Environment Setup

1. Clone the repository
    ```
    git clone https://github.com/CarlaMalo/RobustSleepStaging.git
    cd RobustSleepStaging 
    ```
2. Create a new environment: 
    `python -m venv .venv`
2. Activate environment: 

   Windows: `.venv\Scripts\activate`

   Unix-based: `source .venv/bin/activate`
4. Install dependencies: 

   `pip install -r requirements.txt`
    
6. Download data from https://physionet.org/content/sleep-edfx/1.0.0/ and copy them to folder data.

7. Run the workflow in **RobustSleepStagingV2.ipynb**

8. (Optionally) After running the workflow (which created the feature and model checkpoints) you can run **example_features.ipynb** to analyze the feature selection and the **example_hypnogram.ipynb** to visualize the predictions. 

## Team Members
- Alexandros Christopoulos
- Carla Malo
- Daksa Chellappah
- Enrique Perez
- Roland Widmer

## Additional experiments

The pipeline design supports other model architectures such as CNN. 
Run the workflow experiment in **RobustSleepStagingCNN.ipynb**


(along with the requirements_cnn_with_gpu.txt)