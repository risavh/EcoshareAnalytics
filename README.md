# EcoshareAnalytics


## Project Setup Instructions

**To run the best model to generate predictions follow the below steps:**

1. Clone this repository: `git clone https://github.com/risavh/EcoshareAnalytics.git`
2. Create and activate a virtual environment:
   - Virtualenv: `python -m venv venv` and `source venv/bin/activate`
3. Install project dependencies: `pip install -r requirements.txt`
4. To predict on test / unseen / new data
   - update the 'FILE_LOCATION'.'TEST_DATA' in config.yaml file under src folder to reflect the test data location.
   -  Run predict.py under src folder to generate predicted output in output folder as rishabh_gupta_predictions.pkl filename.
   -  `predict.py`


---

**To reproduce the train models / run through the EDA follow the below steps:**

1. Clone this repository: `git clone https://github.com/risavh/EcoshareAnalytics.git`
2. Change the directory to notebooks
   - `cd notebooks`
3. Create and activate a virtual environment:
   - Virtualenv: `python -m venv venv` and `source venv/bin/activate`
4. Install project dependencies: `pip install -r requirements_train.txt`
5. Run through Jupyter Notebooks after changing the kernel to virtual env.

