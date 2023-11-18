# EcoshareAnalytics


## Project Setup Instructions

1. Clone this repository: `git clone https://github.com/risavh/EcoshareAnalytics.git`
2. Create and activate a virtual environment:
   - Virtualenv: `python -m venv venv` and `source venv/bin/activate`
3. Install project dependencies: `pip install -r requirements.txt`
4. To predict on test / unseen / new data
   - update the 'FILE_LOCATION'.'TEST_DATA' in config.yaml file under src folder to reflect the test data location.
   -  Run predict.py under src folder to generate predicted output in output folder as rishabh_gupta_predictions.pkl filename.
