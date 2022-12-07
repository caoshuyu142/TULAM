# TULAM

PyTorch implementation for paper: TULAM

## Dependencies

- Python 3.8
- torch == 1.9.0
- scikit-learn == 1.0.2
- tqdm == 4.63.0
- pandas == 1.4.1
- numpy == 1.12.3
## Datasets

#### Raw data:

- Foursquare：https://sites.google.com/site/yangdingqi/home/foursquare-dataset
- Gowalla: [SNAP: Network datasets: Gowalla (stanford.edu)](http://snap.stanford.edu/data/loc-gowalla.html)

#### Preprocessed data:

- Each dataset should contain at least 4 columns, which must be transformed as blew:
  - userID
  - latitude
  - longitude
  - utc: UTC timestamp

#### sample data

- The sample data is uploaded in the folder [data](./data).
- the CSV file contain 5 columns:
  - userID
  - TrID
  - rowID
  - colID
## Usage
- Run main.py
- Adjust the hyperparameters and strategies according to the needs
  - e.g. ```python main.py --dataset gowalla --isDense D```

##Citation
If you want to use our codes in your research, please cite:
