# Development of an Artificial Intelligence–Based Model to Precisely Measure Body Surface Area in Psoriasis


## install
```
pip intsall -r requirements.txt
```

## Model Training and Evaluation

```
cd huaxiSkin
python train.py   # Please prepare the datasets
python eval.py

```
## API server
```
cd skinServer
nohup python -u skinServer.py > server.log &
python skinClient.py # Verify service availability
```