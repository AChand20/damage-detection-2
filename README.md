# Damage Detection and Price Estimation of cars.
This project detects damages on cars using derectron2 and also returns the estimated repair cost. The model is deployed using `flask` and can be executed in local host server.

## Steps to run
Run the below commands in the terminal step by step

**Create conda environment**
> python version below should be between 3.7 & 3.9
```
$conda create -n <env_name> python==<version>
$conda activate <env_name>
```

**Clone this repository**

```
$git clone https://github.com/AChand20/damage-detection-2.git
$cd damage-detection-2
```
**Install dependencies**
> For windows machine
```
$requirements_windows.bat
$pip install -r requirements_windows.txt
```
> For linux machine
```
$pip install -r requirements_linux.txt
```
**Run the app**

```
$python app.py
```
