<h1><center>Pipeline HCM AI CHALLENGE 2024 <br> Event Retrieval from Visual Data</center></h1>

## Setup 
```
pip install -r requirements.txt
sudo apt-get install pkg-config libcairo2-dev
pip install pycairo
pip install httpx==0.13.3
pip install numpy==1.23.5
pip install torch==2.4.0
pip install --upgrade torchvision
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
sudo apt-get install apt-transport-https
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-7.x.list
sudo apt-get update
sudo apt-get install elasticsearch
```

## Run 
```
sudo service elasticsearch start
curl -X GET "localhost:9200/"
uvicorn main:app --reload
```
