<h1><center>Pipeline HCM AI CHALLENGE 2023 <br> Event Retrieval from Visual Data</center></h1>

## Setup 
```
pip install -r requirements.txt
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
watchmedo auto-restart --directory=./ --pattern=*.py --recursive -- uvicorn main:app --reload
```
