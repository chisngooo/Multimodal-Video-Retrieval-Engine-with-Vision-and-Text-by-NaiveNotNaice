<h1><center>Pipeline HCM AI CHALLENGE 2024 <br> Event Retrieval from Visual Data</center></h1>

## Setup 
```
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
sudo apt-get install apt-transport-https
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-7.x.list
sudo apt-get update
pip install -r requirements.txt
gdown --folder https://drive.google.com/drive/folders/1FnoIyy_DewSigiX6H121PhPNqIHz4ySB -O data
```

## Run 
```
sudo service elasticsearch start
curl -X GET "localhost:9200/"
uvicorn main:app --reload
```
