<p align="center">
  <img src="/image/AIC2024-Banner.png" width="1080">
</p>

<h1 align="center">Cross-Modal-Video-Retrieval-Engine-with-Vision-and-Text </h1>
![Static Badge](https://img.shields.io/badge/python->=3.10-blue)


## Setup 
```
wget -qO - https://artifacts.elastic.co/GPG-KEY-elasticsearch | sudo apt-key add -
sudo apt-get install apt-transport-https
echo "deb https://artifacts.elastic.co/packages/7.x/apt stable main" | sudo tee -a /etc/apt/sources.list.d/elastic-7.x.list
sudo apt-get update
pip install -r requirements.txt
gdown --folder https://drive.google.com/drive/folders/1n5sLuf9YckDArIktTLd4WRwhl1XxUy7B?hl=vi -O data/bin
gdown --folder https://drive.google.com/drive/folders/16GueLfWnK4yQtPbsaBe8QNk-zUga-QDo?dmr=1&ec=wgc-drive-globalnav-goto -O data/bin
                
```

## Run 
```
sudo service elasticsearch start
curl -X GET "localhost:9200/"
uvicorn main:app --reload
```


## Demo

<p align="center">
  <img src="/image/demo1.jpg" width="600">
</p>

<p align="center">
  <img src="/image/demo2.jpg" width="600">
</p>

<p align="center">
  <img src="/image/demo3.jpg" width="600">
</p>
