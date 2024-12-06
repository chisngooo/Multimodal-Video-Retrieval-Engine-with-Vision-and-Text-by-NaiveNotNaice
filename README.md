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


## Demo Video

<style>
  .video-thumbnail {
    position: relative; /* Đảm bảo phần tử này là container cho nút Play */
    width: 600px; /* Đặt kích thước cố định cho thumbnail */
    height: auto;
    display: inline-block;
  }

  .play-icon {
    position: absolute; /* Đặt nút Play ở vị trí tuyệt đối trong container */
    top: 50%; /* Căn giữa theo chiều dọc */
    left: 50%; /* Căn giữa theo chiều ngang */
    transform: translate(-50%, -50%); /* Điều chỉnh vị trí tâm */
    width: 80px; /* Kích thước của nút Play */
    height: 80px;
    opacity: 0.8; /* Độ trong suốt */
    transition: all 0.3s ease; /* Tạo hiệu ứng khi hover */
  }

  /* Hiệu ứng hover: nút Play rõ ràng hơn */
  .video-thumbnail:hover .play-icon {
    opacity: 1;
  }

  /* Hiệu ứng hover cho ảnh thumbnail */
  .video-thumbnail:hover {
    opacity: 0.8;
  }
</style>

<p align="center">
  <a href="https://drive.google.com/file/d/12bVfUk2ctRZkEcpc4jaMZptK0BWoH2Tq/view?usp=drive_link" class="video-thumbnail">
    <img src="./image/demo_video.jpg" width="600" alt="Demo Video Thumbnail"> <!-- Đường dẫn ảnh thumbnail -->
    <img src="./image/play_button.png" class="play-icon" alt="Play Icon"> <!-- Đường dẫn nút Play -->
  </a>
</p>


<p align="center">
  <a href="https://drive.google.com/file/d/12bVfUk2ctRZkEcpc4jaMZptK0BWoH2Tq/view?usp=drive_link" class="video-thumbnail">
    <img src="/image/demo_video.jpg" width="600">
    <img src="/image/play_button.png" class="play-icon"> 
  </a>
</p>
