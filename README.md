# COTS Detection

## Crown-of-thorns starfish Object Detection
![](https://i.imgur.com/FmKV4Lp.jpg)

The Great Barrier Reef in Australia is world's largest coral reef. The reef contains an abundance of marine life and comprises of over 3000 individual reef systems and coral cays. However, the Great Barrier Reef is currently under threat because of the overpopulation of COTS - Crown-of-thorns starfish. COTS are coral eating venomous starfishes and need to be controlled and provide time for corrals to grow. 

Resource
* [Publication](https://www.barrierreef.org/what-we-do/reef-trust-partnership/crown-of-thorns-starfish-control)

## Dataset
The COTS dataset can be downloaded in the following link:
* [Kaggle](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef)

## Model 
The model was created using the EfficientDet-D0 from Tensorflow Object Detection Model Zoo.   
* [Model](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

## Architecture 
The following tools were used to setup the MLOPs deployment pipeline:
- Fastapi: Main RestAPI Framework
- Docker: Images and Containers
- Streamlit: Frontend UI (In Progress)

# COTS Detection web app installation procedure

### Create new conda environment
Create a new conda environment by running the following command. 

conda create --name myenv python=3.9

(python version 3.9 was used to create the app)

### Clone COTS Object Detection repository
Clone the COTS Object Detection repository by running the following command.

git clone git@github.com:(your profile)/COTS_object_detection.git

![](![](https://i.imgur.com/sr8fjTf.png)


### To build and run docker containers

cd backend

docker build -t (docker image name) . 

docker run -d --name (docker container name) (docker image name)  

uvicorn api_main:app --reload --workers 1 --host 0.0.0.0 --port 8000

### Testing via fastapi

User can test the model via fastapi using swaggerUI by visiting http://localhost:8000/docs

![](https://i.imgur.com/ANc1rRE.png)

