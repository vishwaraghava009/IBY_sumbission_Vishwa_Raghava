Go trhough the python notebook or install the same libraries on your local machine as well by cloning the repo ad following the mentioned changes in colab notebook. 
You may ahavet to create a new groq_api_key if mine doen't work.
You may also have to create a ngok auth number as well f mine doen't work and this specifically if you are using colab for execution.
Reccomended to use L4 GPU or advanced.
Therading has been removed as it got compilacted on colab.
Currently, no enhancer has been incluyded in the pipline but we can add one after motion transfer using LIA (in views.py file).


For the model checkpoints:
1. LIA: https://drive.google.com/drive/folders/1Jnu2iY5OhdEv7PJOdSpLRLm7JNADN7-0?usp=sharing
2. MuseTalk: https://drive.google.com/drive/folders/1r629pRzt54hft7GGPZdAwKKFJF3jmvQP?usp=sharing

Please put the models directory in your drive same as the drive folders provided above if you are using colab
If you are using you local linux machine:
1. Try putting the MuseTalk models in the models directory in thge same structre as provided in the drive link.
2. For LIA vox model you have to change the run_demo.py file to rewrite the model path as mentioned in the colab noteboook as well 
