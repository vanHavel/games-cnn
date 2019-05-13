# games-cnn
![Project Cars screenshot](https://vanhavel.github.io/img/games-cnn/project_cars.jpg)

**Project Cars** ([source](http://store.steampowered.com/app/234630/Project_CARS/))
 - Predicted Genres: *Sports, Racing*
 - True Genres: *Sports, Racing, Simulation*

Recognizing game genres from screenshots using convolutional neural networks for multi label learning. 
For more details see [this blog post](https://vanhavel.github.io/2017/09/12/cnn-games.html).

The code is provided mostly for illustrative purposes. It was written with the memory limits of my machine in mind and might not be completely portable.

If you want to **classify some images** yourself, you can use the script `predict/classify_image.py`. It takes as arguments a lists of paths to jpg images, which you need to supply, and a path to a keras model and a threshold file.

## System Requirements
 - Python3
 - TensorFlow
 - Keras

## Repository Structure
`preprocess` contains some web crawling scripts(`get_ids.py`,`get_data.py`) to get the training data from the Steam store. 

`train` contains scripts to create the training data(`create_training_data.py`) and train a model pretrained on ImageNet(`train_pretrained_model.py`). It also contains an implementation of bp-mll, see [this repository](https://github.com/vanHavel/bp-mll-tensorflow).

`predict` contains scripts to learn optimal thresholds(`get_cutoffs.py`), evaluate a model on the test data(`evaluate.py`) and classify new images(`classify_image.py`).

`cutoffs` contains the threshold files for the best model I trained, plus the default thresholds (0.5).
