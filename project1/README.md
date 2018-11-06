[![Build Status](https://travis-ci.com/mortele/FYS-STK4155.svg?branch=master)](https://travis-ci.com/mortele/FYS-STK4155)
[![codecov](https://codecov.io/gh/mortele/FYS-STK4155/branch/master/graph/badge.svg)](https://codecov.io/gh/mortele/FYS-STK4155)
[![Project Status: Unsupported – The project has reached a stable, usable state but the author(s) have ceased all work on it. A new maintainer may be desired.](https://www.repostatus.org/badges/latest/unsupported.svg)](https://www.repostatus.org/#unsupported)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/435d7b5fb7a44d69a79081d68e8f71a6)](https://app.codacy.com/app/mortele/FYS-STK4155?utm_source=github.com&utm_medium=referral&utm_content=mortele/FYS-STK4155&utm_campaign=Badge_Grade_Settings)


[Exercise test](https://compphysics.github.io/MachineLearning/doc/Projects/2018/Project1/html/Project1-bs.html)

### Project 1: Linear regression models (*deadline 08.10.2018.*)

#### Downloading and building
Clone the repository
```
git clone git@github.com:mortele/FYS-STK4155.git morteleFYS-STK4155
cd morteleFYS-STK4155
```
and install the requirements contained in `requirements.txt` by 
```
pip3 install -r requirements.txt
```


#### Automatic testing
The `pytest` library is used for automatic unit-testing of the source code, contained in the `src` folder. Running 
```
pytest -v
``` 
from anywhere within the repository will run all the tests deviced for this project. The test coverage (percentage of the code tested by at least one such unit-test) is shown above. It is automatically calculated using the `pytest` and the `coverage` library with TravisCI every time a new commit is pushed. 

