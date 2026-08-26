# keras-paddle

PaddlePaddle backend implementation for Keras.

## Local development

For development, you need the `keras` repository and the `keras-paddle` repository checked out locally. That's because the unit tests code is in the `keras` repository.

We first check out the main `keras` repository and the `pluggable_backend` branch.

```bash
gh repo clone keras-team/keras
cd keras
git checkout pluggable_backend
pip install -r requirements-common.txt
cd ..
```

Assuming you have a fork of `keras-paddle`, you will run the following. This also installs `keras-paddle` locally so that `keras` can find and import the `keras-paddle` module.

```bash
gh repo clone <your_github_handle>/keras-paddle
cd keras-paddle
pip install -r requirements.txt
pip install -e .
cd ..
```

Running tests happens from the root of the `keras` repository.

```bash
cd keras
KERAS_BACKEND=paddle pytest keras --ignore=keras/src/applications \
    --ignore=keras/src/wrappers
```
