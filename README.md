## Environment setup

```
python3 -m pip install -r requirements.txt
```

## Model Training

To launch training you can just run hypermodel.py

```
python3 hypermodel.py
```

## Model Evaluation

To start evaluating run ...
```
python3
```

## VeHiCLE model

VEHiCLE, a DL algorithm for resolution enhancement of Hi-C contact data.

![vehicle img](./imgs/vehicle.png)

```
@article{VEHiCLE,
  title={VEHiCLE: a Variationally Encoded Hi-C Loss Enhancement algorithm for improving and generating Hi-C data},
  author={Highsmith, Max and Cheng, Jianlin},
  journal={Scientific Reports},
  volume={11},
  number={1},
  pages={1--13},
  year={2021},
  publisher={Nature Publishing Group}
}
```

## Akita model

Sequence-to-image model that accurately predicts genome folding from DNA sequence.

![hybrid img](./imgs/akita_nature.png)

```
@article{Akita,
  title={Predicting 3D genome folding from DNA sequence with Akita},
  author={Fudenberg, Geoff and Kelley, David R and Pollard, Katherine S},
  journal={Nature Methods},
  volume={17},
  number={11},
  pages={1111--1117},
  year={2020},
  publisher={Nature Publishing Group}
}
```

## Hybrid model

The key idea of our model is incorporation of information about a DNA sequence to a low resolution image to obtain an image with higher resolution

![hybrid img](./imgs/Model_combining.png)
