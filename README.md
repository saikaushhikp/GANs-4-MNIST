# GANs 4 MNIST
A simple implementation of Generative Adversarial Networks for MNIST dataset.

## How to run?
- clone the repo
```bash
git clone https://github.com/saikaushhikp/GANs-4-MNIST.git
```
- install `torchsummary` as
```bash
pip install torchsummary
```
- if not already installed (although it's need isn't required if you don't want model summarization via the code below, just comment it in [`run.py`](run.py)
```python
summary(D, input_size=(1, 28, 28))

summary(G, input_size=(1, noice_dim))

```
- and then , just run the file [`run.py`](run.py) which loads the functions defined in [`utilizations.py`](utilizations.py) and then trains the **GAN** as per the code in [`run.py`](run.py)
```bash
python3 run.py
```
