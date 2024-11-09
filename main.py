from cnn_model import ActivationFunction, InitMethod
from train import train


def main():
    train(ActivationFunction.RELU, InitMethod.KAIMING)


if __name__ == '__main__':
    main()