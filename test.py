import argparse
import torch
import config
import models


parser = argparse.ArgumentParser(description='MI Test')
parser.add_argument('--no-cuda', action='store_false' if config.default.nocuda else 'store_true',
                    default=config.default.nocuda, help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


def test_all_models(cuda):
    models.test_cnn(cuda)
    models.test_capsule(cuda)
    models.test_resnet(cuda)
    models.test_squeezenet(cuda)
    models.test_fullscale(cuda)
    models.test_cnn3d(cuda)


if __name__ == "__main__":
    test_all_models(args.cuda)