"""Medical image transforms."""
import random
import torch
import torchvision.transforms as T

class MedicalTransform:
    def __init__(self, img_size=224, training=True, normalize=True, strength=0.5):
        t = [T.Resize((img_size, img_size))]
        if training:
            t += [T.RandomHorizontalFlip(0.5), T.RandomVerticalFlip(0.3), T.RandomRotation(15),
                  T.ColorJitter(0.2*strength, 0.2*strength, 0.1*strength), T.RandomAffine(0, translate=(0.05,0.05), scale=(0.95,1.05))]
        if normalize:
            t += [T.ToTensor(), T.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])]
        else:
            t += [T.ToTensor()]
        self.transform = T.Compose(t)
    def __call__(self, img): return self.transform(img)

class SimulateSpeckleNoise:
    def __init__(self, prob=0.3, intensity=(0.02, 0.08)):
        self.prob = prob; self.intensity = intensity
    def __call__(self, t):
        if random.random() > self.prob: return t
        return t + torch.randn_like(t) * random.uniform(*self.intensity)

def get_train_transforms(img_size=224, strength=0.5): return MedicalTransform(img_size, True, True, strength)
def get_val_transforms(img_size=224): return MedicalTransform(img_size, False, True)
