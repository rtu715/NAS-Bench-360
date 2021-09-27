# pylint: disable=E1101,R,C
import os
import numpy as np
import shutil
import requests
import zipfile
from dataset import Shrec17, CacheNPY, ToMesh, ProjectOnSphere
from subprocess import check_output
import torch
import torchvision
import types
import importlib.machinery


class KeepName:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, file_name):
        return file_name, self.transform(file_name)


def main(log_dir, augmentation, dataset, batch_size, num_workers):
    print(check_output(["nodejs", "--version"]).decode("utf-8"))

    torch.backends.cudnn.benchmark = True

    # Increasing `repeat` will generate more cached files
    transform = torchvision.transforms.Compose([
        CacheNPY(prefix="b64_", repeat=augmentation, pick_randomly=False, transform=torchvision.transforms.Compose(
            [
                ToMesh(random_rotations=True, random_translation=0.1),
                ProjectOnSphere(bandwidth=64)
            ]
        )),
        lambda xs: torch.stack([torch.FloatTensor(x) for x in xs])
    ])
    transform = KeepName(transform)

    test_set = Shrec17("data", dataset, perturbed=True, download=True, transform=transform)

    loader = importlib.machinery.SourceFileLoader('model', os.path.join(log_dir, "model.py"))
    mod = types.ModuleType(loader.name)
    loader.exec_module(mod)

    model = mod.Model(55)
    model.cuda()

    model.load_state_dict(torch.load(os.path.join(log_dir, "state.pkl")))

    resdir = os.path.join(log_dir, dataset + "_perturbed")
    if os.path.isdir(resdir):
        shutil.rmtree(resdir)
    os.mkdir(resdir)

    predictions = []
    ids = []

    loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)

    for batch_idx, data in enumerate(loader):
        model.eval()

        if dataset != "test":
            data = data[0]

        file_names, data = data
        batch_size, rep = data.size()[:2]
        data = data.view(-1, *data.size()[2:])

        data = data.cuda()
        with torch.no_grad():
            pred = model(data).data
        pred = pred.view(batch_size, rep, -1)
        pred = pred.sum(1)

        predictions.append(pred.cpu().numpy())
        ids.extend([x.split("/")[-1].split(".")[0] for x in file_names])

        print("[{}/{}]      ".format(batch_idx, len(loader)))

    predictions = np.concatenate(predictions)

    predictions_class = np.argmax(predictions, axis=1)

    for i in range(len(ids)):
        if i % 100 == 0:
            print("{}/{}    ".format(i, len(ids)), end="\r")
        idfile = os.path.join(resdir, ids[i])

        retrieved = [(predictions[j, predictions_class[j]], ids[j]) for j in range(len(ids)) if predictions_class[j] == predictions_class[i]]
        retrieved = sorted(retrieved, reverse=True)
        retrieved = [i for _, i in retrieved]

        with open(idfile, "w") as f:
            f.write("\n".join(retrieved))

    url = "https://shapenet.cs.stanford.edu/shrec17/code/evaluator.zip"
    file_path = "evaluator.zip"

    r = requests.get(url, stream=True)
    with open(file_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=16 * 1024 ** 2):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    zip_ref = zipfile.ZipFile(file_path, 'r')
    zip_ref.extractall(".")
    zip_ref.close()

    print(check_output(["nodejs", "evaluate.js", os.path.join("..", log_dir) + "/"], cwd="evaluator").decode("utf-8"))
    shutil.copy2(os.path.join("evaluator", log_dir + ".summary.csv"), os.path.join(log_dir, "summary.csv"))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument("--augmentation", type=int, default=1,
                        help="Generate multiple image with random rotations and translations")
    parser.add_argument("--dataset", choices={"test", "val", "train"}, default="val")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=1)

    args = parser.parse_args()

    main(**args.__dict__)
