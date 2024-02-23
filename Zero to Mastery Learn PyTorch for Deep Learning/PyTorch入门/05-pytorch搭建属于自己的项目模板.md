# PyTorch 入门-构建属于你的项目框架 05

## 0.什么是模块?

模块化的概念是将代码转化为一系列不同的Python脚本，提供类似的功能。

例如，我们可以将笔记本代码从一系列单元格转化为以下Python文件：

* `data_setup.py` - 一个用于准备和下载数据（如果需要的话）的文件。
* `engine.py` - 包含各种训练函数的文件。
* `model_builder.py` 或 `model.py` - 用于创建PyTorch模型的文件。
* `train.py` - 一个用于利用所有其他文件并训练目标PyTorch模型的文件。
* `utils.py` - 一个专门用于实用工具函数的文件。

### 0.1 PyTorch在实际应用中

在您的实践中，您会发现许多基于PyTorch的机器学习项目的代码仓库都会提供如何运行PyTorch代码的说明，通常以Python脚本的形式呈现。

例如，您可能会被指示在终端/命令行中运行类似以下代码来训练模型：

```
python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-python-train-command-line-annotated.png" alt="command line call for training a PyTorch model with different hyperparameters" width=1000/> 

*在命令行上使用不同的超参数设置运行PyTorch的`train.py`脚本。*

在这种情况下，`train.py`是目标Python脚本，它可能包含用于训练PyTorch模型的函数。

而`--model`、`--batch_size`、`--lr`和`--num_epochs`被称为参数标志（argument flags）。

您可以将它们设置为任何您喜欢的值，如果它们与`train.py`兼容，它们将起作用；如果不兼容，将出现错误。

例如，假设我们想要在批量大小为32、学习率为0.001的情况下，对我们的TinyVGG模型进行10个时期的训练：

```
python train.py --model tinyvgg --batch_size 32 --lr 0.001 --num_epochs 10
```

您可以在`train.py`脚本中设置任意数量的这些参数标志，以满足您的需求。

PyTorch用于训练最先进的计算机视觉模型的博客文章采用了这种风格。

<img src="https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/05-training-sota-recipe.png" alt="PyTorch training script recipe for training state of the art computer vision models" width=800/>

### 0.2 我们的目标

在本节结束时，我们希望实现以下两点：

1. 使用命令行中的一行代码 `python train.py` 来训练我们在笔记本04（Food Vision Mini）中构建的模型的能力。
2. 具有可重复使用的Python脚本的目录结构，例如：

```
going_modular/
├── going_modular/
│   ├── data_setup.py
│   ├── engine.py
│   ├── model_builder.py
│   ├── train.py
│   └── utils.py
├── models/
│   ├── 05_going_modular_cell_mode_tinyvgg_model.pth
│   └── 05_going_modular_script_mode_tinyvgg_model.pth
└── data/
    └── pizza_steak_sushi/
        ├── train/
        │   ├── pizza/
        │   │   ├── image01.jpeg
        │   │   └── ...
        │   ├── steak/
        │   └── sushi/
        └── test/
            ├── pizza/
            ├── steak/
            └── sushi/
```

### 0.3 需要注意的事项

* **文档字符串** - 编写可重复使用和易于理解的代码很重要。考虑到这一点，我们将放入脚本中的每个函数/类都是根据Google的[Python文档字符串风格](https://google.github.io/styleguide/pyguide.html#383-functions-and-methods)创建的。
* **脚本顶部的导入** - 由于我们将要创建的所有Python脚本都可以被视为独立的小程序，因此所有脚本需要在脚本的开头导入它们所需的模块，例如：

<font color=Blue>In[*]</font>

```python
# Import modules required for train.py
import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms
```

## 1. 获取数据

<font color=Blue>In[*]</font>

```python 
import os
import requests
import zipfile
from pathlib import Path

# Setup path to data folder
data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

# If the image folder doesn't exist, download it and prepare it... 
if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True, exist_ok=True)
    
# Download pizza, steak, sushi data
with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    print("Downloading pizza, steak, sushi data...")
    f.write(request.content)

# Unzip pizza, steak, sushi data
with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
    print("Unzipping pizza, steak, sushi data...") 
    zip_ref.extractall(image_path)

# Remove zip file
os.remove(data_path / "pizza_steak_sushi.zip")
```

这将导致一个名为`data`的文件，其中包含另一个名为`pizza_steak_sushi`的目录，其中包含以标准图像分类格式存储的披萨、牛排和寿司的图像。

```
data/
└── pizza_steak_sushi/
    ├── train/
    │   ├── pizza/
    │   │   ├── train_image01.jpeg
    │   │   ├── test_image02.jpeg
    │   │   └── ...
    │   ├── steak/
    │   │   └── ...
    │   └── sushi/
    │       └── ...
    └── test/
        ├── pizza/
        │   ├── test_image01.jpeg
        │   └── test_image02.jpeg
        ├── steak/
        └── sushi/
```

## 2. 创建数据集和数据加载器（`data_setup.py`）

一旦我们有了数据，我们就可以将其转化为PyTorch的`Dataset`和`DataLoader`（一个用于训练数据，一个用于测试数据）。

我们将有用的`Dataset`和`DataLoader`创建代码转化为一个名为`create_dataloaders()` 的函数。

然后，我们可以使用以下命令将其写入文件：`%%writefile going_modular/data_setup.py`。

<font color=Blue>In[*]</font>

```py title="data_setup.py"
%%writefile going_modular/data_setup.py
"""
Contains functionality for creating PyTorch DataLoaders for 
image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str, 
    test_dir: str, 
    transform: transforms.Compose, 
    batch_size: int, 
    num_workers: int=NUM_WORKERS
):
  """Creates training and testing DataLoaders.

  Takes in a training directory and testing directory path and turns
  them into PyTorch Datasets and then into PyTorch DataLoaders.

  Args:
    train_dir: Path to training directory.
    test_dir: Path to testing directory.
    transform: torchvision transforms to perform on training and testing data.
    batch_size: Number of samples per batch in each of the DataLoaders.
    num_workers: An integer for number of workers per DataLoader.

  Returns:
    A tuple of (train_dataloader, test_dataloader, class_names).
    Where class_names is a list of the target classes.
    Example usage:
      train_dataloader, test_dataloader, class_names = \
        = create_dataloaders(train_dir=path/to/train_dir,
                             test_dir=path/to/test_dir,
                             transform=some_transform,
                             batch_size=32,
                             num_workers=4)
  """
  # Use ImageFolder to create dataset(s)
  train_data = datasets.ImageFolder(train_dir, transform=transform)
  test_data = datasets.ImageFolder(test_dir, transform=transform)

  # Get class names
  class_names = train_data.classes

  # Turn images into data loaders
  train_dataloader = DataLoader(
      train_data,
      batch_size=batch_size,
      shuffle=True,
      num_workers=num_workers,
      pin_memory=True,
  )
  test_dataloader = DataLoader(
      test_data,
      batch_size=batch_size,
      shuffle=False, # don't need to shuffle test data
      num_workers=num_workers,
      pin_memory=True,
  )

  return train_dataloader, test_dataloader, class_names
```

如果我们想要创建`DataLoader`，我们现在可以像这样使用`data_setup.py`中的函数：

<font color=Blue>In[*]</font>

```python
# Import data_setup.py
from going_modular import data_setup

# Create train/test dataloader and get class names as a list
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(...)
```

## 3. 创建模型（`model_builder.py`）

在过去的几个笔记本中（笔记本03和笔记本04），我们多次构建了TinyVGG模型。

因此，将模型放入其文件中以便反复重用是有道理的。

让我们将我们的`TinyVGG()`模型类放入一个脚本中，使用以下命令：`%%writefile going_modular/model_builder.py`：

<font color=Blue>In[*]</font>

```python title="model_builder.py"
%%writefile going_modular/model_builder.py
"""
Contains PyTorch model code to instantiate a TinyVGG model.
"""
import torch
from torch import nn 

class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/
  
  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
  def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(in_channels=input_shape, 
                    out_channels=hidden_units, 
                    kernel_size=3, 
                    stride=1, 
                    padding=0),  
          nn.ReLU(),
          nn.Conv2d(in_channels=hidden_units, 
                    out_channels=hidden_units,
                    kernel_size=3,
                    stride=1,
                    padding=0),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2,
                        stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=0),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          # Where did this in_features shape come from? 
          # It's because each layer of our network compresses and changes the shape of our inputs data.
          nn.Linear(in_features=hidden_units*13*13,
                    out_features=output_shape)
      )
    
  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
      # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion
```

现在，我们不再需要每次从头编写TinyVGG模型，可以使用以下方式导入它：

<font color=Blue>In[*]</font>

```python
import torch
# Import model_builder.py
from going_modular import model_builder
device = "cuda" if torch.cuda.is_available() else "cpu"

# Instantiate an instance of the model from the "model_builder.py" script
torch.manual_seed(42)
model = model_builder.TinyVGG(input_shape=3,
                              hidden_units=10, 
                              output_shape=len(class_names)).to(device)
```

## 4. 创建`train_step()` 和 `test_step()` 函数以及 `train()` 函数来组合它们

之前我们编写了几个训练函数：

1. `train_step()` - 接受一个模型、一个`DataLoader`、一个损失函数和一个优化器，并在`DataLoader`上训练模型。
2. `test_step()` - 接受一个模型、一个`DataLoader`和一个损失函数，并在`DataLoader`上评估模型。
3. `train()` - 为给定的时期数执行1和2，并返回一个结果字典。

由于这些函数将成为我们模型训练的“引擎”，我们可以将它们全部放入一个名为`engine.py`的Python脚本中，使用以下命令：`%%writefile going_modular/engine.py`：

<font color=Blue>In[*]</font>

```python title="engine.py"
%%writefile going_modular/engine.py
"""
Contains functions for training and testing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
  """Trains a PyTorch model for a single epoch.

  Turns a target PyTorch model to training mode and then
  runs through all of the required training steps (forward
  pass, loss calculation, optimizer step).

  Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:
    
    (0.1112, 0.8743)
  """
  # Put model in train mode
  model.train()
  
  # Setup train loss and train accuracy values
  train_loss, train_acc = 0, 0
  
  # Loop through data loader data batches
  for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      # 1. Forward pass
      y_pred = model(X)

      # 2. Calculate  and accumulate loss
      loss = loss_fn(y_pred, y)
      train_loss += loss.item() 

      # 3. Optimizer zero grad
      optimizer.zero_grad()

      # 4. Loss backward
      loss.backward()

      # 5. Optimizer step
      optimizer.step()

      # Calculate and accumulate accuracy metric across all batches
      y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
      train_acc += (y_pred_class == y).sum().item()/len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch 
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
  """Tests a PyTorch model for a single epoch.

  Turns a target PyTorch model to "eval" mode and then performs
  a forward pass on a testing dataset.

  Args:
    model: A PyTorch model to be tested.
    dataloader: A DataLoader instance for the model to be tested on.
    loss_fn: A PyTorch loss function to calculate loss on the test data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A tuple of testing loss and testing accuracy metrics.
    In the form (test_loss, test_accuracy). For example:
    
    (0.0223, 0.8985)
  """
  # Put model in eval mode
  model.eval() 
  
  # Setup test loss and test accuracy values
  test_loss, test_acc = 0, 0
  
  # Turn on inference context manager
  with torch.inference_mode():
      # Loop through DataLoader batches
      for batch, (X, y) in enumerate(dataloader):
          # Send data to target device
          X, y = X.to(device), y.to(device)
  
          # 1. Forward pass
          test_pred_logits = model(X)

          # 2. Calculate and accumulate loss
          loss = loss_fn(test_pred_logits, y)
          test_loss += loss.item()
          
          # Calculate and accumulate accuracy
          test_pred_labels = test_pred_logits.argmax(dim=1)
          test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
          
  # Adjust metrics to get average loss and accuracy per batch 
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc

def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          device: torch.device) -> Dict[str, List]:
  """Trains and tests a PyTorch model.

  Passes a target PyTorch models through train_step() and test_step()
  functions for a number of epochs, training and testing the model
  in the same epoch loop.

  Calculates, prints and stores evaluation metrics throughout.

  Args:
    model: A PyTorch model to be trained and tested.
    train_dataloader: A DataLoader instance for the model to be trained on.
    test_dataloader: A DataLoader instance for the model to be tested on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

  Returns:
    A dictionary of training and testing loss as well as training and
    testing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
                  train_acc: [...],
                  test_loss: [...],
                  test_acc: [...]} 
    For example if training for epochs=2: 
                 {train_loss: [2.0616, 1.0537],
                  train_acc: [0.3945, 0.3945],
                  test_loss: [1.2641, 1.5706],
                  test_acc: [0.3400, 0.2973]} 
  """
  # Create empty results dictionary
  results = {"train_loss": [],
      "train_acc": [],
      "test_loss": [],
      "test_acc": []
  }
  
  # Loop through training and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
      train_loss, train_acc = train_step(model=model,
                                          dataloader=train_dataloader,
                                          loss_fn=loss_fn,
                                          optimizer=optimizer,
                                          device=device)
      test_loss, test_acc = test_step(model=model,
          dataloader=test_dataloader,
          loss_fn=loss_fn,
          device=device)
      
      # Print out what's happening
      print(
          f"Epoch: {epoch+1} | "
          f"train_loss: {train_loss:.4f} | "
          f"train_acc: {train_acc:.4f} | "
          f"test_loss: {test_loss:.4f} | "
          f"test_acc: {test_acc:.4f}"
      )

      # Update results dictionary
      results["train_loss"].append(train_loss)
      results["train_acc"].append(train_acc)
      results["test_loss"].append(test_loss)
      results["test_acc"].append(test_acc)

  # Return the filled results at the end of the epochs
  return results
```

现在我们有了`engine.py`脚本，我们可以通过以下方式导入其中的函数：

<font color=Blue>In[*]</font>

```python
# Import engine.py
from going_modular import engine

# Use train() by calling it from engine.py
engine.train(...)
```

## 5. 创建保存模型的函数（`utils.py`）

通常，在模型训练过程中或训练后，您会希望保存模型。

由于我们在以前的笔记本中已经多次编写了保存模型的代码，因此将其转化为一个函数并保存到文件中是有道理的。

将辅助函数存储在名为`utils.py`（缩写为utilities）的文件中是常见的做法。

让我们将我们的`save_model()`函数保存到一个名为`utils.py`的文件中，使用以下命令：`%%writefile going_modular/utils.py`：

<font color=Blue>In[*]</font>

```python title="utils.py"
%%writefile going_modular/utils.py
"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path

def save_model(model: torch.nn.Module,
               target_dir: str,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    target_dir: A directory for saving the model to.
    model_name: A filename for the saved model. Should include
      either ".pth" or ".pt" as the file extension.
  
  Example usage:
    save_model(model=model_0,
               target_dir="models",
               model_name="05_going_modular_tingvgg_model.pth")
  """
  # Create target directory
  target_dir_path = Path(target_dir)
  target_dir_path.mkdir(parents=True,
                        exist_ok=True)
  
  # Create model save path
  assert model_name.endswith(".pth") or model_name.endswith(".pt"), "model_name should end with '.pt' or '.pth'"
  model_save_path = target_dir_path / model_name

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {model_save_path}")
  torch.save(obj=model.state_dict(),
             f=model_save_path)
```

现在，如果我们想要使用我们的`save_model()`函数，而不是再次编写它，我们可以通过以下方式导入并使用它：

<font color=Blue>In[*]</font>

```python
# Import utils.py
from going_modular import utils

# Save a model to file
save_model(model=...
           target_dir=...,
           model_name=...)
```

## 6. 训练、评估和保存模型（`train.py`）

如前所述，您通常会在PyTorch的代码库中找到将所有功能组合到一个名为`train.py`的文件中的情况。

这个文件基本上是在说“使用任何可用的数据来训练模型”。

在我们的`train.py`文件中，我们将组合我们创建的其他Python脚本的所有功能，并使用它来训练一个模型。

这样，我们可以在命令行上使用一行代码来训练一个PyTorch模型：

```
python train.py
```

要创建`train.py`，我们将按照以下步骤进行：

1. 导入各种依赖项，包括`torch`、`os`、`torchvision.transforms`以及`going_modular`目录中的所有脚本，包括`data_setup`、`engine`、`model_builder`、`utils`。

   * **注意：** 由于`train.py`将位于`going_modular`目录内，我们可以通过`import ...`而不是`from going_modular import ...`导入其他模块。

2. 设置各种超参数，如批量大小、时期数、学习率和隐藏单元数（这些可以在将来通过[Python的`argparse`](https://docs.python.org/3/library/argparse.html)设置）。
3. 设置训练和测试目录。
4. 设置与设备无关的代码。
5. 创建必要的数据转换。
6. 使用`data_setup.py`创建DataLoader。
7. 使用`model_builder.py`创建模型。
8. 设置损失函数和优化器。
9. 使用`engine.py`训练模型。
10. 使用`utils.py`保存模型。

我们可以使用以下命令从笔记本单元格中创建文件`%%writefile going_modular/train.py`：

<font color=Blue>In[*]</font>

```python title="train.py"
%%writefile going_modular/train.py
"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils

from torchvision import transforms

# Setup hyperparameters
NUM_EPOCHS = 5
BATCH_SIZE = 32
HIDDEN_UNITS = 10
LEARNING_RATE = 0.001

# Setup directories
train_dir = "data/pizza_steak_sushi/train"
test_dir = "data/pizza_steak_sushi/test"

# Setup target device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create transforms
data_transform = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])

# Create DataLoaders with help from data_setup.py
train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    transform=data_transform,
    batch_size=BATCH_SIZE
)

# Create model with help from model_builder.py
model = model_builder.TinyVGG(
    input_shape=3,
    hidden_units=HIDDEN_UNITS,
    output_shape=len(class_names)
).to(device)

# Set loss and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=LEARNING_RATE)

# Start training with help from engine.py
engine.train(model=model,
             train_dataloader=train_dataloader,
             test_dataloader=test_dataloader,
             loss_fn=loss_fn,
             optimizer=optimizer,
             epochs=NUM_EPOCHS,
             device=device)

# Save the model with help from utils.py
utils.save_model(model=model,
                 target_dir="models",
                 model_name="05_going_modular_script_mode_tinyvgg_model.pth")
```

太棒了！

现在，我们可以在命令行上运行以下命令来训练一个PyTorch模型：

```
python train.py
```

这将利用我们创建的所有其他代码脚本。

如果我们想要的话，我们可以调整我们的`train.py`文件，以使用Python的`argparse`模块的参数标志输入，这将允许我们提供不同的超参数设置，就像之前讨论的那样：

```
python train.py --model MODEL_NAME --batch_size BATCH_SIZE --lr LEARNING_RATE --num_epochs NUM_EPOCHS
```

## **练习：**

1. 将获取数据的代码（来自上面的第1节）转化为一个Python脚本，比如`get_data.py`。
   * 当您使用`python get_data.py`运行脚本时，它应该检查数据是否已经存在，如果存在则跳过下载。
   * 如果数据下载成功，您应该能够从`data`目录访问`pizza_steak_sushi`图像。

2. 使用[Python的`argparse`模块](https://docs.python.org/3/library/argparse.html)来能够为训练过程发送`train.py`的自定义超参数值。
   * 添加一个参数以使用不同的：
     * 训练/测试目录
     * 学习率
     * 批量大小
     * 要训练的时期数
     * TinyVGG模型中的隐藏单元数
   * 保持上述每个参数的默认值与它们已经是什么（与笔记本05中的一样）。
   * 例如，您应该能够运行类似以下的命令来训练一个学习率为0.003，批量大小为64的TinyVGG模型，训练20个时期：`python train.py --learning_rate 0.003 --batch_size 64 --num_epochs 20`。
   * **注意：** 由于`train.py`利用了我们在第05节创建的其他脚本，如`model_builder.py`、`utils.py`和`engine.py`，您需要确保它们也可供使用。您可以在课程GitHub上的[`going_modular`文件夹](https://github.com/mrdbourke/pytorch-deep-learning/tree/main/going_modular/going_modular)中找到它们。

3. 创建一个用于预测（比如`predict.py`）的脚本，给定一个带有保存模型的文件路径的目标图像。
   * 例如，您应该能够运行命令`python predict.py some_image.jpeg`，让经过训练的PyTorch模型对图像进行预测并返回其预测结果。
   * 要查看示例预测代码，请参阅笔记本04中的[在自定义图像上进行预测部分](https://www.learnpytorch.io/04_pytorch_custom_datasets/#113-putting-custom-image-prediction-together-building-a-function)。
   * 您可能还需要编写代码来加载已训练的模型。

## 扩展学习

* 要了解有关构建Python项目结构的更多信息，请查看Real Python的[Python应用程序布局指南](https://realpython.com/python-application-layouts/)。
* 关于如何为PyTorch代码设计样式的想法，可以查看Igor Susmelj的[PyTorch样式指南](https://github.com/IgorSusmelj/pytorch-styleguide#recommended-code-structure-for-training-your-model)（本章的大部分样式基于此指南以及各种类似的PyTorch存储库）。
* 对于一个示例的`train.py`脚本和训练最先进的图像分类模型的各种其他PyTorch脚本，请查看PyTorch团队的GitHub上的[`classification`存储库](https://github.com/pytorch/vision/tree/main/references/classification)。
