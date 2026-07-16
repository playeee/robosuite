# 安装
**robosuite** 官方支持在 Python 3 上的 macOS 和 Linux 系统。它可以在带屏幕显示的情况下运行以进行可视化，或在无头模式下运行以进行模型训练，支持或不支持 GPU 均可。

基础安装需要 MuJoCo 物理引擎（使用 [mujoco](https://github.com/deepmind/mujoco)，请参阅链接以获取安装故障排除和进一步说明）和 [numpy](http://www.numpy.org/)。为避免干扰系统软件包，建议首先运行 `virtualenv -p python3 . && source bin/activate` 在虚拟环境下安装，或通过安装 [Anaconda](https://www.anaconda.com/) 并运行 `conda create -n robosuite python=3.10` 来设置 Conda 环境。

### 从 pip 安装

**注意**：对于希望使用最新代码并开发自定义功能的用户，建议从源代码安装。

1. 设置好 mujoco 后，可以使用以下命令安装 robosuite
   ```sh
   $ pip install robosuite
   ```

2. 使用以下命令测试您的安装
   ```sh
   $ python -m robosuite.demos.demo_random_action
   ```

   <div class="admonition warning">
   <p class="admonition-title">Mac 用户请注意！</p>

   希望使用默认 mjviewer 渲染器的 Mac 用户需要在 "python" 命令前加上 "mj" 前缀：`mjpython ...`
   </div>

### 从源代码安装

1. 克隆 robosuite 仓库
   ```sh 
   $ git clone https://github.com/ARISE-Initiative/robosuite.git
   $ cd robosuite
   ```

2. 使用以下命令安装基础依赖项
   ```sh
   $ pip3 install -r requirements.txt
   ```
   这也将把我们的库安装为可编辑软件包，这样本地更改将反映到其他地方，而无需重新安装该软件包。

3.（可选）我们还提供附加功能，例如 [OpenAI Gym](https://github.com/openai/gym) [接口](source/robosuite.wrappers)、由 [PyBullet](http://bulletphysics.org) 提供支持的[逆运动学控制器](source/robosuite.controllers)，以及使用 [SpaceMouse](https://www.3dconnexion.com/products/spacemouse.html) 和 [DualSense](https://www.playstation.com/en-us/accessories/dualsense-wireless-controller/) 设备的[遥操作](source/robosuite.devices)。要启用这些附加功能，请运行以下命令安装额外依赖项
   ```sh
   $ pip3 install -r requirements-extra.txt
   ```

4. 使用以下命令测试您的安装
   ```sh
   $ python robosuite/demos/demo_random_action.py
   ```

   <div class="admonition warning">
   <p class="admonition-title">Mac 用户请注意！</p>

   希望使用默认 mjviewer 渲染器的 Mac 用户需要在 "python" 命令前加上 "mj" 前缀：`mjpython ...`
   </div>

### 在 Windows 上安装

在 Windows 机器上安装 **robosuite** 时常遇到问题。**robosuite** 可以使用以下步骤在 Windows 上安装。

1. 请遵循[从 pip 安装](#install-from-pip)部分中的步骤 1，或[从源代码安装](#install-from-source)部分中的步骤 1 和 2。在此过程中，您可能会遇到一些错误。请参阅以下步骤了解如何修复这些错误。

2. 如果您遇到错误 `FileNotFoundError: [Errno 2] No such file or directory: 'C:\\tmp\\robosuite.log'`，请在 `C:\` 下创建一个名为 `tmp` 的目录。

3. 您还可能面临 `mujoco.dll not found` 的问题。如果您在 conda 环境中运行（强烈推荐），请转到安装软件包的位置（即 site-packages）。如果不确定 MuJoCo 软件包的位置，请打开一个新的 python shell 并运行以下命令。

   ```python
   import mujoco
   print(mujoco.__path__)
   ```

   如果 MuJoCo 软件包尚不存在，请通过运行以下命令安装它

   ```sh
   $ pip install mujoco
   ```

   在 MuJoCo 软件包内，应该有一个名为 `mujoco.dll` 的文件。如果您使用 pip 安装了 robosuite，请将此文件复制并粘贴到 `anaconda3\envs\{your env name}\Lib\site-packages\robosuite\utils `。如果您从源代码安装了 robosuite，请将此文件直接复制并粘贴到 `robosuite\utils` 中。 

4. 您还可能会遇到 `EGL` 问题。如果发生这种情况，请进入 `robosuite\utils\binding_utils.py`（在 site-packages 或克隆的仓库中，取决于您是从 pip 还是从源代码安装），并在第 43 行将 `"egl"` 更改为 `"wgl"`。应该如下所示：

   ```python
    if _SYSTEM == "Darwin":
        os.environ["MUJOCO_GL"] = "cgl"
    else:
        os.environ["MUJOCO_GL"] = "wgl"
   ```

5. 通过运行以下命令测试您的 **robosuite** 安装

   ```sh
   $ python robosuite/demos/demo_random_action.py
   ```
