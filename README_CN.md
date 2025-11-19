# DANT Curation App

## 安装

1. 使用新的conda环境：

```bash
conda create -n dant python=3.10.6
conda activate dant
```

2. 下载代码：https://github.com/jiumao2/DANT_UI
3. 进入代码目录，安装依赖：

```bash
cd DANT_UI
pip install -r requirements.txt
```
## 运行

1. 准备数据文件（建议使用SSD），文件夹应为以下格式：

```plaintext
├── DANT_Curation
    ├── ACG.npy
    ├── channel_locations.npy
    ├── Channels.npy
    ├── ClusterMatrixRaw.npy
    ├── IdxClustersRaw.npy
    ├── locations.npy
    ├── NumInChannels.npy
    ├── peth.npy
    ├── session_index.npy
    ├── SimilarityMatrix.npy
    ├── waveforms_corrected.npy
    ├── sort_index.npy
    ├── Meta.csv
    └── session_names.csv
└── PETH_Figs
    ├── Animal_Session_Ch1_Unit1.png
    ├── Animal_Session_Ch1_Unit2.png
    ├── ...
```

2. 进入代码目录，运行：

```bash
python myapp.py
```

3. 进入第一个界面，选择数据文件夹，然后点击“Load Data”加载数据（可能需要一段时间）。

![](./doc/UI_LoadData.png)

4. 加载完成后，自动进入第二个Splitting界面。检查的逻辑是先检查每一个Cluster内是否有错误聚类的结果，将其拆分。最后在第三个Merging界面，将相同的Cluster合并。

5. 点击Save按钮保存当前结果。当前结果会被存储在`DANT_Curation/IdxClusters.npy`文件中并在下一次加载数据时被自动读取。

## Splitting界面使用说明

![](./doc/UI_tab2.png)

1. 左上角显示一些元信息，并且包括右侧PETH图像选用的神经元（Unit1和Unit2）。

2. 左侧为Similarity Matrix热图，显示当前Cluster内各个神经元之间的相似度。该相似度由DANT得到，主要参考了Waveform的相似度，以及较小程度的ACG相似度。点击热图中的某个位置，可以选择对应的两个神经元，右侧会显示它们的PETH和Waveform对比图。点击对角线可以选择单独的某些神经元，再点击Waveform图上方的“Select units”按钮可以显示这些神经元与其他神经元的对比。

3. 中间包括了神经元的一些信息，比如在电极上的位置，Waveform，ACG以及PETH。

4. 左键热图的对角线可以选择准备进行拆分的神经元，会在上方Units to split中显示。点击Split按钮后这些神经元的Cluster ID会被修改为新的ID，而剩下的神经元Cluster ID不变。点击Undo按钮可以撤销上一次拆分操作（无法撤销上上次的操作）。

5. 点击Previous Cluster和Next Cluster按钮可以切换到上一个或下一个Cluster进行检查和拆分。

## Merging界面使用说明

![](./doc/UI_tab3.png)

1. 左上角显示一些元信息，并且包括右侧PETH图像选用的神经元（Unit1和Unit2）。Cluster1是当前选择的Cluster，Cluster2与Cluster1较为相似，可能需要合并的神经元。

2. 左侧为Similarity Matrix热图，显示当前Cluster1和Cluster2内各个神经元之间的相似度（同上）。改图上方的黑色线条表示Cluster1的Units。

3. 左下方为与Cluster1最相似的15个Cluster列表，点击某一行可以选择该Cluster作为Cluster2进行对比。Size表示该Cluster包含的Unit数量，Sim表示两个Cluster的Unit之间的平均相似度，Distance表示两个Cluster的中心之间的距离（由HDBSCAN将Unit排序得到）。

4. 右侧图像与Splitting界面类似，显示了Cluster1和Cluster2内神经元的Waveform，ACG以及PETH。Cluster1的Units由Copper colormap显示，Cluster2的Units由Winter colormap显示。

5. 点击Merge按钮后，Cluster2内的所有神经元的Cluster ID会被修改为Cluster1的ID。点击Undo按钮可以撤销上一次合并操作（无法撤销上上次的操作）。

6. 点击Previous Pair和Next Pair按钮可以切换到上一个或下一个Cluster对进行检查和合并。
