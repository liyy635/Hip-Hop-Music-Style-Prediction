未完成，目前模型为能识别12种风格的识别器：
Afro，Boombap，Detroit，Drill，Garage，Gospel，Jazz，Jerk，Jersey-Club，Memphis，Rage，Trap(想给每种风格写一个简介，但gpt太垃圾了）
使用方法：先配置好环境。cd到Contrast_Training，终端输入python Train.py
里面有五种模式：
1.训练：我提供了一个Mymusic all的训练集，可以使用这个训练集进行训练，我耗时大约5-6h。
2.预测单曲：选择后，终端会先让你选择你训练模型用的训练集，直接输入就行；然后复制你想预测的单曲的地址就行
3.测试：Top-K指的是：当正确答案出现在可能性的前几名时，视为预测正确，因此Top-K越大正确率会越高，我建议设为2。测试集也可以选为Mymusic all，因为我训练的时候还没准备专门的测试集。
4.批量预测文件夹：把你要预测的单曲放在一个文件夹里，可以直接预测文件夹里所有单曲。
5.子类测试：这部分专门用来看某一类风格的预测精准度，是我自己用来优化部分风格表现的参照
接下来的目标：
1.训练逻辑改为：把Mymusic all分为9:1或8：2的训练集：测试集，再进行训练与测试
2.加入在训练本轮模型之前删除的Cloud，Emo，Lo-fi，New-Wave，Mumble，这部分的表现会差一点，先不考虑了
3.在日后继续添加风格种类：Baton-Rough，G-Funk，House，Hyperpop，Murder，New-Glo，Plugg，Ratchet，Religia，Melodic Rap(这部分范围很广），考虑加入UK-Drill和UK-Garage