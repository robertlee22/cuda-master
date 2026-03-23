根据git日志查询结果：

**v1论文发表版本（2022年5月）：**
- 提交编号：1fcbe6f (完整: 1fcbe6f0d088d807ba585ddb850eb3497cd7b65b)
- 提交日期：2022-05-20
- 提交消息：First release
- 作者：tridpq@gmail.com

**v1的完整最终版本（最后一个v1 release）：**
- 版本标签：v1.0.9
- 提交编号：6d48e14 (完整: 6d48e14a6c2f551db96f0badc658a6279a929df3)
- 发布日期：2023-07-17
- 作者：tridpq@gmail.com

如果你想要的是论文发表时的"最初版本"，使用 `1fcbe6f`；
如果需要v1系列的"最完整/最稳定版本"，使用 `v1.0.9` 或 `6d48e14`。

你可以用这样的命令检出特定版本：
```bash
git checkout 1fcbe6f      # 论文发表版本
git checkout v1.0.9       # 或最后的v1版本
```



Kernel traits 分布式写法：
```   dim3 grid(launch_params.params.h, launch_params.params.b);
  kernel<<<grid, Kernel_traits::THREADS, smem_size, launch_params.stream>>>(
​        launch_params.params);
```

  

![image-20260320174706239](/Users/peets/Library/Application Support/typora-user-images/image-20260320174706239.png)





# v1工程阅读

Br和Bc通过 Kernel_traits 定义来生成。

看到了流水线实现方式，有种感想，和事件异步设计相似。流水线是少数重要事件，关注密集生产，事件异步更稀疏。



难点：pmax LSE、softmax 的计算没看懂，没看到 m、l 的更新。

v0.2.4 04c4c6106e2c055b4d13c0ebbd6f6a709fd0f5bc



给到一个block cta 的任务， Nxd 的 QKV，计算出 O出来。

