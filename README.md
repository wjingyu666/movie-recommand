# movie-recommand
电子商务课程作业
eval_cold_start_topk.py:
从 train_data.csv 里找出冷启动用户：训练集中交互数 ≤ COLD_THRESHOLD

对这些用户模拟冷启动：不用个性化模型，只用“豆瓣评分 + 热度”的全局热门榜来推荐 Top-K

用 test_data.csv 给每个用户选 1 条正样本（默认选该用户测试集中最高评分那条）做 Leave-One-Out

输出冷启动用户的 HR@K / NDCG@K

同时给出**非冷启动用户（warm）**在同样热门策略下的结果做对比（体现冷启动更难）

你只要保证这四个文件路径正确即可运行：
train_data.csv、test_data.csv、movie_douban_info.csv（含 movieId,douban_rating,douban_votes）、以及这份脚本本身。


eval_components_topk.py:
下面给你一份组件级评估（Component-level Evaluation）完整可运行代码：在同一套 Leave-One-Out + 99 负采样候选集上，对比两种排序结果的 Top-K 指标：

Recall-only（召回排序）：只按你系统里的“豆瓣评分+热度”打分排序

Final（召回 + NeuMF重排）：对同一候选集用 NeuMF 预测评分排序

输出：两套方法各自的 HR@K / NDCG@K，以及提升幅度（Δ）。

你无需改推荐系统，只是新增这个评估脚本即可。


eval_topk.py:
下面给你一份可直接运行的 Top-K 推荐质量评估脚本（Leave-One-Out + 负采样），会输出 HR@K / NDCG@K（支持多个 K），并且不需要改你现有推荐系统：只读取你训练好的 neumf_model.h5 和 train_data.csv/test_data.csv。

不建议删掉 eval_topk.py。
正确做法是：两个都保留，各自承担不同评估角色。

下面我用「评估目的 → 评估层级 → 老师视角」给你讲清楚。

一句话结论版（可直接记）

eval_topk.py：回答
👉「这个推荐系统整体效果好不好？」

eval_components_topk.py：回答
👉「这个系统里哪个组件带来了提升？」

两者不是重复，而是互补。
