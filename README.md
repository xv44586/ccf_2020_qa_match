# 比赛
贝壳找房-房产行业聊天问答匹配， 比赛地址[https://www.datafountain.cn/competitions/474/datasets](https://www.datafountain.cn/competitions/474/datasets)

# 简单说明
样本为一个问题多个回答，其中回答有的是针对问题的回答（1），也有不是的（0），其中回答是按顺序排列的。即：
query1: [(answer1, 0), (answer2, 1),...]
任务是对每个回答进行分类，判断是不是针对问题的回答。

# Baseline
## 思路一：
不考虑回答之间的顺序关系，将其拆为query-answer 对，然后进行判断。

![](./img/pair.png)

代码实现：[match_pair](https://github.com/xv44586/ccf_2020_qa_match/ccf_2020_qa_match_pair.py)

单模型提交f1:0.752

## 思路二：
考虑对话连贯性，同时考虑其完整性，将所有回答顺序拼接后再与问题拼接，组成query-answer1-answer2，然后针对每句回答进行分类。
即：将每句回答后面的[SEP] 作为最终的特征向量，然后去做二分类。
![](./img/point.png)

代码实现：[match_point](https://github.com/xv44586/ccf_2020_qa_match/ccf_2020_qa_match_point.py)

单模型提交f1: 0.75