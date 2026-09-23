# AI 学习路线：深度学习 → 大模型 → 强化学习（全资源清单）

> 覆盖四个阶段的 GitHub 仓库 + 网课，中英文都有。标 ⭐ 的是该阶段首选。
> 学习主线：深度学习基础 → Transformer/NLP → 大模型（训练/微调/部署）→ 强化学习 → RLHF（三者交汇点）

---

## 阶段 0：先修基础（Python + 数学，1~4 周）

### GitHub / 书
| 资源 | 说明 |
|---|---|
| [Python 官方教程](https://docs.python.org/zh-cn/3/tutorial/) 或 [廖雪峰 Python 教程](https://www.liaoxuefeng.com/wiki/1016959663602400) | 中文 Python 入门 |
| [Mathematics for Machine Learning](https://github.com/mml-book/mml-book.github.io) | 免费官方书，覆盖线代/概率/优化，专为 ML 设计 |
| [NumPy / Pandas 官方 10 分钟入门](https://pandas.pydata.org/docs/user_guide/10min.html) | 数据处理速成 |

### 网课 / 视频
| 资源 | 说明 |
|---|---|
| 3Blue1Brown《线性代数的本质》《微积分的本质》《神经网络》 | YouTube / B站官方号，建立几何直觉，强烈推荐 |
| 吴恩达 Machine Learning Specialization（Coursera） | 机器学习总览，可倍速过一遍 |

---

## 阶段 1：深度学习基础（2~3 个月）⭐ 核心

### GitHub / 书
| 资源 | 说明 |
|---|---|
| ⭐ [d2l-ai/d2l-zh《动手学深度学习》](https://github.com/d2l-ai/d2l-zh)（[在线阅读](https://zh.d2l.ai)） | 李沐著，中文首选一站式：理论 + 可运行 PyTorch 代码 + 习题，覆盖到注意力机制 |
| [d2l-ai/d2l-en](https://github.com/d2l-ai/d2l-en) | 英文原版 |
| [labmlai/annotated_deep_learning_paper_implementations](https://github.com/labmlai/annotated_deep_learning_paper_implementations) | 60+ 经典论文（Transformer、GAN、DDPG…）逐行注释实现，神级参考 |
| [datawhalechina/pumpkin-book《南瓜书》](https://github.com/datawhalechina/pumpkin-book) | 周志华《西瓜书》公式逐个推导，配合食用 |
| [scutan90/DeepLearning-500-questions](https://github.com/scutan90/DeepLearning-500-questions) | 中文深度学习面试 500 问，查漏补缺 |
| [microsoft/AI-For-Beginners](https://github.com/microsoft/AI-For-Beginners) | 微软官方 12 周课程，带实验 |
| [karpathy/micrograd](https://github.com/karpathy/micrograd) | 100 行实现反向传播引擎（配 Zero to Hero 第一课） |

### 网课
| 资源 | 说明 |
|---|---|
| ⭐ 李沐《动手学深度学习 v2》（B站搜"跟李沐学AI"） | 与 d2l-zh 书完全配套的视频 |
| ⭐ [Karpathy: Neural Networks Zero to Hero](https://karpathy.ai/zero-to-hero.html) | 从手写反向传播一路写到 GPT，英文原理解构最强课 |
| [fast.ai Practical Deep Learning](https://course.fast.ai)（[fastai/fastbook](https://github.com/fastai/fastbook)） | 自顶向下，先跑通再讲原理，工程味浓 |
| 吴恩达 Deep Learning Specialization（Coursera） | 经典四门课，体系全面 |
| [斯坦福 CS231n](https://cs231n.stanford.edu)（[课程笔记](https://cs231n.github.io)） | 计算机视觉视角的深度学习，笔记是经典 |
| 李宏毅机器学习（B站/YouTube，[课程主页](https://speech.ee.ntu.edu.tw/~hylee/index.php)） | 中文授课，2023+ 版本已直接覆盖 Transformer/LLM/RLHF |

---

## 阶段 2：NLP 与 Transformer（2~4 周）

| 资源 | 类型 | 说明 |
|---|---|---|
| [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) | 图解博客 | 最著名的 Transformer 图解 |
| [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) | 代码博客 | 哈佛 NLP：逐行实现 Transformer |
| Karpathy《Let's build GPT》（在 Zero to Hero 系列内） | 视频 | 从零写出一个 GPT，必看 |
| [斯坦福 CS224n](https://web.stanford.edu/class/cs224n/) | 公开课 | NLP with Deep Learning，经典系统课 |
| [斯坦福 CS25: Transformers United](https://web.stanford.edu/class/cs25/) | 公开课 | Transformer 前沿研讨会，嘉宾全是业界大牛 |
| [Hugging Face LLM Course 前半部分](https://huggingface.co/learn/llm-course) | 互动网课 | Transformers 库实战 |

---

## 阶段 3：大模型 LLM（2~4 个月）⭐ 核心

### GitHub / 书
| 资源 | 说明 |
|---|---|
| ⭐ [mlabonne/llm-course](https://github.com/mlabonne/llm-course) | 全网最著名 LLM 学习路线（40k+ star）：LLM Fundamentals → LLM Scientist（微调/DPO/量化/合并）→ LLM Engineer（RAG/部署），全程配 Colab 笔记本，有中文翻译版 |
| ⭐ [rasbt/LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) | Sebastian Raschka：用 PyTorch 从零实现 GPT（注意力/训练/微调/LoRA 全手写），同名纸质书 |
| [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) | 300 行极简 GPT 训练代码，读代码学预训练 |
| [datawhalechina/llm-cookbook](https://github.com/datawhalechina/llm-cookbook) | 吴恩达《ChatGPT Prompt Engineering》等系列课程的官方中文版 |
| [datawhalechina/self-llm](https://github.com/datawhalechina/self-llm) | 《开源大模型食用指南》：国产/开源模型（Qwen、LLaMA、ChatGLM…）环境搭建 + 微调实操 |
| [liguodongiot/llm-action](https://github.com/liguodongiot/llm-action) | 中文 LLM 训练/推理实战大全：分布式训练、显存优化、加速方案 |
| [microsoft/generative-ai-for-beginners](https://github.com/microsoft/generative-ai-for-beginners) | 微软官方 21 课生成式 AI 应用开发 |
| [Hannibal046/Awesome-LLM](https://github.com/Hannibal046/Awesome-LLM) | LLM 论文/模型/资源大清单 |
| [openai/openai-cookbook](https://github.com/openai/openai-cookbook) | OpenAI API 实战示例集 |
| [InternLM/Tutorial](https://github.com/InternLM/Tutorial) | 书生·浦语实战营：国产全链路（预训练→微调→部署）中文教程 |

### 网课
| 资源 | 说明 |
|---|---|
| ⭐ [斯坦福 CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/) | 2025 全套视频免费（Stanford YouTube）：tokenizer、架构、并行训练、Scaling Laws、数据、对齐、推理，研究生级硬核课，Karpathy 客座讲座 |
| [Hugging Face LLM Course](https://huggingface.co/learn/llm-course) | 免费互动：Transformers、微调、语义搜索、Agent |
| 李宏毅《生成式人工智能导论》（YouTube/B站） | 中文，LLM 原理与应用的最佳入门 |
| [DeepLearning.AI Short Courses](https://www.deeplearning.ai/short-courses/) | 吴恩达系列短课（1~2 小时/门）：Prompt Engineering、LangChain、RAG、Fine-tuning、RLHF 等几十门 |

---

## 阶段 4：强化学习（1~3 个月）⭐ 核心

### GitHub / 书
| 资源 | 说明 |
|---|---|
| ⭐ [datawhalechina/easy-rl《动手学强化学习》](https://github.com/datawhalechina/easy-rl)（[在线阅读](https://datawhalechina.github.io/easy-rl/)） | 中文首选"蘑菇书"：MDP → Q-learning → DQN → PPO，理论 + 代码 |
| [datawhalechina/joyrl](https://github.com/datawhalechina/joyrl) | 蘑菇书姊妹篇，纯代码驱动的强化学习教程 |
| [openai/spinning-up](https://github.com/openai/spinning-up)（[网站版](https://spinningup.openai.com)） | OpenAI 官方深度 RL 教程：策略梯度/PPO 等算法原理 + 高质量参考实现 |
| [Sutton & Barto《Reinforcement Learning: An Introduction》](http://incompleteideas.net/book/the-book-2nd.html) | RL 圣经，官方免费 PDF |
| [vwxyzjn/cleanrl](https://github.com/vwxyzjn/cleanrl) | 每个算法单文件实现，读代码学算法 |
| [DLR-RM/stable-baselines3](https://github.com/DLR-RM/stable-baselines3) | 生产级 RL 算法库（配合项目实践） |
| [Farama-Foundation/Gymnasium](https://github.com/Farama-Foundation/Gymnasium)（[文档](https://gymnasium.farama.org)） | 标准训练环境库（Gym 继任者） |
| [aikorea/awesome-rl](https://github.com/aikorea/awesome-rl) | RL 资源大清单 |

### 网课
| 资源 | 说明 |
|---|---|
| ⭐ David Silver《RL Course》(UCL/DeepMind, YouTube) | RL 理论经典中的经典，8 讲，配 Sutton & Barto |
| DeepMind x UCL《Reinforcement Learning Lecture Series》(YouTube) | 2021 更新版，内容更现代 |
| [伯克利 CS285: Deep RL](https://rail.eecs.berkeley.edu/deeprlcourse/) | Sergey Levine，深度 RL 最强学术课，YouTube 有全套录像 |
| [Hugging Face Deep RL Course](https://huggingface.co/learn/deep-rl-course) | 最友好的动手课：训练智能体 + 发布到 Hub，含 PPO/AILF 等 |
| [OpenAI Spinning Up](https://spinningup.openai.com) | 教程型"网课"，算法讲解 + 实现指南 |

---

## 阶段 5：RLHF / 对齐（RL × LLM 交汇点，2~4 周）

| 资源 | 类型 | 说明 |
|---|---|---|
| ⭐ [RLHF Book](https://rlhfbook.com) | 免费在线书 | Nathan Lambert 著：奖励建模、PPO/DPO、对齐全流程——把 RL 和 LLM 接起来的关键读物 |
| [huggingface/trl](https://github.com/huggingface/trl) | 库 + 文档 | Transformer 强化学习库：SFT/DPO/GRPO 等 Trainer，动手做对齐的标配 |
| [Hugging Face Blog: Illustrated RLHF](https://huggingface.co/blog/rlhf) | 图解博客 | ChatGPT 背后的 RLHF 图解 |
| DeepLearning.AI 短课《Reinforcement Learning from Human Feedback》 | 网课 | 1 小时级速览 RLHF 流程 |
| 论文：InstructGPT、DPO、GRPO | 论文 | 对齐三阶段（SFT→RM→PPO）与免 RM 方案 |

---

## 持续追踪 / 信息源

- [Hugging Face Daily Papers](https://huggingface.co/papers)：每日 AI 论文热榜
- [Papers with Code](https://paperswithcode.com)：论文 + 代码 + 榜单
- Datawhale 开源社区（[GitHub](https://github.com/datawhalechina)）：中文 AI 开源课程全家桶
- 知乎 / B站"跟李沐学AI"、"王树森《深度强化学习》"（中文视频补充）

---

## 只想要一门课？（单课全覆盖版）

- ⭐ **李宏毅《机器学习》2022 春季**（B站 108 集，[官方 Syllabus](https://speech.ee.ntu.edu.tw/~hylee/ml/2022-spring.php)）：一门课从 ML 基础 → 深度学习 → Transformer → 自监督(GPT/BERT) → 强化学习(PPO/Q-learning) → ChatGPT 原理(RLHF)，中文授课，覆盖度最全。续作是 2024/2025《生成式AI导论》（专讲 LLM 训练三阶段）
- ⭐ **[mlabonne/llm-course](https://github.com/mlabonne/llm-course)**（GitHub 英文）：单仓库路线图，Part1 深度学习/NLP 基础 → Part2 LLM 训练（含 DPO/RLHF/GRPO 强化学习对齐）→ Part3 工程部署，全程 Colab
- [Microsoft AI-For-Beginners](https://github.com/microsoft/AI-For-Beginners)：24 课带测验/lab，神经网络→Transformer→深度强化学习，LLM 较浅
- DeepMind × UCL《Advanced Deep Learning & RL》(YouTube)：前 8 讲 DL + 后 8 讲 RL，无 LLM

> 注意：单课方案必有取舍——RL 部分是"理解 RLHF 够用"级别。想 RL 更扎实，补 easy-rl 或 HF Deep RL Course。

## 懒人最小组合（时间紧只看这四个）

1. [d2l-zh《动手学深度学习》](https://zh.d2l.ai) — 打基础
2. [mlabonne/llm-course](https://github.com/mlabonne/llm-course) — LLM 全栈
3. [easy-rl《动手学强化学习》](https://datawhalechina.github.io/easy-rl/) — RL 基础（重点 PPO）
4. [RLHF Book](https://rlhfbook.com) — 三者交汇收尾
