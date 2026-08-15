# 文献综述方法论参考

> 本文档整理公开的高质量文献综述框架，作为管线优化的设计依据。
> 来源：PRISMA、Cochrane、Arksey & O'Malley (scoping)、CASP、主题合成法等。

## 一、检索阶段的方法论

### 1.1 迭代检索 (Iterative Search)

**学者做法**：
- 第一轮：宽泛关键词，建立领域认知
- 第二轮：根据第一轮结果调整关键词（加入发现的术语、去除噪音词）
- 第三轮：针对性检索（补充遗漏子领域、追踪新方向）
- 停止条件：主题饱和（新检索不再产出新相关文献）

**技术实现要点**：
```
while not saturated:
    results = search(queries)
    new_findings = extract(results)
    queries = refine(queries, new_findings, gaps)
    if marginal_gain < threshold:
        break
```

### 1.2 引用图滚雪球 (Citation Snowballing)

**学者做法**：
- 向后滚雪球：对高相关论文，检查其参考文献列表（找经典源头）
- 向前滚雪球：对高相关论文，查找引用它的后续研究（找最新发展）
- 通常做 1-2 层深度，设纳入阈值（如被引 ≥ 5 次）

**技术实现要点**：
```
for paper in top_relevant:
    refs = get_references(paper.doi)   # OpenAlex/S2 API
    cits = get_citations(paper.doi)
    candidates.extend(filter_relevant(refs + cits))
```

### 1.3 查询多样性保证

**学者做法**：
- 确保关键词覆盖：方法维、应用维、理论维、实证维
- 避免同义词堆砌（浪费检索次数），追求语义正交
- 用领域标准术语（MeSH、ACM CCS 等）

**技术实现要点**：
```
query_dimensions = ["methods", "applications", "theory", "empirical", "survey"]
queries = ensure_coverage(planner_output, query_dimensions)
```

### 1.4 质量信号综合排序

**学者做法**（隐式）：
- 期刊声誉（对应我们的 SJR 分区）
- 被引次数（但要注意自引和年份偏倚）
- 发表年份（新≠好，但旧≠过时）
- 作者声誉（h-index、机构）
- 方法严谨性（从摘要判断：样本量、对照组、统计检验等）

**综合评分公式**：
```
quality = w1 * venue_quartile
        + w2 * log(1 + citations) / years_since_pub
        + w3 * method_rigor_score (from abstract)
        + w4 * author_h_index (if available)
```

---

## 二、筛选阶段的方法论

### 2.1 多阶段筛选

**PRISMA 标准**：
- 阶段 1：标题筛选（快速排除明显无关）
- 阶段 2：摘要筛选（详细判断相关性）
- 阶段 3：全文筛选（确认方法学质量）

**我们的映射**：
- 阶段 1：关键词匹配 + LLM 快速判断（当前 filter）
- 阶段 2：LLM 详细评分（当前 filter 的 relevance_score）
- 阶段 3：全文获取后的方法学评估（当前缺失，需要加）

### 2.2 双人独立筛选 + 一致性检验

**Cochrane 标准**：
- 两名研究者独立筛选，计算 Kappa 系数
- 分歧通过第三方仲裁解决

**技术映射**：
- 用两个不同 LLM（便宜模型 + 强模型）独立评分
- 分歧大的论文标记为"边界案例"
- 可以用第三个模型仲裁，或交给用户决定

### 2.3 排除日志

**PRISMA 要求**：
- 每篇被排除的论文必须记录排除原因
- 排除原因分类：主题不相关 / 方法不符 / 数据不全 / 质量不达标

**我们已实现**：`03_filter_log.json`

---

## 三、抽取阶段的方法论

### 3.1 结构化数据提取

**Cochrane 标准**：
- 标准化表格提取：研究特征、方法、参与 者、干预、结局
- 双人独立提取，核对一致性

**应该提取的字段**（超越当前的5字段）：
```
研究特征：
  - research_question
  - study_type (RCT/cohort/survey/theoretical/simulation)
  - sample_size / dataset
  - time_period
  - geographic_scope

方法：
  - methodology (详细描述)
  - theoretical_framework
  - key_variables / constructs
  - measurement_instruments

发现：
  - key_findings (按重要性排序)
  - effect_sizes / statistical_significance
  - effect_direction (positive/negative/mixed/null)

质量：
  - limitations (作者自述)
  - potential_biases (我们评估)
  - evidence_level (1-4)
```

### 3.2 证据等级评定

**标准分类**：
- Level 1a: 系统评价/Meta分析
- Level 1b: 单个 RCT
- Level 2a: 队列研究
- Level 2b: 病例对照研究
- Level 3: 病例报告/横断面
- Level 4: 专家意见/理论推导

**技术映射**：
- LLM 从摘要判断研究类型
- 按类型映射到证据等级
- 在综述中标注每条声明的证据等级

### 3.3 主题合成 (Thematic Synthesis)

**三步法**：
1. **编码**：对每篇文献逐行/逐段编码
2. **主题发展**：相似编码聚类成描述性主题
3. **分析性主题**：超越描述，形成解释框架

**技术映射**：
- 编码 → 我们的信息抽取（但需要更细粒度）
- 主题发展 → 我们的 organize（但需要渐进式，不是一次性）
- 分析性主题 → 我们的 synth（但需要更深的解释性）

### 3.4 矛盾检测与解释

**高质量综述的核心**：
- 主动寻找矛盾发现，而不是回避
- 对矛盾提供解释：方法差异？样本差异？时间差异？测量差异？
- 用"矛盾→解释"结构组织论述

**技术映射**：
- synth 已经提取 contradictions
- 但缺少对矛盾的结构化解释
- 需要新增：对每对矛盾，标注可能的原因（方法差异/样本差异/测量差异等）

---

## 四、合成阶段的方法论

### 4.1 研究比较矩阵

**学者做法**：
- 生成表格：研究 × 关键特征
- 帮助读者快速理解领域全貌
- 高亮相似性和差异性

**技术映射**：
```python
comparison_matrix = {
    "columns": ["Study", "Method", "Dataset", "Key Finding", "Limitation"],
    "rows": [extracted_paper.to_row() for paper in papers]
}
```

### 4.2 时间线分析

**学者做法**：
- 按时间排列研究，展示领域演进
- 识别范式转换点
- 标记里程碑论文

**技术映射**：
```
timeline = sorted(papers, key=lambda p: p.year)
milestones = [p for p in timeline if p.citation_count > threshold]
```

### 4.3 偏倚评估

**ROBINS-I 框架**（非随机研究）：
- 混杂偏倚
- 选择偏倚
- 测量偏倚
- 报告偏倚

**简化版 LLM 实现**：
- 对每篇论文问：样本是否有代表性？测量是否客观？是否选择性报告？
- 输出三维度偏倚风险：低/中/高

---

## 五、质量闭环

### 5.1 主题饱和度检测

**判断标准**：
- 最近 N 篇新纳入文献不再产出新的 key_findings
- 新文献的 research_question 与已有文献高度重叠
- 检索词的变化不再影响结果集

**技术映射**：
```python
def check_saturation(recent_papers, existing_findings):
    new_findings = [f for p in recent_papers for f in p.key_findings]
    overlap = compute_overlap(new_findings, existing_findings)
    return overlap > 0.8  # 80% 以上是重复发现 → 饱和
```

### 5.2 PRISMA 流程图自动生成

**应包含**：
- 初始检索结果数
- 去重后数量
- 标题筛选排除数
- 摘要筛选排除数
- 全文筛选排除数
- 最终纳入数
- 每阶段的排除原因分布

---

## 六、管线改进路线图

### 优先级 1：检索质量（当前最薄弱）

| 改进项 | 方法论依据 | 预期效果 |
|--------|-----------|---------|
| 迭代查询扩展 | 迭代检索 | 覆盖更多子领域 |
| 引用图滚雪球 | Citation snowballing | 补充经典+追踪前沿 |
| 查询多样性保证 | 查询多样性 | 避免单一视角 |
| 多信号排序增强 | 质量信号综合 | 提高候选池质量 |

### 优先级 2：抽取深度

| 改进项 | 方法论依据 | 预期效果 |
|--------|-----------|---------|
| 深度结构化抽取 | 结构化数据提取 | 支撑更深的综合分析 |
| 证据等级评定 | 证据分级 | 提升综述可信度 |
| 方法严谨性评分 | 偏倚评估 | 帮助读者判断证据强度 |
| 研究比较矩阵 | 研究比较矩阵 | 结构化呈现领域全貌 |

### 优先级 3：合成智能

| 改进项 | 方法论依据 | 预期效果 |
|--------|-----------|---------|
| 矛盾解释框架 | 矛盾检测与解释 | 深度批判性分析 |
| 时间线分析 | 时间线分析 | 展示领域演进 |
| 主题饱和度 | 主题饱和度 | 智能停止检索 |
| 渐进式主题发展 | 主题合成 | 更有层次的综述结构 |
