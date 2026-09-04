# Task 场景重复检索 MVP

这是一个 CPU-only 的 Task 级视频场景去重原型：对每个已切分的动作片段抽取 8 帧，用 CLIP ViT-B/32 生成视觉向量；同时对 Task JSON 中的 `scene + task_name + details` 生成文本向量。系统用 Faiss `IndexFlatIP` 做精确视觉余弦召回，再融合双向帧覆盖度和视频—文本跨模态匹配重排 Top-K。

## 分数定义

- `raw_cosine`：整个 Task 聚合向量的余弦相似度。
- `frame_coverage`：查询和候选各 8 个帧向量构成 `8×8` 余弦矩阵，计算查询→候选与候选→查询的逐帧最佳匹配均值。它准确说是“双向最佳帧匹配”，不检查帧的时序顺序，不等于动作时长覆盖率。
- `text_alignment`：查询携带 Task 描述时，它是“查询描述文本向量”与“候选 `scene+task_name+details` 文本向量”的余弦，并参与最终分数。未携带描述时，API 返回查询视频向量与候选文本向量的 CLIP 跨模态余弦作为诊断项，但它不参与当前 `video_only` 最终分数。
- `visual_similarity`：`0.65 * raw_cosine + 0.35 * frame_coverage`。
- `corpus_percentile`：候选整体视觉余弦在当前库的所有非自匹配对中所处百分位。
- `similarity_percent`：先用库内随机负对 p95 对整体视觉余弦做归一化，再融合帧覆盖度。未提供 Task 描述时权重为视觉显著性 70% + 帧覆盖 30%；提供描述时为视觉显著性 55% + 帧覆盖 25% + 文本显著性 20%。该分数只是库内归一化的 MVP 检索分，不是重复概率。
- `warning_level`：只有 `raw_cosine>=0.995 且 frame_coverage>=0.98` 才直接 `high`；库内归一化分较高或处于随机负对 p99 以上时为 `review`，其余为 `low`。生产化前必须用业务标注对学习概率和阈值。

## API

- `GET /`：视频地址 + Task 帧范围检索页，会展示并播放 Top-K 候选片段。
- `GET /health`：健康检查。
- `GET /stats`：索引统计。
- `POST /search/upload`：上传已截取的 Task 视频，可选表单字段 `description`。
- `POST /search/url`：传入裸 `oss://bucket/key` 或 OSS 签名 HTTPS URL；推荐使用 `start_frame/end_frame/fps/description`，帧区间为 `[start_frame, end_frame)`。为了向后兼容，仍支持 `start_seconds/end_seconds`。

## 2000 Task 人工盲标评测

- `GET /label`：人工盲标工作台。独立评测池为 2000 个 Task，其中 50 个作为参考资产、1800 个作为正式查询、150 个作为坏片替补。
- 50 个参考通过视觉向量 spherical k-means 选择各簇最接近中心的真实 Task，并约束不重复使用同一源视频。其余1950条按与50条参考的最大视觉余弦分为高/中/低三层，每层固定随机抽600条，避免评测集只有明显负样本或只有模型喜欢的近邻。
- 每个查询在建库时冻结模型 Top-5、分数和预警等级；盲标完成前不向标注人显示分数/排名，快速候选的卡片顺序也按查询稳定打乱。可随时浏览全部50个参考，避免 Top-5 漏召回时无法人工纠正。
- 人工标签为 `duplicate / review / novel / bad`；`duplicate/review` 必须选择对应参考资产。`bad` 会自动从150条保留集提升一条，保持有效目标为1800条。
- 标注存在 SQLite WAL 数据库，支持进度恢复、撤销和 CSV 导出。全部完成后 `/label/api/report` 解锁 Recall@1/5、high precision、novel high 误报率及人工标签×模型预警混淆矩阵。

## 统一多模态母测试集（小规模先导）

- `GET /study`：新的多模态共享母测试集入口，当前只开放“场景轮”。旧 `/label` 及其历史标注数据库保持不变。
- 母测试集包含100个参考资产、120个查询、每个查询5个固定候选，共600个`query-reference` Pair。以后动作轮、文本轮和整体轮复用相同`pair_id`，不重新抽样。
- 查询按高视觉高文本、高视觉低文本、低视觉高文本、低视觉低文本四个象限各抽30条；候选由视觉Top-2、文本Top-2和确定性随机负例组成，去重后固定顺序。
- 场景轮显示RGB视频、Task范围和库内已有的`scene/task_name/details`描述，但不返回或展示候选来源、向量分数及排名；描述仅作为人工辅助信息，不改变场景轮标签口径。
- 每个Pair标记`3同一物理工位 / 2布局明显相似 / 1同类环境 / 0环境不同 / 9无法判断`。一个查询的5对必须同时完成后才能提交。
- `GET /study/api/scene-report`在600对完成前隐藏模型指标；完成后给出分级NDCG@5、Top-1人工场景等级、等级≥2命中率和四象限分层结果。
- `GET /study/errors`：场景轮完成后的错题分析页。固定前80个查询为开发数据，只展示模型余弦Top-1低于人工最佳等级的排序错误；可对“同类异地、动作/工具干扰、人物遮挡、角度/光照”等根因进行归类。后40个查询不进入错题页，作为后续锁定测试数据。
- `GET /study/errors/safari`：Safari自带翻译专用版本。文档声明为英文以触发地址栏翻译，中文界面标记为`translate=no`，只有每个视频下的`scene/task_name/details`描述块标记为`lang=en translate=yes`。
- 数据设计、四轮标注口径和后续融合路线见[多模态Task去重MVP整体计划.md](./多模态Task去重MVP整体计划.md)。

如果上游 Task JSON 已有 `scene/task_name/details`，查询时应同时传入 canonical description，以启用文本—文本精排。仅传视频时，同一批头戴相机与相似环境会导致 CLIP 向量聚簇，只适合做候选召回，不足以独立完成细粒度工作语义判定。

开发环境默认只监听 `127.0.0.1:8000`。部署到测试服务器后，systemd 会将服务监听在 `0.0.0.0:8000`，可直接打开：

`http://112.74.108.93:8000/study`

若云安全组未放行 8000 端口，需先添加 TCP 入站规则；本地开发仍可建立 SSH 隧道：

```bash
ssh -L 8000:127.0.0.1:8000 root@<server-ip>
```

然后浏览器打开 `http://127.0.0.1:8000`。正式对外暴露前应增加 API Key/OIDC、请求限流和上传大小的网关层限制。

## 安全边界

源码和 systemd 单元中不保存 OSS AK/SK。MVP 的凭据仅放在 root 可读的 `/etc/task-scene-dedup.env`（`0600`），服务端只为输入和候选视频生成 15 分钟签名 URL。裸 `oss://` 只允许白名单 bucket，HTTPS 输入只允许 `*.aliyuncs.com`，防止任意 SSRF。生产环境应换成 ECS RAM Role，不应使用长期 AK/SK。

## 服务器资源策略

建库时通过 OSS 签名 URL + HTTP Range 读取 Task 范围，每个 Task 完成后立即删除临时帧。索引只保留 512 维视觉 Task 向量、8×512 帧向量、512 维文本向量、一张缩略图和元数据，1000 个 Task 不需要保存 1000 份原视频。
