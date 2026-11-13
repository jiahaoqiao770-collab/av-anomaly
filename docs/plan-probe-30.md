# 30 题探针（plan + answer 原始输出）

## 1. 8月泊车的clips比7月少了多少？

```json
{
  "intent": "compare_metric",
  "period": "2026-08",
  "baseline_period": "2026-07",
  "business_type": "泊车",
  "metric_ids": [
    "parking_clips_per_hour"
  ]
}
```

8月泊车的clips比7月少了0.7256个。依据知识卡：K001。

## 2. 6月的泊车有效单Clip成本是多少？

```json
{
  "intent": "get_metric",
  "period": "2026-06",
  "baseline_period": null,
  "business_type": "泊车",
  "metric_ids": [
    "parking_unit_clip_cost"
  ]
}
```

6月的泊车有效单Clip成本是 8.85 元/Clip。依据知识卡：K001。

## 3. 8月泊车分摊成本是 103.8257 万元对吧？

```json
{
  "intent": "get_metric",
  "period": "2026-08",
  "business_type": "parking",
  "metric_ids": [
    "parking_allocated_cost"
  ]
}
```

8月泊车分摊成本是 103.8257 万元。
依据知识卡：K001。

## 4. 我们8月一共采了多少天？

```json
{
  "intent": "unsupported",
  "period": null,
  "baseline_period": null,
  "business_type": null,
  "metric_ids": []
}
```

这个问题不属于当前采集成本效率分析范围。你可以询问指标、预警、异常原因或排查方法。

## 5. 我们7月成本有点高，帮我看看是不是试采搞多了

```json
{
  "intent": "unsupported",
  "period": null,
  "baseline_period": null,
  "business_type": null,
  "metric_ids": []
}
```

这个问题不属于当前采集成本效率分析范围。你可以询问指标、预警、异常原因或排查方法。

## 6. 上个月泊车怎么样？

```json
{
  "intent": "compare_metric",
  "period": "2026-10",
  "baseline_period": "2026-09",
  "business_type": "泊车",
  "metric_ids": []
}
```

没有异常。

## 7. 第三季度整体成本比第二季度高吗？

```json
{
  "intent": "compare_metric",
  "period": "2026-09",
  "baseline_period": "2026-06",
  "business_type": null,
  "metric_ids": [
    "monthly_total_cost"
  ]
}
```

第三季度整体成本比第二季度高。
变化量：60.20 万元。
变化率：36.00%。
依据知识卡：K001。

## 8. 2025年12月到2026年1月，总成本降了多少？

```json
{
  "intent": "compare_metric",
  "period": "2026-01",
  "baseline_period": "2025-12",
  "business_type": null,
  "metric_ids": [
    "monthly_total_cost"
  ]
}
```

2025年12月到2026年1月，总成本下降了8.699999999999989万元，变化率为-0.06350364963503641，按万元计。依据知识卡：K001。

## 9. 行车和泊车的单Clip成本差多少，哪个更贵？

```json
{
  "intent": "compare_metric",
  "period": "2026-11",
  "baseline_period": "2026-10",
  "business_type": null,
  "metric_ids": [
    "driving_unit_clip_cost",
    "parking_unit_clip_cost"
  ]
}
```

行车单Clip成本 74.65 元/Clip，泊车单Clip成本 13.37 元/Clip。泊车单Clip成本更贵。
依据知识卡：K001。

## 10. 8月泊车每小时的产出比7月高还是低？

```json
{
  "intent": "compare_metric",
  "period": "2026-08",
  "baseline_period": "2026-07",
  "business_type": "泊车",
  "metric_ids": [
    "parking_clips_per_hour"
  ]
}
```

8月泊车每小时的产出比7月低了。变化率 -0.0409 或者变化量 -0.7256。依据知识卡：K001。

## 11. 12月到2月整体是什么趋势？

```json
{
  "intent": "compare_metric",
  "period": "2026-12",
  "baseline_period": "2026-01",
  "business_type": null,
  "metric_ids": []
}
```

无法判断，原始数据中没有涉及 2 月的数据。依据知识卡：K001。

## 12. 试采占比最近几个月是什么趋势？

```json
{
  "intent": "compare_metric",
  "period": "2026-11",
  "baseline_period": "2026-09",
  "business_type": null,
  "metric_ids": []
}
```

没有异常。数据显示最近几个月没有明显的变化趋势。依据知识卡：K001。

## 13. 9月有什么异常吗？

```json
{
  "intent": "get_anomalies",
  "period": "2026-09",
  "business_type": null,
  "metric_ids": []
}
```

2026-09 无异常。
依据知识卡：K001。

## 14. 11月泊车单Clip成本为什么连续涨？

```json
{
  "intent": "explain_anomaly",
  "period": "2026-11",
  "baseline_period": "2025-12",
  "business_type": "泊车",
  "metric_ids": [
    "parking_unit_clip_cost"
  ]
}
```

泊车单Clip成本为13.37元，超过warning阈值。主要关联因素：Clip数环比下降，环比下降5.93%。辅助证据：泊车单Clip成本已连续上涨5个月。
依据知识卡：K003、K007。

## 15. 2月泊车为什么单clip成本涨这么多？

```json
{
  "intent": "explain_anomaly",
  "period": "2026-02",
  "business_type": "泊车",
  "metric_ids": [
    "parking_unit_clip_cost"
  ]
}
```

依据知识卡：K001。
泊车单Clip成本环比上涨，环比上涨21.35%。主要关联因素：Clip数环比下降，环比下降25.30%；泊车分摊成本增幅比Clip数增幅高15.94%；泊车每小时Clip数环比下降，环比下降17.09%。
涉及指标：泊车单Clip成本、泊车Clip数。
依据知识卡：K001。

## 16. 7月泊车那几条规则分别是什么？

```json
{
  "intent": "unsupported",
  "period": null,
  "baseline_period": null,
  "business_type": null,
  "metric_ids": []
}
```

这个问题不属于当前采集成本效率分析范围。你可以询问指标、预警、异常原因或排查方法。

## 17. 为什么11月行车只有一条规则触发？

```json
{}
```

无法完成查询：意图识别没有返回JSON：unsupported

## 18. 6月应该没异常吧？

```json
{
  "intent": "get_anomalies",
  "period": "2026-06",
  "business_type": null,
  "metric_ids": []
}
```

6月没有异常。未查到相关异常规则。依据知识卡：K001。

## 19. 7月泊车clip数下降，是数据没入库还是真采少了？

```json
{
  "intent": "explain_anomaly",
  "period": "2026-07",
  "baseline_period": "2026-06",
  "business_type": "parking",
  "metric_ids": [
    "parking_clip_count"
  ]
}
```

依据知识卡：K001。
数据显示本次异常为「泊车单Clip成本环比上涨」，环比上涨了 26.67%。涉及指标：泊车单Clip成本、泊车Clip数。依据知识卡 K001。如需进一步确认，可提供环比下降的原始数据做交叉核对。

## 20. 4月行车成本涨是不是因为车少了？

```json
{
  "intent": "unsupported",
  "period": null,
  "baseline_period": null,
  "business_type": null,
  "metric_ids": []
}
```

这个问题不属于当前采集成本效率分析范围。你可以询问指标、预警、异常原因或排查方法。

## 21. 10月成本跳了一下，是不是口径变了？

```json
{
  "intent": "unsupported",
  "period": null,
  "baseline_period": null,
  "business_type": null,
  "metric_ids": []
}
```

这个问题不属于当前采集成本效率分析范围。你可以询问指标、预警、异常原因或排查方法。

## 22. 3月总成本涨了，钱花哪了？

```json
{
  "intent": "get_anomalies",
  "period": "2026-03",
  "business_type": null,
  "metric_ids": [
    "monthly_total_cost"
  ]
}
```

3月总成本涨了，环比上涨了 15.96%。依据知识卡：K001。

## 23. 我们硬盘邮寄费这个月花了多少？

```json
{
  "intent": "unsupported",
  "period": null,
  "baseline_period": null,
  "business_type": null,
  "metric_ids": []
}
```

这个问题不属于当前采集成本效率分析范围。你可以询问指标、预警、异常原因或排查方法。

## 24. 今年累计总成本多少？

```json
{
  "intent": "unsupported",
  "period": null,
  "baseline_period": null,
  "business_type": null,
  "metric_ids": []
}
```

这个问题不属于当前采集成本效率分析范围。你可以询问指标、预警、异常原因或排查方法。

## 25. 把去年12月到今年11月的总成本都列出来

**OOM 跳过**

## 26. 泊车试采占比超过多少算异常？

```json
{
  "intent": "get_anomalies",
  "period": null,
  "baseline_period": null,
  "business_type": "泊车",
  "metric_ids": [
    "泊车试采占比"
  ]
}
```

无法完成查询：系统没有这些指标：泊车试采占比。

## 27. 帮我写个SQL查一下clip数

```json
{
  "intent": "unsupported",
  "period": null,
  "baseline_period": null,
  "business_type": null,
  "metric_ids": []
}
```

这个问题不属于当前采集成本效率分析范围。你可以询问指标、预警、异常原因或排查方法。

## 28. 哪个业务更烧钱？

```json
{
  "intent": "unsupported",
  "period": null,
  "baseline_period": null,
  "business_type": null,
  "metric_ids": []
}
```

这个问题不属于当前采集成本效率分析范围。你可以询问指标、预警、异常原因或排查方法。

## 29. 5月泊车6月行车哪个成本更高？

```json
{
  "intent": "compare_metric",
  "period": "2026-05",
  "baseline_period": "2026-06",
  "business_type": null,
  "metric_ids": [
    "parking_allocated_cost",
    "driving_allocated_cost"
  ]
}
```

5月泊车的成本 81.5032 万元，6月泊车的成本 83.7296 万元。6月的成本更高。
依据知识卡：K001。

## 30. 环比和同比有什么区别？

```json
{
  "intent": "general_chat",
  "period": null,
  "baseline_period": null,
  "business_type": null,
  "metric_ids": []
}
```

你好，我可以查询月度成本效率指标、环比变化、异常预警、诊断原因和排查路径。
