# Deployment Decision Matrix

| Option | Best fit | Advantages | Risks to validate |
|---|---|---|---|
| Trial or hosted API | Fast PoC, high model quality, elastic traffic | No model operations, quick iteration | Data egress, vendor availability, token cost, rate limits |
| llama.cpp local service | Data cannot leave the site, edge device, small traffic | Data locality, fixed model version, OpenAI-compatible integration | Model quality, hardware capacity, upgrade and monitoring effort |
| Production GPU serving | Higher concurrency and shared enterprise service | Throughput, batching and centralized operations | GPU procurement, scheduling, isolation, observability and cost |

## Decision sequence

1. Confirm whether customer data may leave the network.
2. Confirm peak concurrency, daily volume, context length, first-token target and full-answer target.
3. Establish a quality baseline on the real business question set.
4. Run the same question set on the candidate model and target hardware.
5. Compare quality, latency, throughput, memory, operational effort and total cost.
6. Present a PoC recommendation with explicit assumptions and a production validation plan.

The Colab or Apple Silicon measurements in this repository are local engineering evidence, not a production capacity promise. Re-run on the customer's target hardware before making a capacity or SLA commitment.
