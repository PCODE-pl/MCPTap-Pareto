# ROUTERS

This document lists providers whose official documentation indicates that the
main API product forwards requests to external model providers or their
inference endpoints. Here, “external infrastructure” means infrastructure of a
named upstream or an undisclosed partner, rather than simply a model created by
another company. It excludes platforms with their own managed/serverless
inference and cases where public material does not establish the dominant
serving path.

Checked: 2026-08-18.

- **ai-router** — Its documentation calls the service an AI API gateway for unified model access and explicitly describes routing and failover; it does not describe a proprietary model-serving layer.[1]
- **aihubmix** — The service describes itself as an authorized aggregation platform for models from Microsoft Azure, AWS, GCP, Alibaba Cloud, and Baidu Cloud. Its Google Cloud control cluster does not make those offered third-party model services its own inference stack.[2]
- **anyapi** — One endpoint exposes hundreds of models from OpenAI, Anthropic, Google, Meta, and others; automatic routing handles fallback and cost/availability optimization.[3]
- **auriko** — Described as an LLM routing layer, it sends requests between providers, supports BYOK, and reports upstream-provider errors.[4]
- **cloudflare-ai-gateway** — The gateway controls, caches, retries, and fails over calls to providers such as OpenAI, Anthropic, and Google. This applies to AI Gateway, not Workers AI, which also has Cloudflare-hosted models.[5]
- **cortecs** — The model API exposes a `providers` field with independent hosts such as Tensorix, Berget, and AKI, showing selection of an external inference host.[6]
- **crossmodel** — The documentation defines the product as a multi-provider API gateway providing one interface for multiple model providers.[7]
- **edenai** — Eden AI is a unified gateway to 500+ models from 50+ providers, with routing/fallback and billing based on underlying-provider prices.[8]
- **empiriolabs** — Its aggregated endpoint routes to models from independent companies including Alibaba, Amazon, Google, DeepSeek, and Mistral, and the documentation names 34 providers. The company also offers GPU Cloud; this finding applies to its multi-provider API.[9]
- **fastrouter** — The LLM gateway routes prompts to 100+ models, advertises dynamic routing, failover, and BYOK/hosted keys, and labels catalog models by their originating providers.[10]
- **gitlab** — GitLab Duo defaults are explicitly identified as Claude on Vertex and Codestral on Fireworks, placing their inference on external model platforms rather than GitLab.[28]
- **impossibl** — This gateway exposes OpenAI, Anthropic, Google, xAI, and other models through `provider/model` IDs, charges provider prices, and supports provider-billed BYOK.[11]
- **kenari** — Its documentation explicitly says that the gateway routes to many model providers and forwards BYOK traffic without a Kenari token charge.[12]
- **kilo** — The Gateway product advertises routing to frontier paid models, zero inference markup, and personal keys; it is a provider-selection/intermediation layer rather than the host of those models.[13]
- **llmgateway** — The open-source gateway sits between an application and providers such as OpenAI, Anthropic, and Google AI Studio, selects providers, performs failover, and lets BYOK bypass its billing.[14]
- **merge-gateway** — It explicitly routes to OpenAI, Anthropic, Google, and AWS Bedrock through one endpoint with failover.[15]
- **model-oracle-ai** — The service selects an underlying model and provider behind its endpoint, supports fallback across linked providers, and lists OpenAI, Anthropic, SiliconFlow, StreamLake, and Bedrock.[16]
- **modelis** — It is an independent gateway to models from 11 labs; `modelis-auto` selects a model per request, and the company disclaims affiliation with those providers.[17]
- **nano-gpt** — Documentation describes auto-routing, pinned-provider surcharges, BYOK, and constraints caused by upstream-provider capabilities.[18]
- **neon** — Neon explicitly states that its AI Gateway is backed by Databricks Foundation Model APIs, so Neon does not perform the model inference itself.[19]
- **ofox** — OfoxAI aggregates 100+ models from leading providers, offers provider routing, and passes through model list prices without a platform markup.[20]
- **openrouter** — Its unified API for hundreds of models provides routing, fallbacks, and cost tracking across providers.[21]
- **orcarouter** — Documentation explicitly describes routing from one endpoint to OpenAI, Anthropic, Google, DeepSeek, xAI, Alibaba, Moonshot, MiniMax, and other providers at provider cost.[22]
- **requesty** — Requesty routes to 668+ models from 31 providers, passes through upstream prices, and identifies models available through third-party hosts.[23]
- **unorouter** — This OpenAI-compatible gateway offers one key for 200+ models across OpenAI, Anthropic, Google, DeepSeek, Moonshot, Zhipu, Qwen, and others.[24]
- **vercel** — AI Gateway is a unified endpoint for hundreds of models, with provider-level fallbacks, allowlists, and BYOK; the documentation also distinguishes each AI provider’s terms.[29]
- **vivgrid** — For global models, the documentation states that the model host remains the provider’s Global Standard endpoint; VivGrid can only provide regional acceleration/routing.[25]
- **xpersona** — The API offers genuine GPT, Claude, and Gemini through one endpoint and native routing. These closed models require an external host or lab, although the documentation does not disclose the downstream for every route.[26]
- **zenmux** — ZenMux says that its models are sourced from official providers including Anthropic, OpenAI, Google, Alibaba, ByteDance, Zhipu, DeepSeek, and Moonshot, and that it provides model routing.[27]

## Sources

[1] https://api.ai-router.dev/llms.txt — AI-ROUTER documentation
[2] https://docs.aihubmix.com/llms.txt — AIHubMix documentation
[3] https://docs.anyapi.ai — AnyAPI documentation
[4] https://docs.auriko.ai — Auriko documentation
[5] https://developers.cloudflare.com/ai-gateway — Cloudflare AI Gateway documentation
[6] https://api.cortecs.ai/v1/models — Cortecs model API
[7] https://www.crossmodel.ai/docs — CrossModel documentation
[8] https://docs.edenai.co/llms.txt — Eden AI documentation
[9] https://docs.empiriolabs.ai/llms.txt — EmpirioLabs documentation
[10] https://fastrouter.ai/llms.txt — FastRouter documentation
[11] https://impossibl.com/docs/models — Impossibl model documentation
[12] https://kenari.id/llms.txt — Kenari documentation
[13] https://kilo.ai — Kilo Gateway website
[14] https://llmgateway.io/docs — LLM Gateway documentation
[15] https://docs.merge.dev/merge-gateway — Merge Gateway documentation
[16] https://modeloracle.com/llms.txt — Model Oracle AI documentation
[17] https://modelishub.com/pricing — Modelis pricing
[18] https://nano-gpt.com/llms.txt — NanoGPT documentation
[19] https://neon.com/docs — Neon documentation
[20] https://ofox.ai/llms.txt — OfoxAI documentation
[21] https://openrouter.ai/llms.txt — OpenRouter documentation
[22] https://docs.orcarouter.ai — OrcaRouter documentation
[23] https://requesty.ai/solution/llm-routing/models — Requesty model routing
[24] https://unorouter.com/llms.txt — UnoRouter documentation
[25] https://docs.vivgrid.com/models — Vivgrid models documentation
[26] https://www.xpersona.co/docs — Xpersona documentation
[27] https://zenmux.ai/llms.txt — ZenMux documentation
[28] https://docs.gitlab.com/user/gitlab_duo/model_selection — GitLab Duo AI models
[29] https://vercel.com/docs/ai-gateway — Vercel AI Gateway documentation
