

import os
from tqdm.asyncio import tqdm_asyncio  
import tqdm
import asyncio, re
import json
from swift.llm import VllmEngine, RequestConfig, InferRequest


# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['MAX_PIXELS'] = '1003520'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'

def infer_batched(
    reqs,
    model_name='google/medgemma-4b-it',
    max_model_len=1000,
    use_hf=True,
    max_tokens=850,
    temperature=1.0,
    batch_size=1000,
):
    # Initialize engine once
    engine = VllmEngine(
        model_name,
        max_model_len=max_model_len,
        use_hf=use_hf,
        use_async_engine=True,
        tensor_parallel_size=len(os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")) if os.getenv("CUDA_VISIBLE_DEVICES") else 1,
        limit_mm_per_prompt=json.loads(os.getenv("LIMIT_MM_PER_PROMPT", '{"image":1,"video":0}')),
    )

    pat = re.compile(r"(?:<\|?.+?\|?>|\[[^\]]+\])")
    answers, logprobs = [], []

    n = len(reqs)
    desc = f"Inference (batch={batch_size})"

    # >>> Create ONE event loop and reuse it (key change)
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        with tqdm.tqdm(total=n, desc=desc, unit="req") as pbar:
            for i in range(0, n, batch_size):
                j = min(i + batch_size, n)
                batch_reqs = []
                batch_cfgs = []

                for r in reqs[i:j]:
                    local = dict(r)
                    t = local.pop("temperature", temperature)
                    batch_reqs.append(InferRequest(**local))
                    batch_cfgs.append(RequestConfig(
                        max_tokens=max_tokens,
                        temperature=t,
                        top_p=0.9,
                        logprobs=1,
                        return_details=False,
                    ))

                async def _run():
                    # tiny robustness: don't fail whole batch on one error
                    return await tqdm_asyncio.gather(
                        *[engine.infer_async(req, cfg) for req, cfg in zip(batch_reqs, batch_cfgs)]
                    )

                # >>> Use the persistent loop, NOT asyncio.run (key change)
                resp = loop.run_until_complete(_run())

                for r in resp:
                    if isinstance(r, Exception):
                        answers.append(None)
                        logprobs.append(None)
                        continue
                    answers.append((r.choices[0].message.content or "").strip())
                    lp = [
                        d["logprob"]
                        for d in (r.choices[0].logprobs.get("content") or [])
                        if not pat.match(d["token"])
                    ] if getattr(r.choices[0], "logprobs", None) else None
                    logprobs.append(lp)

                pbar.update(len(batch_reqs))
    finally:
        # Close the loop only when *all* batches are done
        try:
            loop.stop()
        except Exception:
            pass
        loop.close()

    return answers, logprobs
