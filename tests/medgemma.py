

import os
from tqdm.asyncio import tqdm_asyncio  
import tqdm
import asyncio, re

# os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['MAX_PIXELS'] = '1003520'
os.environ['VIDEO_MAX_PIXELS'] = '50176'
os.environ['FPS_MAX_FRAMES'] = '12'

from swift.llm import VllmEngine, RequestConfig, InferRequest

def infer(req_iter, temperature=1.0, model_name='google/medgemma-4b-it', max_model_len=1000, use_hf=True, max_tokens=850):
    engine = VllmEngine(model_name, max_model_len=max_model_len, use_hf=use_hf, use_async_engine=True,
                         tensor_parallel_size=len(os.getenv("CUDA_VISIBLE_DEVICES", "").split(",")) if os.getenv("CUDA_VISIBLE_DEVICES") else 1,
                         limit_mm_per_prompt=json.loads(os.getenv("LIMIT_MM_PER_PROMPT", '{"image": 1, "video": 0}'))
                         )
    pattern = re.compile(r"(?:<\|?.+?\|?>|\[[^\]]+\])")

    async def _run():
        tasks = []
        for req in req_iter:
            t = req.pop("temperature", temperature)
            tasks.append(engine.infer_async(InferRequest(**req), 
                RequestConfig(max_tokens=max_tokens, temperature=t, top_p=0.9, logprobs=1, return_details=False)))
        return await tqdm_asyncio.gather(*tasks)
    resp = asyncio.run(_run())
    logprobs = [[d['logprob'] for d in r.choices[0].logprobs['content'] if not pattern.match(d['token'])] for r in resp]
    answers = [r.choices[0].message.content.strip() for r in resp]
    return answers, logprobs

if __name__ == '__main__':
    ##test code
    def req_iter(n=100):
        imgs = [
            'http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png',
            # '/home/sushant/D1/uncertainly/cluster_analysis.png'
        ]
        msgs = [
            ('<image><image> Compare i these', imgs),
            ('<image> What are  differences?', imgs),
            ('<image> what is in  image?', [imgs[0]])
        ]
        for i in range(n):
            print(f"I am loaded with index {i}")
            m, im = msgs[i % 3]
            yield {"messages":[{'role': 'user', 'content': m}], "images":im, "index": "su_"+ str(i)}

    answers, logprobs = infer(req_iter, temperature=0.7)
    print(answers)