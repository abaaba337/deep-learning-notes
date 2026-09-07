"""Local LLaVA/AWQ quantization and reproducible benchmark inference."""
from pathlib import Path
import argparse
import json

ROOT = Path(__file__).resolve().parents[3]


def new_output(path):
    if path.exists():
        raise FileExistsError(f'Choose a new output path; refusing to overwrite {path}')
    path.parent.mkdir(parents=True,exist_ok=True)


def quantize(args):
    import torch
    from transformers import AutoTokenizer, AutoProcessor
    if not torch.cuda.is_available():
        raise RuntimeError('This historical AutoAWQ experiment requires CUDA; see README')
    texts = [json.loads(line)['text'] for line in args.calibration.read_text(encoding='utf-8').splitlines() if line.strip()]
    if not texts or not all(isinstance(t,str) and t.strip() for t in texts):
        raise ValueError('Calibration JSONL needs nonempty text fields')
    new_output(args.output)
    from importlib.metadata import distribution
    from patch_autoawq import patched
    package = distribution('autoawq')
    source = package.locate_file('awq/quantize/quantizer.py').read_text(encoding='utf-8')
    if package.version != '0.2.7.post3' or patched(source) != source:
        raise RuntimeError('Run patch_autoawq.py in the pinned environment before quantization')
    tokenizer = AutoTokenizer.from_pretrained(args.model,trust_remote_code=False)
    processor = AutoProcessor.from_pretrained(args.model,trust_remote_code=False)
    # Bypass the old auto-loader's unsupported use_cache kwarg for vision models.
    from awq.models.auto import AWQ_CAUSAL_LM_MODEL_MAP
    from transformers import AutoConfig
    kind = AutoConfig.from_pretrained(args.model).model_type
    if kind not in ('llava','llava_next'):
        raise ValueError('This practice supports llava and llava_next checkpoints')
    wrapper = AWQ_CAUSAL_LM_MODEL_MAP[kind]
    model = wrapper.from_pretrained(str(args.model),model_type=kind,trust_remote_code=False,safetensors=True)
    model.quantize(tokenizer,calib_data=texts,quant_config={
        'zero_point':True,'q_group_size':128,'w_bit':4,'version':'GEMM'},
        n_parallel_calib_samples=1,max_calib_samples=args.samples,max_calib_seq_len=512)
    model.save_quantized(str(args.output))
    tokenizer.save_pretrained(args.output)
    processor.save_pretrained(args.output)
    (args.output/'experiment.json').write_text(json.dumps({
        'source_model':str(args.model),'calibration':str(args.calibration),'samples':args.samples,
        'note':'Text calibration of language layers; vision tower remains floating point.'},indent=2),encoding='utf-8')


def infer(args):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForVision2Seq
    questions = json.loads(args.questions.read_text(encoding='utf-8'))
    selected = list(questions.items())[:args.limit] if args.limit else list(questions.items())
    for key, item in selected:
        if not key.isdecimal() or not isinstance(item.get('Rephrased Question'),str):
            raise ValueError('Expected numeric IDs and Rephrased Question fields')
        if not (args.images/f'{key}.jpg').is_file():
            raise FileNotFoundError(args.images/f'{key}.jpg')
    if not selected:
        raise ValueError('No benchmark questions')
    new_output(args.output)
    processor = AutoProcessor.from_pretrained(args.model,trust_remote_code=False)
    processor.tokenizer.padding_side = 'left'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = torch.float16 if device=='cuda' else torch.float32
    if args.awq:
        from awq import AutoAWQForCausalLM
        model = AutoAWQForCausalLM.from_quantized(str(args.model),fuse_layers=False,trust_remote_code=False).model
    else:
        model = AutoModelForVision2Seq.from_pretrained(args.model,torch_dtype=dtype,trust_remote_code=False).to(device)
    model.eval()
    results={}
    with torch.inference_mode():
        for key,item in selected:
            prompt = processor.apply_chat_template([{'role':'user','content':[
                {'type':'image'},{'type':'text','text':item['Rephrased Question']}]}],add_generation_prompt=True)
            with Image.open(args.images/f'{key}.jpg') as image:
                inputs = processor(images=image.convert('RGB'),text=prompt,return_tensors='pt').to(model.device)
            if 'pixel_values' in inputs:
                inputs['pixel_values'] = inputs['pixel_values'].to(dtype=model.dtype)
            ids = model.generate(**inputs,max_new_tokens=args.max_new_tokens,do_sample=False)
            answer = processor.batch_decode(ids[:,inputs['input_ids'].shape[1]:],skip_special_tokens=True)[0]
            results[key] = dict(item,ans={args.label:{'text':answer}})
            temp = args.output.with_suffix('.tmp')
            temp.write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf-8')
            temp.replace(args.output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    q=sub.add_parser('quantize'); q.add_argument('--model',type=Path,required=True)
    q.add_argument('--calibration',type=Path,required=True,help='UTF-8 JSONL with text fields')
    q.add_argument('--samples',type=int,default=128)
    q.add_argument('--output',type=Path,default=ROOT/'outputs/mllm-compression-safety/llava-awq')
    i=sub.add_parser('infer');i.add_argument('--model',type=Path,required=True)
    i.add_argument('--questions',type=Path,required=True);i.add_argument('--images',type=Path,required=True)
    i.add_argument('--label',required=True);i.add_argument('--awq',action='store_true')
    i.add_argument('--limit',type=int,default=0);i.add_argument('--max-new-tokens',type=int,default=300)
    i.add_argument('--output',type=Path,default=ROOT/'outputs/mllm-compression-safety/answers.json')
    args=p.parse_args()
    if not args.model.is_dir():p.error('--model must be a local model directory')
    if args.command=='quantize':
        if args.samples<1:p.error('--samples must be positive')
        quantize(args)
    else:
        if args.limit<0 or args.max_new_tokens<1:p.error('Invalid generation limits')
        infer(args)


if __name__=='__main__':
    main()
