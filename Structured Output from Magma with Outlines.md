I would like to refer my earlier posted issue here: [Evaluation and Finetuning Scripts](https://github.com/microsoft/Magma/issues/78). I am getting the same error as discussed above in this issue. I am explaining how I got it in the post below.

In the context of the earlier issue I referenced above, I was trying to get structured output from the magma model. Specifically, I wanted the output in the following JSON format:

```
{"ACTION": "One of the following UI actions - CLICK, TYPE, or SELECT",
 "MARK": "A numeric id, e.g., 5, - this refers to the id of the SoM marker for the UI element on which action is to be taken",
 "VALUE": "A string for the value of the action if it is a TYPE action, else None",
 "COORDINATES": "location of the UI element on which action is to be taken, normalized by the image dimensions, e.g., (0.83, 0.41)"}
```

Initially, I tried changing the prompts by adding the output template given above to them. Even then, the magma model gives only the `coordinates` and `mark` values as the outputs. Adding the output format given above to the prompts did not make any difference.

Next, I followed this blog: [Structured Generation from Images or Documents Using Vision Language Models
](https://huggingface.co/learn/cookbook/structured_generation_vision_language_models) get structured output from the magma model. The steps in this blog are not directly applicable to the magma model, so we modified the code so as to suit the magma model.

This is the main snippet from the blog:
```
def get_model_and_processor_class(model_name: str):
    model = AutoModelForImageTextToText.from_pretrained(model_name)
    processor = AutoProcessor.from_pretrained(model_name)
    classes = model.__class__, processor.__class__
    del model, processor
    return classes


model_class, processor_class = get_model_and_processor_class(model_name)

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

model = transformers_vision(
    model_name,
    model_class=model_class,
    device=device,
    model_kwargs={"torch_dtype": torch.bfloat16, "device_map": "auto"},
    processor_kwargs={"device": device},
    processor_class=processor_class,
)
```

This does not work with magma. So, after some experimentation, I replace this with the following:
```
from outlines.models.transformers_vision import transformers_vision, TransformersVision

model_name = "microsoft/Magma-8B"
dtype = torch.bfloat16

model = AutoModelForCausalLM.from_pretrained("microsoft/Magma-8B", trust_remote_code=True, torch_dtype=dtype)
processor = AutoProcessor.from_pretrained("microsoft/Magma-8B", trust_remote_code=True)
model.to("cuda:0")

if torch.cuda.is_available():
    device = "cuda:0"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

outlines_model = TransformersVision(model, processor.tokenizer, processor)
```

For generating the structured output, we created the following class, and made the model: 
```
class Magma_Structured_Output(BaseModel):
    action: str = Field(..., description="One of the actions: CLICK, TYPE ,SELECT")
    coordinates: List[float] = Field(..., description="Coordinates of the selected element of the screen")
    mark: int = Field(..., description="SoM marking")
    value: str = Field(..., description="Value for the action")

structured_generator = outlines.generate.json(outlines_model, Magma_Structured_Output)
```

Now for the output template, prompt and the final generation code:
```
output_template = {"ACTION": "One of the following UI actions - CLICK, TYPE, or SELECT",
                   "MARK": "A numeric id, e.g., 5, - this refers to the id of the SoM marker for the UI element on which action is to be taken",
                   "VALUE": "A string for the value of the action if it is a TYPE action, else None",
                   "COORDINATES": "location of the UI element on which action is to be taken, normalized by the image dimensions, e.g., (0.83, 0.41)"}

prompt = f"""
You are agent that can see, think and act. Imagine that you are imitating humans doing web navigation for a task step by step. 
At each stage, you can see the webpage like humans by a screenshot and know the previous actions before the current step decided by yourself through recorded history. 
You need to decide on the following action to take. 
You can click an element with the mouse, select an option, or type text with the keyboard. 
The output format should be a dictionary like: {output_template}

You are asked to complete the following task: Buy a $25 digital gift card for Tim Stebee, whose email address is scisoorbros@gmail.com. Fill in sender name Jeerimiah Waton. 
The previous actions you have taken: 

[textbox]  Recipient Name -> TYPE: Tim Stebee\n[textbox]  Recipient Email -> TYPE: scisoorbros@gmail.com

For your convinience, I have labeled the candidates with numeric marks and bounding boxes on the screenshot. 

What is the next action you would take?

Return your response as a valid JSON object in the format {output_template}
""".strip()

messages = [
    {
        "role": "user",
        "content": [{"type": "image"}, {"type": "text", "text": prompt}],
    },
]

formatted_prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

result = structured_generator(formatted_prompt, [image])

print("Result: ", result)
print("\n Code done")
```

For completeness, I will include the initial part of the code which includes the imports and the image which is loaded and passed into the model in the above snippet:
```
import json
import outlines
import outlines.generate
import outlines.generate.json
from outlines.models.transformers import transformers
from outlines.models.transformers_vision import transformers_vision, TransformersVision
from pydantic import BaseModel, Field
from typing import List, Optional

from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor
from datasets import load_dataset

import warnings
warnings.filterwarnings('ignore')

hf_dataset_name = 'MagmaAI/Magma-Mind2Web-SoM' 
path_to_store = "my_dataset_dir"

mind2Web_SoM = load_dataset(hf_dataset_name, cache_dir=path_to_store)
print("Dataset structure: ", mind2Web_SoM)
print("\nFirst sample ID: \n", mind2Web_SoM['train'][0]['id'])
print("\nFirst sample screenshot: \n", mind2Web_SoM['train'][0]['image'])
print("\nType of First sample screenshot: \n", type(mind2Web_SoM['train'][0]['image']))
print("\nFirst sample user conversation: \n", mind2Web_SoM['train'][0]['conversations'][0])
print("\nFirst sample assistant conversation: \n", mind2Web_SoM['train'][0]['conversations'][1])
image = mind2Web_SoM['train'][0]['image']
```

So we are using the first example from the [SoM annotated Mind2Web Dataset](https://huggingface.co/datasets/MagmaAI/Magma-Mind2Web-SoM) above. 

This code needs `outlines` library to run. However, we get some errors from the `outlines` library now. The first error that came was:
```
TypeError: MagmaProcessor.__call__() got an unexpected keyword argument 'text'
```

Another related error is with the keyword argument `'image'` from `MagmaProcessor`.

I fixed this by going into the `outlines.models.transformers_vision.py` file and doing the following in the `generate` method of the `TransformersVision` class - 
```
# inputs = self.processor(
#     text=prompts, images=media, padding=True, return_tensors="pt"
# ).to(self.model.device)

inputs = self.processor(
prompts, media, padding=True, return_tensors="pt"
).to(self.model.device)
```

So, basically, we comment the original lines and replace it with the uncommented lines above. At this point, we get the following error message:
```
File "/home/fte5/.cache/huggingface/modules/transformers_modules/microsoft/Magma-8B/b33355b3cffebdf9d8e60207f30a2cb1193b55c0/modeling_magma.py", line 619, in forward
    image_num_patches = [(imsize[imsize.sum(1) > 0,0] * imsize[imsize.sum(1) > 0,1]).tolist() for imsize in image_sizes]
  File "/home/fte5/.cache/huggingface/modules/transformers_modules/microsoft/Magma-8B/b33355b3cffebdf9d8e60207f30a2cb1193b55c0/modeling_magma.py", line 619, in <listcomp>
    image_num_patches = [(imsize[imsize.sum(1) > 0,0] * imsize[imsize.sum(1) > 0,1]).tolist() for imsize in image_sizes]
IndexError: Dimension out of range (expected to be in range of [-1, 0], but got 1)
```

After going through the `modeling_magma.py`, `processing_magma.py`, and `image_processing_magma.py` files, I found that the above error was coming because we are trying to run the magma model on a single image/instance. So, to get around that, and to keep using the `outlines` library in a consistent manner, I add the following lines in the `generate` method of the `TransformersVision` class - 
```
if len(inputs['pixel_values'].shape) == 4:
    inputs['pixel_values'] = inputs['pixel_values'].unsqueeze(0)
if len(inputs['image_sizes'].shape) == 2:
    inputs['image_sizes'] = inputs['image_sizes'].unsqueeze(0)
```

These lines are added just after the lines we added earlier to the same file as mentioned above.

Now, the code runs but there is a `RunTimeError` that says there is a data mismatch:
```
RuntimeError: Input type (float) and bias type (c10::BFloat16) should be the same
```

To address this, I modify the `inputs = self.processor(prompts, media, padding=True, return_tensors="pt").to(self.model.device)` line in the `generate` method of the `TransformersVision` class to: `inputs = self.processor(prompts, media, padding=True, return_tensors="pt").to(self.model.device).to(self.model.dtype)`. With this modification, the above error is resolved. 

But here is where we run into the error being talked about in this issue:
```
File "/home/fte5/.cache/huggingface/modules/transformers_modules/microsoft/Magma-8B/b33355b3cffebdf9d8e60207f30a2cb1193b55c0/modeling_magma.py", line 675, in forward
    inputs_embeds, attention_mask, position_ids, labels = self._merge_input_ids_with_image_features(
  File "/home/fte5/.cache/huggingface/modules/transformers_modules/microsoft/Magma-8B/b33355b3cffebdf9d8e60207f30a2cb1193b55c0/modeling_magma.py", line 453, in _merge_input_ids_with_image_features
    raise ValueError(
ValueError: Number of image tokens in input_ids (0) different from num_images (1).
```

The line numbers is 675 instead of 674 because I added an additional print statement in the `modeling_magma.py` file above.

At this point, I followed your suggestions in the comment above: [transformers version](https://github.com/microsoft/Magma/issues/77#issuecomment-2887829394) and changed the `transformers` library version in the `magma` environment. First I tried with `transformers==4.49.0` and then with the version from [https://github.com/jwyang/transformers.git@dev/jwyang-v4.48.2](https://github.com/jwyang/transformers.git@dev/jwyang-v4.48.2). In both the cases, I get the above error. It is coming from the cached files of magma model. 

Surprisingly enough, the inference codes are working fine with all the following transformer versions: the default one you get when setting up the magma environment first (4.51.3), the versions mentioned in the above comment (4.49.0 and the custom version 4.48.2 from [https://github.com/jwyang/transformers.git@dev/jwyang-v4.48.2](https://github.com/jwyang/transformers.git@dev/jwyang-v4.48.2). 

The inference code also works fine with the image from the [SoM annotated Mind2Web Dataset](https://huggingface.co/datasets/MagmaAI/Magma-Mind2Web-SoM) used in the above comment. As mentioned, using the output template in the prompts still does not make any difference in the output.

I realized that I had not included separate system and user prompts in the earlier code. After that change, the above code also worked fine and generated structured outputs as required. However, there is a variability in the output now. I am fixing it by looking further into the `outlines` library.

Added `np.random.seed(0)` at the beginning, and `sampler=outlines.samplers.greedy()` to `outlines.generate.json()` to get deterministic output. 

files changed inside libraries: 
1. `outlines.models.transformers_vision`: major changes: `lines: 46-80`

2. `/home/fte5/.cache/huggingface/modules/transformers_modules/microsoft/Magma-8B/b33355b3cffebdf9d8e60207f30a2cb1193b55c0/modeling_magma.py`: print statements added at `lines: 675-681` and `line: 619`.

changes to `eval_dataset.py`:
1. split conversations[0] field into the common system_prompt and the different user_prompt parts. this can be used as a mapping function to the MagmaAI/Magma-Mind2Web-SoM dataset to feed it to the model for finetuning, inference, etc.

2. added np.random.seed(0) at the beginning, and sampler=outlines.samplers.greedy() to outlines.generate.json() to get deterministic output. you can use this for OS-Atlas too.

jupyter notebook for inference on MagmaAI/Magma-Mind2Web-SoM: inference_mind2web_example.ipynb

things to do:

we need the SoM parameters for the Mind2Web dataset that the Magma people used. we will use these parameters to map the Mind2Web images to SoM marked images. we also need the mapping between mark and type, or at least a way to convert between the ground truths of these datasets. SoM will give us marks and the corresponding box coordinates so the mapping between marks and coordinates is with us. to be consistent with the mapping between mark and type we need the SoM parameters used by the magma people, because type is only present in the dataset ground truth. I don't think it is part of the SoM function output. too many or too few boxes might lead to false predictions during inference.
we need to compare the normal Mind2Web dataset ground truths to the SoM annotated Mind2Web dataset ground truths. this will tell us what output format to apply to the magma outputs while finetuning on each dataset. 
is the outlines library the only way to ensure structured output? it changes the sampling strategy. so is it a good thing to use it during finetuning? what sampling strategy should we use during finetuning and evaluation? what do the magma people use?
