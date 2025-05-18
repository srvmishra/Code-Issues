# Code-Issues
This repository includes the issues I faced during cloning and using some code repositories and their solutions.

## Magma: A Foundation Model for Multimodal AI Agents
- Project page: https://microsoft.github.io/Magma/
- Arxiv Link: https://www.arxiv.org/pdf/2502.13130
- GitHub Code: https://github.com/microsoft/Magma
  
### 1. Environment Creation Issue due to `pyav` version mismatch: https://github.com/microsoft/Magma/issues/76
  > 
  > Following the comment: https://github.com/microsoft/Magma/issues/76#issuecomment-2867103221, I tried creating the environment again and I was able to do it
  > 
  > The issues I faced and how I resolved them are in the file: [Magma-issues.md](https://github.com/srvmishra/Code-Issues/blob/main/Magma-issues.md)
  >
  > This issue seems to be resolved because the environment is created and after that, I ran the [Inference with bitsandbytes
](https://github.com/microsoft/Magma?tab=readme-ov-file#inference-with-bitsandbytes) code and it ran perfectly. Some warnings were there but the code output was fine. 

 ### 2. `gradio_client` issue: while running `python agents/ui_agent/app.py`
  >
  > The things that I have tried are given in the file: [gradio client issues](https://github.com/srvmishra/Code-Issues/blob/main/gradio%20client%20issue.md)
  >
  > This issue seems to be persisting and there are some things that I have not tried yet. I have listed them in the above file. Will look into it later when making some demos of my own.
 
 ### 3. Dataset loading issue: while using `load_dataset()` from `datasets`
  > 
  > The error that I was getting was `ValueError: Invalid pattern: '**' can only be an entire path component`.
  >
  > Current magma environment contains the following: `datasets==2.14.4` and `fsspec==2025.3.2`. Upon upgrading the `datasets` library using `pip install -U datasets`, the error got resolved. We need the latest version of datasets. The command installs `datasets==3.6.0` and `fsspec==2025.3.0`. This is documented in the file [dataset loading issue.md](https://github.com/srvmishra/Code-Issues/blob/main/dataset%20loading%20issue.md).

 ### 4. `TypeError` during running the inference with bits and bytes code
  >
  > The code for inference with bits and bytes is given at: [inference with bits and bytes](https://github.com/microsoft/Magma?tab=readme-ov-file#inference-with-bitsandbytes).
  >
  > I cloned the magma repository, created the enviroment, and copied the code from above into a file within the cloned repository. I ran the file.
  >
  > The error is documented in the file [inference code issue.md](https://github.com/srvmishra/Code-Issues/blob/main/inference%20code%20issue.md). I was not able to fix the issue but restarting the system fixed the issue without making any changes to the code. The same error is raised in the issue [Issue#77](https://github.com/microsoft/Magma/issues/77) which is currently open but it arises from a different file.

 ### 5. Getting Structured Output from Magma model
  >
  > First I posted the issue [in this comment](https://github.com/microsoft/Magma/issues/77#issuecomment-2888326658) and then in this file [Structured Output from Magma with Outlines.md](https://github.com/srvmishra/Code-Issues/blob/main/Structured%20Output%20from%20Magma%20with%20Outlines.md). I was able to get the structured output as desired but adding the output template to the prompt made no difference in the generation side.
  >
  > The problem is we cannot use the `outlines` library for finetuning because it changes the way tokens are sampled from the model. At least I was not able to find a way to finetune VLMs with `outlines`.
