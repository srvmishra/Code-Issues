# Code-Issues
This repository includes the issues I faced during cloning and using some code repositories and their solutions.

## Magma: A Foundation Model for Multimodal AI Agents
- Project page: https://microsoft.github.io/Magma/
- Arxiv Link: https://www.arxiv.org/pdf/2502.13130
- GitHub Code: https://github.com/microsoft/Magma
  
  > **Environment Creation Issue due to `pyav` version mismatch**: https://github.com/microsoft/Magma/issues/76
  > 
  > Following the comment: https://github.com/microsoft/Magma/issues/76#issuecomment-2867103221, I tried creating the environment again and I was able to do it
  > 
  > The issues I faced and how I resolved them are in the file: [Magma-issues.md](https://github.com/srvmishra/Code-Issues/blob/main/Magma-issues.md)
  >
  > This issue seems to be resolved because the environment is created. And after that, I ran the [Inference with bitsandbytes
](https://github.com/microsoft/Magma?tab=readme-ov-file#inference-with-bitsandbytes) code and it ran perfectly. Some warnings were there but the code output was fine. 

  > **`gradio_client` issue**: while running `python agents/ui_agent/app.py`
  >
  > The things that I have tried are given in the file: [gradio client issues]()
