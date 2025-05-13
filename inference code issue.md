I ran the `inference with bitsandbytes` and `som example` files that I had created. Only the `som example` ran fine while the `inference with bitsandbytes` gave the following error:

```
File "/home/io452/.cache/huggingface/modules/transformers_modules/microsoft/Magma-8B/ee95aa930708b9991562153ec419b64a25e33024/modeling_magma.py", line 674, in forward
    inputs_embeds, attention_mask, position_ids, labels = self._merge_input_ids_with_image_features(
  File "/home/io452/.cache/huggingface/modules/transformers_modules/microsoft/Magma-8B/ee95aa930708b9991562153ec419b64a25e33024/modeling_magma.py", line 448, in _merge_input_ids_with_image_features
    num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
TypeError: sum() received an invalid combination of arguments - got (bool, dim=int), but expected one of:
 * (Tensor input, *, torch.dtype dtype)
 * (Tensor input, tuple of ints dim, bool keepdim, *, torch.dtype dtype, Tensor out)
 * (Tensor input, tuple of names dim, bool keepdim, *, torch.dtype dtype, Tensor out)
 ```

 I closed the terminals and opened a new one. Upon running the inference code, I get the following error:

 ```
   File "/home/io452/miniconda3/envs/magma/lib/python3.10/site-packages/transformers/utils/hub.py", line 517, in cached_files
    raise EnvironmentError(
OSError: microsoft/Magma-8B does not appear to have a file named model-00001-of-00004.safetensors. Checkout 'https://huggingface.co/microsoft/Magma-8B/tree/main'for available files.
 ```

The second error was gone when I shifted the network connection while the first error persisted. The first error is coming from the cached files coming from huggingface downloads - I cleared the cache and ran the code again. It still persisted. I could not fix the error. But upon restarting the system next morning, the error was gone. I am running Ubuntu from WSL server in a Windows system.
