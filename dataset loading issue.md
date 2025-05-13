Issue with loading the data - running `load_dataset_builder("MagmaAI/Magma-AITW-SoM")` gives the error:

```
File "/home/io452/miniconda3/envs/magma/lib/python3.10/site-packages/fsspec/spec.py", line 611, in glob
    pattern = glob_translate(path + ("/" if ends_with_sep else ""))
  File "/home/io452/miniconda3/envs/magma/lib/python3.10/site-packages/fsspec/utils.py", line 731, in glob_translate
    raise ValueError(
ValueError: Invalid pattern: '**' can only be an entire path component
```

current magma environment contains the following: `datasets==2.14.4` and `fsspec==2025.3.2`

It was recommended to update the `datasets` library as a solution to [a similar problem](https://stackoverflow.com/questions/77671277/valueerror-invalid-pattern-can-only-be-an-entire-path-component) arising while loading another dataset. Yet another solution was to downgrade the `fsspec` library instead. So I went with the first option.

after applying the updates, the magma environment now contains `datasets==3.6.0` and `fsspec==2025.3.0`

However, compatibility issues may arise between the present version of `fsspec` and other existing libraries. Need to check those. I do not see compatibility issues yet.

I switched to downloading the `osunlp/Mind2Web` dataset I got `SSLError`. This is a network issue and got resolved after switching the network.
