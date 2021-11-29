# DOTA Dataset Preparation Instructions

Install the requirements:

``` python
pip install -r requirements.txt
```

## Split Raw Images into Patches

Run the `split_dota.py` script 

``` python
python split_dota.py --srcpath <path> --dstpath <path> --dota-version <1.0/1.5> --patchsize 1024 --overlap 200
```

Run with `--help` for information about script arguments:

``` sh
python split_dota.py --help
    usage: split_dota.py [-h] --srcpath SRCPATH --dstpath DSTPATH --dota-version {1.0,1.5} [--patchsize PATCHSIZE] [--overlap OVERLAP]

    Split the DOTA 1.0/1.5 raw data into image patches of the given size.

    options:
    -h, --help            show this help message and exit
    --srcpath SRCPATH     DOTA 1.0/1.5 dataset raw data directory (source). (default: None)
    --dstpath DSTPATH     DOTA 1.0/1.5 dataset split data directory (destination). (default: None)
    --dota-version {1.0,1.5}
                            DOTA dataset version, can be one of [1.0, 1.5]. (default: None)
    --patchsize PATCHSIZE
                            Patchsize of each image patch. (default: 1024)
    --overlap OVERLAP     Overlap between image patches. (default: 200)
```

## Acknowledgements

Most of the files here are borrowed from [AerialDetection/DOTA_devkit](https://github.com/dingjiansw101/AerialDetection/tree/master/DOTA_devkit). The `split_dota.py` is a rewritten version of their `prepare_dota1.py` script which now also allows for flexible patchsizes (`--patchsize`), overlap (`--overlap`), selection of DOTA version (`--dota-version`), and some additional error checks to make the script more userfriendly.

