# Synthehicle Data

## License

All of our scripts in this repository are under MIT license. Data was generated using CARLA, and Synthehicle data is distributed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). 

## Download

If you want to download parts of the Synthehicle dataset, please follow the instructions of the respective subsections. 

### Tracking Data

The following should cover most use cases (2D and 3D detection, re-identification and tracking). Includes `*.mp4` for training and test scenes, as well as ground truth (MOT and COCO formats) for training scenes. Also includes `segmap` and `seginfo` files and camera calibrations.

    https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\synthehicle_core.tar.gz

Mirror

    https://drive.google.com/file/d/1JkI1mhcQUmeQ6pudsvk7ADG0IEF3SVsv/view?usp=sharing

### Segmentations

Includes ground truth instance and semantic segmentations (1920x1080) for training scenes.

    https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\synthehicle_segmentations.tar.gzaa
    https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\synthehicle_segmentations.tar.gzab
    https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\synthehicle_segmentations.tar.gzac
    https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\synthehicle_segmentations.tar.gzad

Extract the data via 

```bash
cat synthehicle_segmentations.tar.* | tar -xzvf -
```

### Depth Images

Includes ground truth depth images (1920x1080) for training scenes.

    https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\synthehicle_depth.tar.gzaa
    https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\synthehicle_depth.tar.gzab
    https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=18e2eac4&path=Datenbanken\Synthehicle\synthehicle_depth.tar.gzac

Extract the data via 

```bash
cat synthehicle_depth.tar.* | tar -xzvf -
```
