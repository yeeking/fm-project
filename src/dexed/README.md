# Various hacks of dexed 

## Hacked version of dexed for rendering patches

This one renders a dataset of WAV files by loading MIDI cartridge files into dexed and playing notesd on every patch.

How to build my hacky version of dexed that just renders a folder of carts to WAVs:

First build it:

```
git clone https://github.com/asb2m10/dexed.git
cp -r dexed-patch-dataset/* ./dexed/
cd dexed
git submodule update --init --recursive
cmake -B build .
cmake --build build --config Release -j 12
```

Then run it:
```
/build/Source/Dexed_artefacts/Standalone/Dexed 
```



